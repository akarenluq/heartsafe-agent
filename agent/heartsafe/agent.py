"""HeartSafe agent: sits between the monitoring platform's alerts and the clinical team.

For every platform alert it runs its tools (data-quality check, scale-change memory
with nurse verification, ECG reliability, MRI reliability), decides accept / repeat measurement / escalate
with a deterministic policy from config.yaml, writes an explainable review, and
opens or updates a tracked case. Missed readings create outreach tasks. After a
clinician resolves a clinical case, the patient loop follows the instruction to the
patient: confirmation, action (with caregiver or community health worker follow-up)
and a check of the readings afterwards, reopening a case if they did not improve.
The clinical decision always stays with the clinical team.

`run_baseline` is the comparison: the monitoring platform alone, every alert goes to the nurse
queue with no reliability check and no follow-up tracking.
"""
import numpy as np
import pandas as pd

from .clinicians import followup_times, patient_response
from .monitoring import OPTIONAL, SYMPTOMS, patient_limits, rule_levels
from .reliability import assess_ecg, assess_mri, assess_reading
from .review import build_review
from .signals import ecg_output, mri_output

SEV = {1: "caution", 2: "critical"}
RANK = {"low": 0, "caution": 1, "critical": 2}
INSTRUCTION = {"critical": "Follow the care team's urgent plan (assessment visit or transfer) and confirm when done.",
               "caution": "Follow the care team's plan (e.g. medication change), confirm, and keep daily readings."}
DATA_FAULTS = {"implausible_weight", "implausible_bp", "implausible_hr", "implausible_spo2", "unit_error",
               "bp_artifact"}
HIST_KEYS = ["day", "weight_kg", "sbp", "hr", "sob", "swelling", *OPTIONAL]


class PatientState:
    def __init__(self):
        self.hist, self.offset, self.pending_step, self.missed = [], 0.0, None, 0
        self.case_id, self.case_resolved_h = None, None
        self.mri_done = False
        self.verify = None  # open nurse verification of a possible new scale
        self.loops = []     # patient loops waiting for an outcome check or a failure


class HeartSafeAgent:
    def __init__(self, cfg, patients, tracker, rng, horizon_h, ecg_pool=None, mri_pool=None, safeguards=None,
                 truth=None, scale_truth=None):
        self.cfg, self.tr, self.rng, self.horizon = cfg, tracker, rng, horizon_h
        self.patients = patients.set_index("patient_id", drop=False).to_dict("index")
        self.ecg_pool, self.mri_pool = ecg_pool, mri_pool
        self.safeguards = [s for s in (safeguards or []) if s.get("approved")]
        # Ground truth is used ONLY by the simulated world (what a repeat ECG or MRI would show),
        # never by the agent's decisions.
        self._truth = truth or {}
        # What the nurse finds when calling about a weight jump (simulation only). None = assume
        # the new scale is confirmed (real data: the nurse records the answer in the tracker).
        self._scale_truth = scale_truth
        self.decisions = []
        self.gt_cases = []  # per case: would it have stalled without tracking (evaluation only)
        self.gt_loops = []  # per loop: would the patient have completed it with no follow-up (evaluation only)
        self._demo_loop_done = False

    # ---- helpers -------------------------------------------------------
    def _ack_window(self, patient, severity):
        h = self.cfg["followup"]["ack_deadline_h"][severity]
        for s in self.safeguards:
            if s["type"] == "tighter_ack_deadline" and str(patient.get(s["field"])) == str(s["value"]):
                h *= s["factor"]
        return h

    def _open_case(self, st, now):
        return st.case_id if st.case_id and (st.case_resolved_h is None or st.case_resolved_h > now) else None

    def _create_case(self, patient, st, now, kind, severity, review, resolve_note=None):
        ack_h = self._ack_window(patient, severity)
        res_h = self.cfg["followup"]["resolve_deadline_h"][severity]
        review["next_step"] = review["next_step"].replace("{ack}", f"{ack_h:g}")
        cid = self.tr.create(patient["patient_id"], kind, severity, now, ack_h, res_h, review)
        f = followup_times(severity, patient["remoteness"], ack_h, res_h, self.cfg, self.rng, reminders=True,
                           force_forget=patient["patient_id"] == "DEMO-MARY" and kind == "clinical")
        self.gt_cases.append({"case_id": cid, "gt_stalled": bool(f.get("forgot"))})
        for t, to in f["reminders"]:
            if now + t <= self.horizon:
                self.tr.remind(cid, now + t, to)
        who = "nurse_backup" if any(to == "backup" for _, to in f["reminders"]) else "nurse_primary"
        resolved_h = None
        self.last_ack_h = now + f["ack"]
        if resolve_note:  # a verification call: resolved as soon as the nurse reaches the patient
            f["resolve"] = f["ack"] + 1.0
        if now + f["ack"] <= self.horizon:
            self.tr.act(cid, "acknowledge", now + f["ack"], who, "clinician")
            if now + f["ack"] + 0.5 <= self.horizon:
                self.tr.act(cid, "review", now + f["ack"] + 0.5, who, "clinician")
            if now + f["resolve"] <= self.horizon:
                self.tr.act(cid, "resolve", now + f["resolve"], who, "clinician",
                            note=resolve_note or "Patient contacted; plan documented.")
                resolved_h = now + f["resolve"]
        if kind == "clinical":
            st.case_id, st.case_resolved_h = cid, resolved_h
            if resolved_h is not None:
                self._start_loop(patient, st, cid, severity, resolved_h)
        return cid, resolved_h

    # ---- patient loop --------------------------------------------------
    def _confirm_window(self, patient, severity):
        h = self.cfg["patient_loop"]["confirm_deadline_h"][severity]
        for s in self.safeguards:
            if s["type"] == "earlier_patient_followup" and str(patient.get(s["field"])) == str(s["value"]):
                h *= s["factor"]
        return h

    def _start_loop(self, patient, st, cid, severity, t0):
        """Send the clinician's instruction and simulate the patient side's response."""
        if severity not in self.cfg["patient_loop"]["confirm_deadline_h"]:
            return
        win = self._confirm_window(patient, severity)
        self.tr.start_loop(cid, t0, INSTRUCTION[severity], win)
        demo = patient["patient_id"] == "DEMO-MARY" and not self._demo_loop_done
        self._demo_loop_done = self._demo_loop_done or demo
        r = patient_response(severity, patient, win, self.cfg, self.rng, force_no_response=demo)
        self.gt_loops.append({"case_id": cid, "gt_self_completed": bool(r["gt_self_completed"])})
        H = self.horizon
        for dt, to in r["contacts"]:
            if t0 + dt <= H and (r["confirm"] is None or dt <= r["confirm"] or to == r["by"]):
                self.tr.contact(cid, t0 + dt, to)
        if r["confirm"] is not None and t0 + r["confirm"] <= H:
            self.tr.patient_step(cid, "confirm", t0 + r["confirm"], r["by"], r["by"])
            if r["action"] is not None and t0 + r["action"] <= H:
                self.tr.patient_step(cid, "action_done", t0 + r["action"], r["by"], r["by"])
        loop = {"case_id": cid, "severity": severity, "check_h": None, "fail_h": None, "reason": None}
        if r["fail"] is not None:
            loop["fail_h"], loop["reason"] = t0 + r["fail"], r["fail_reason"]
        elif r["action"] is not None:
            loop["check_h"] = t0 + r["action"] + self.cfg["patient_loop"]["outcome_check_after_h"]
        st.loops.append(loop)

    def _reopen(self, p, st, row, now, loop, reason):
        if loop["fail_h"] is not None:
            # the instruction did not reach the patient or was not done: a follow-up task for the nurse
            review = build_review(p, row.day, [f"patient loop: {reason} (case {loop['case_id']})"], st.hist, None,
                                  loop["severity"], ["patient_loop_reopened"], [], kind="patient_followup",
                                  ack_h="{ack}")
            new_id, _ = self._create_case(p, st, now, "patient_followup", loop["severity"], review)
            self.tr.close_loop(loop["case_id"], now, reason, reopened_case_id=new_id)
            return self._record(row, "loop_reopened", case_id=new_id, flags="patient_not_reached_or_not_done",
                                notes=reason)
        open_id = self._open_case(st, now)
        if open_id:
            self.tr.add_evidence(open_id, now, {"loop_of_case": loop["case_id"], "reason": reason})
            new_id = open_id
        else:
            review = build_review(p, row.day, [f"patient loop: {reason}"], st.hist, None, loop["severity"],
                                  ["patient_loop_reopened"], [], ack_h="{ack}")
            new_id, _ = self._create_case(p, st, now, "clinical", loop["severity"], review)
        self.tr.close_loop(loop["case_id"], now, reason, reopened_case_id=new_id)
        self._record(row, "loop_reopened", case_id=new_id, flags="patient_loop_reopened", notes=reason)

    def _check_loops(self, p, st, row, now, today=None):
        """Failures fire on time; the outcome check waits for the first reading after it is due."""
        keep = []
        for loop in st.loops:
            if loop["fail_h"] is not None and now >= loop["fail_h"]:
                self._reopen(p, st, row, now, loop, loop["reason"])
            elif loop["check_h"] is not None and now >= loop["check_h"] and today is not None:
                level, fired = self._corrected_level(st, today, p)
                if level == 0:
                    self.tr.close_loop(loop["case_id"], now, "readings improved after the action")
                    self._record(row, "loop_closed", case_id=loop["case_id"])
                else:
                    self._reopen(p, st, row, now, loop, "readings did not improve after the action ("
                                 + "; ".join(fired) + ")")
            else:
                keep.append(loop)
        st.loops = keep

    def _record(self, row, decision, **kw):
        self.decisions.append({"patient_id": row.patient_id, "day": row.day, "alert_level": row.alert_level,
                               "alert_rules": row.alert_rules, "decision": decision, **kw})

    @staticmethod
    def _today(row, offset=0.0):
        t = {"day": row.day, "weight_kg": row.weight_kg - offset, "raw_weight_kg": row.weight_kg,
             "sbp": row.sbp, "hr": row.hr, "sob": bool(row.sob), "swelling": bool(row.swelling)}
        for c, default in OPTIONAL.items():
            v = getattr(row, c, default)
            t[c] = bool(v) if isinstance(default, bool) else float(v)
        return t

    def _corrected_level(self, st, today, p=None):
        days = np.arange(today["day"] - 14, today["day"] + 1)
        num = ["weight_kg", "sbp", "hr"] + [c for c, v in OPTIONAL.items() if not isinstance(v, bool)]
        arr = {k: np.full(len(days), np.nan) for k in num}
        arr.update({k: np.full(len(days), OPTIONAL.get(k, False)) for k in ("sob", "swelling", *[
            c for c, v in OPTIONAL.items() if isinstance(v, bool)])})
        for h in st.hist + [today]:
            i = h["day"] - days[0]
            if 0 <= i < len(days):
                for k in arr:
                    arr[k][i] = h.get(k, arr[k][i])
        p = p or {}
        extra = {k: arr[k] for k in OPTIONAL}
        lv, fired = rule_levels(arr["weight_kg"], arr["sbp"], arr["hr"], arr["sob"].astype(bool),
                                arr["swelling"].astype(bool), self.cfg["alert_rules"], extra,
                                patient_limits(self.cfg["alert_rules"], p), p.get("creatinine_baseline_umol"))
        return int(lv[-1]), fired[-1]

    def _check_verification(self, p, st, row, now):
        """The nurse has called about a weight jump: re-baseline (new scale) or escalate (real gain)."""
        v = st.verify
        if v is None or now < v["due_h"]:
            return
        st.verify = None
        if v["is_scale"]:
            st.offset += v["step"]
            for h in st.hist:
                if h["day"] >= v["step_day"]:
                    h["weight_kg"] -= v["step"]
            self._record(row, "scale_verified", case_id=v["case_id"], flags="device_change_confirmed",
                         notes=f"nurse confirmed a new scale ({v['step']:+.1f} kg); readings re-baselined")
        else:
            review = build_review(p, row.day, ["nurse call: the weight gain is real, not a new scale"], st.hist,
                                  None, "caution", ["scale_check_rejected"], [], ack_h="{ack}")
            cid, _ = self._create_case(p, st, now, "clinical", "caution", review)
            self._record(row, "escalate", severity="caution", case_id=cid, flags="scale_check_rejected")

    # ---- main loop -----------------------------------------------------
    def run(self, feed):
        r = self.cfg["reliability"]
        for pid, g in feed.groupby("patient_id", sort=False):
            p, st = self.patients[pid], PatientState()
            for row in g.sort_values("day").itertuples():
                now = row.day * 24 + 9.0
                self._check_verification(p, st, row, now)
                if not row.submitted:
                    self._check_loops(p, st, row, now)
                    st.missed += 1
                    if st.missed == r["missing_outreach_days"]:
                        review = build_review(p, row.day, [f"no readings for {st.missed} days"], st.hist, None,
                                              "low", ["missed_readings"], [], kind="outreach", ack_h="{ack}")
                        cid, _ = self._create_case(p, st, now, "outreach", "low", review)
                        self._record(row, "outreach", severity="low", case_id=cid, flags="missed_readings")
                    continue
                st.missed = 0
                today = self._today(row, st.offset)
                self._check_loops(p, st, row, now, today)
                if row.alert_level > 0:
                    self._handle_alert(p, st, row, today, now)
                st.hist.append({k: today[k] for k in HIST_KEYS})
                st.hist = st.hist[-14:]
        return pd.DataFrame(self.decisions)

    def _handle_alert(self, p, st, row, today, now):
        r = self.cfg["reliability"]
        flags, notes = assess_reading(st.hist, today, self.cfg, 0.0)

        # 1. one-off measurement faults -> ask for a repeat, no case
        if DATA_FAULTS & set(flags):
            return self._record(row, "repeat", flags=";".join(flags), notes=" | ".join(notes))

        # 2. scale-change memory: a jump that persists with no symptoms may be a new scale. A nurse
        #    verifies it before the baseline changes; until then weight-only alerts wait in that case.
        symptoms = any(today.get(s) for s in SYMPTOMS)
        if st.verify is not None:
            _, fired_now = self._corrected_level(st, today, p)
            if not symptoms and all(f.startswith("weight") for f in fired_now):
                self.tr.add_evidence(st.verify["case_id"], now, {"day": row.day, "rules": fired_now, "flags": flags})
                return self._record(row, "nurse_verify_update", case_id=st.verify["case_id"],
                                    flags=";".join(flags) or "awaiting_nurse_verification",
                                    notes="weight alert held until the nurse verifies the possible new scale")
            # symptoms or any non-weight rule: real risk does not wait for the verification
        elif st.pending_step is not None:
            step_day, step, pre = st.pending_step
            st.pending_step = None
            if abs((today["weight_kg"] - pre) - step) <= r["step_persist_tolerance_kg"]:
                notes.append(f"weight {step:+.1f} kg since day {step_day} persisted on repeat; "
                             "nurse to confirm a new scale before re-baselining")
                review = build_review(p, row.day, ["weight jump persisted on repeat: possible new scale"], st.hist,
                                      today, "caution", flags + ["scale_check"], notes, kind="scale_check",
                                      ack_h="{ack}")
                is_scale = True if self._scale_truth is None else (p["patient_id"], row.day) in self._scale_truth
                note = ("Nurse called patient: new scale confirmed." if is_scale else
                        "Nurse called patient: same scale, weight gain is real; escalated.")
                cid, _ = self._create_case(p, st, now, "scale_check", "caution", review, resolve_note=note)
                st.verify = {"case_id": cid, "due_h": self.last_ack_h, "step": step, "step_day": step_day,
                             "is_scale": is_scale}
                return self._record(row, "nurse_verify", case_id=cid, flags=";".join(flags + ["scale_check"]),
                                    notes=" | ".join(notes))
        elif "step_change" in flags:
            pre = float(np.median([h["weight_kg"] for h in st.hist[-3:]]))
            st.pending_step = (row.day, today["weight_kg"] - pre, pre)
            return self._record(row, "repeat", flags=";".join(flags), notes=" | ".join(notes))

        # 3. does the alert still stand on corrected data?
        level, fired = self._corrected_level(st, today, p)
        if level == 0:
            return self._record(row, "filtered", flags=";".join(flags) or "corrected_data_normal",
                                notes=" | ".join(notes) or "alert not supported after data-quality correction")

        # 4. evidence from the ECG device, if the patient has one
        ecg, ecg_log = None, {}
        if p["ecg_kit"]:
            truth = self._truth.get((p["patient_id"], row.day), False)
            out, gt = ecg_output(p, truth, self.cfg, self.rng, self.ecg_pool,
                                 demo=p["patient_id"] == "DEMO-MARY" and truth)
            ecg = assess_ecg(out, self.cfg)
            ecg_log = {"ecg_checked": True, "ecg_reliable": ecg["reliable"], "ecg_action": ecg["action"],
                       "ecg_abnormal": ecg["abnormal"], "gt_ecg_fault": gt.get("fault", "none")}
            if not ecg["reliable"]:
                flags.append("ecg_unreliable")
        severity = SEV[level]
        if severity == "caution" and ecg and ecg["reliable"] and ecg["abnormal"] and symptoms:
            severity = "critical"
            fired = fired + ["reliable abnormal ECG with symptoms"]

        # 5. escalate: update the open case or open a new one
        open_id = self._open_case(st, now)
        if open_id:
            self.tr.add_evidence(open_id, now, {"day": row.day, "rules": fired, "flags": flags})
            case = self.tr.get(open_id)
            if RANK[severity] > RANK[case["severity"]] and case["state"] == "created":
                self.tr.escalate_severity(open_id, now, severity, now + self._ack_window(p, severity))
            return self._record(row, "escalate_update", severity=severity, case_id=open_id, flags=";".join(flags),
                                **ecg_log)

        review = build_review(p, row.day, fired, st.hist, today, severity, flags, notes, ecg=ecg, ack_h="{ack}")
        cid, resolved_h = self._create_case(p, st, now, "clinical", severity, review)
        self._record(row, "escalate", severity=severity, case_id=cid, flags=";".join(flags), **ecg_log)

        # 6. imaging follow-up for some critical cases: MRI results get their own reliability check
        demo = p["patient_id"] == "DEMO-MARY" and not st.mri_done
        if severity == "critical" and resolved_h and (
                demo or self.rng.random() < self.cfg["simulation"]["mri_referral_prob"]):
            st.mri_done = True
            t_mri = resolved_h + 72
            if t_mri <= self.horizon:
                truth = self._truth.get((p["patient_id"], row.day), False)
                out, gt = mri_output(p, truth, self.cfg, self.rng, self.mri_pool, demo=demo)
                mri = assess_mri(out)
                mri_log = {"patient_id": p["patient_id"], "day": int(t_mri // 24), "alert_level": 0, "alert_rules": "",
                           "flags": ";".join(mri["flags"]), "mri_reliable": mri["reliable"],
                           "mri_scanner": mri["scanner"], "gt_mri_fault": gt["fault"],
                           "gt_mri_unreliable": gt["unreliable"], "gt_mri_regional": gt["regional"]}
                if not mri["reliable"] or demo:
                    rv = build_review(p, int(t_mri // 24), ["cardiac MRI result returned"], st.hist, None,
                                      "caution", mri["flags"], [], mri=mri, kind="imaging", ack_h="{ack}")
                    mcid, _ = self._create_case(p, st, t_mri, "imaging", "caution", rv)
                    self.decisions.append({**mri_log, "decision": "imaging_review", "severity": "caution",
                                           "case_id": mcid})
                else:  # accepted (possibly with a caveat); documented, no case
                    self.decisions.append({**mri_log, "decision": "imaging_accept"})


def run_baseline(feed, patients, cfg, rng, horizon_h):
    """Platform alone: each alert episode goes to the nurse queue; no reliability check, no tracking."""
    pats = patients.set_index("patient_id").to_dict("index")
    rows = []
    for pid, g in feed[feed.alert_level > 0].groupby("patient_id", sort=False):
        busy_until = -1.0
        for row in g.sort_values("day").itertuples():
            now = row.day * 24 + 9.0
            if now < busy_until:
                continue
            sev = SEV[row.alert_level]
            f = followup_times(sev, pats[pid]["remoteness"], cfg["followup"]["ack_deadline_h"][sev],
                               cfg["followup"]["resolve_deadline_h"][sev], cfg, rng, reminders=False)
            ack = now + f["ack"] if f["ack"] is not None else None
            res = now + f["resolve"] if f["resolve"] is not None else None
            busy_until = res if res is not None else now + 24  # a missed alert: the next day's alert gets a fresh chance
            rows.append({"patient_id": pid, "day": row.day, "severity": sev, "created_h": now,
                         "acknowledged_h": ack if ack is not None and ack <= horizon_h else None,
                         "resolved_h": res if res is not None and res <= horizon_h else None,
                         "lost": f["lost"]})
    return pd.DataFrame(rows)
