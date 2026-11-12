"""HeartSafe agent: sits between the monitoring platform's alerts and the clinical team.

For every platform alert it runs its tools (data-quality check, scale-change memory
with nurse verification, ECG reliability, MRI reliability), decides accept / repeat measurement / escalate
with a deterministic policy from config.yaml, writes an explainable review, and
opens or updates a tracked case. Missed readings create outreach tasks. After a
clinician resolves a clinical case, the patient loop follows the instruction to the
patient: confirmation, action (with caregiver or community health worker follow-up)
and a check of the readings afterwards, reopening a case if they did not improve.
Every ECG the patient records is checked, not only those on alert days. A reliable ECG is
also named: new atrial fibrillation (critical with symptoms, caution without), atrial
fibrillation already in the enrollment record (known, not new), and bradycardia or tachycardia
from the ECG heart rate against the patient's own limits. The patient's sleep log is compared
with their own usual sleep; worse sleep opens a low-priority check-in call, never a clinical
alert. Chest pain or
fainting together with a reliable abnormal ECG is an emergency: the patient is told at once
"If you feel worse, call 911." and a clinician must start a video call within 15 minutes and
record "call 911" or "cancel the alarm" with a reason. The agent never calls 911 itself.
The clinical decision always stays with the clinical team.

`run_baseline` is the comparison: the monitoring platform alone, every alert goes to the nurse
queue with no reliability check and no follow-up tracking.
"""
import re

import numpy as np
import pandas as pd

from .clinicians import followup_times, patient_response
from .monitoring import OPTIONAL, SYMPTOM_TEXT, SYMPTOMS, patient_limits, rule_levels
from .reliability import assess_ecg, assess_mri, assess_reading
from .review import build_review
from .selfcare import SLEEP_SYMPTOMS
from .selfcare import summary as selfcare_summary
from .signals import ecg_output, mri_output

SEV = {1: "caution", 2: "critical"}
RANK = {"low": 0, "caution": 1, "critical": 2, "emergency": 3}
INSTRUCTION = {"critical": "Follow the care team's urgent plan (assessment visit or transfer) and confirm when done.",
               "caution": "Follow the care team's plan (e.g. medication change), confirm, and keep daily readings."}
DATA_FAULTS = {"implausible_weight", "implausible_bp", "implausible_hr", "implausible_spo2", "unit_error",
               "bp_artifact"}
HIST_KEYS = ["day", "weight_kg", "sbp", "hr", "sob", "swelling", *OPTIONAL]


def rule_key(text):
    """The rule behind an alert reason, without its numbers: 'weight +2.6 kg in 2 days' -> 'weight # kg in # days'."""
    return re.sub(r"[+-]?\d+(\.\d+)?", "#", text)


class PatientState:
    def __init__(self):
        self.hist, self.offset, self.pending_step, self.missed = [], 0.0, None, 0
        self.case_id, self.case_resolved_h = None, None
        self.mri_done = False
        self.verify = None  # open nurse verification of a possible new scale
        self.loops = []     # patient loops waiting for an outcome check or a failure
        self.af = None      # new atrial fibrillation already reported to the care team: {case_id, severity, day}
        self.selfcare = None   # today's sleep and breathing summary
        self.sleep_day = None  # day of the last sleep check-in


class HeartSafeAgent:
    def __init__(self, cfg, patients, tracker, rng, horizon_h, ecg_pool=None, mri_pool=None, safeguards=None,
                 truth=None, scale_truth=None, rhythm_truth=None):
        self.cfg, self.tr, self.rng, self.horizon = cfg, tracker, rng, horizon_h
        # Its own random stream for the simulated follow-up of sleep check-ins and new-AF cases, so
        # adding them leaves the rest of the simulated world as it was.
        self.rng_extra = rng.spawn(1)[0]
        self.patients = patients.set_index("patient_id", drop=False).to_dict("index")
        self.ecg_pool, self.mri_pool = ecg_pool, mri_pool
        self.safeguards = [s for s in (safeguards or []) if s.get("approved")]
        # Ground truth is used ONLY by the simulated world (what a repeat ECG or MRI would show),
        # never by the agent's decisions.
        self._truth = truth or {}
        self._rhythm_truth = rhythm_truth or {}  # (patient, day) -> "af" (known pattern) or "new_af"
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

    def _create_case(self, patient, st, now, kind, severity, review, resolve_note=None, trigger=None, rng=None):
        """trigger: (rule level, rules fired) that started a clinical case, for its patient loop."""
        rng = rng or self.rng
        ack_h = self._ack_window(patient, severity)
        res_h = self.cfg["followup"]["resolve_deadline_h"][severity]
        review["next_step"] = review["next_step"].replace("{ack}", f"{ack_h:g}")
        cid = self.tr.create(patient["patient_id"], kind, severity, now, ack_h, res_h, review)
        f = followup_times(severity, patient["remoteness"], ack_h, res_h, self.cfg, rng, reminders=True,
                           force_forget=patient["patient_id"] == "DEMO-MARY" and kind == "clinical")
        self.gt_cases.append({"case_id": cid, "gt_stalled": bool(f.get("forgot")),
                              "standard_ack_deadline_h": now + self.cfg["followup"]["ack_deadline_h"][severity]})
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
                self._start_loop(patient, st, cid, severity, resolved_h, trigger, rng)
        return cid, resolved_h

    # ---- patient loop --------------------------------------------------
    def _confirm_window(self, patient, severity):
        h = self.cfg["patient_loop"]["confirm_deadline_h"][severity]
        for s in self.safeguards:
            if s["type"] == "earlier_patient_followup" and str(patient.get(s["field"])) == str(s["value"]):
                h *= s["factor"]
        return h

    def _start_loop(self, patient, st, cid, severity, t0, trigger=None, rng=None):
        """Send the clinician's instruction and simulate the patient side's response."""
        if severity not in self.cfg["patient_loop"]["confirm_deadline_h"]:
            return
        win = self._confirm_window(patient, severity)
        self.tr.start_loop(cid, t0, INSTRUCTION[severity], win)
        demo = patient["patient_id"] == "DEMO-MARY" and not self._demo_loop_done
        self._demo_loop_done = self._demo_loop_done or demo
        r = patient_response(severity, patient, win, self.cfg, rng or self.rng, force_no_response=demo)
        self.gt_loops.append({"case_id": cid, "gt_self_completed": bool(r["gt_self_completed"])})
        H = self.horizon
        for dt, to in r["contacts"]:
            if t0 + dt <= H and (r["confirm"] is None or dt <= r["confirm"] or to == r["by"]):
                self.tr.contact(cid, t0 + dt, to)
        if r["confirm"] is not None and t0 + r["confirm"] <= H:
            self.tr.patient_step(cid, "confirm", t0 + r["confirm"], r["by"], r["by"])
            if r["action"] is not None and t0 + r["action"] <= H:
                self.tr.patient_step(cid, "action_done", t0 + r["action"], r["by"], r["by"])
        loop = {"case_id": cid, "severity": severity, "check_h": None, "fail_h": None, "reason": None,
                "level": trigger[0] if trigger else None,
                "rules": {rule_key(f) for f in trigger[1]} if trigger else None}
        if r["fail"] is not None:
            loop["fail_h"], loop["reason"] = t0 + r["fail"], r["fail_reason"]
        elif r["action"] is not None:
            loop["check_h"] = t0 + r["action"] + self.cfg["patient_loop"]["outcome_check_after_h"]
        st.loops.append(loop)

    @staticmethod
    def _worse(loop, level, fired):
        """Worse than when the case started: a higher alert level, or a critical alert for a new reason."""
        if loop.get("rules") is None:
            return level > 0
        return level > loop["level"] or (level == 2 and bool({rule_key(f) for f in fired} - loop["rules"]))

    @classmethod
    def _not_improved(cls, loop, level, fired):
        """The outcome check: a rule that started the case still fires, or the readings got worse."""
        if loop.get("rules") is None:
            return level > 0
        return bool({rule_key(f) for f in fired} & loop["rules"]) or cls._worse(loop, level, fired)

    def _reopen(self, p, st, row, now, loop, reason, trigger=None):
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
            new_id, _ = self._create_case(p, st, now, "clinical", loop["severity"], review, trigger=trigger)
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
                if not self._not_improved(loop, level, fired):
                    self.tr.close_loop(loop["case_id"], now, "readings improved after the action")
                    self._record(row, "loop_closed", case_id=loop["case_id"])
                else:
                    why = "readings got worse" if self._worse(loop, level, fired) else \
                        "readings did not improve"
                    self._reopen(p, st, row, now, loop, f"{why} after the action (" + "; ".join(fired) + ")",
                                 trigger=(level, fired))
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
            cid, _ = self._create_case(p, st, now, "clinical", "caution", review,
                                       trigger=(1, ["weight +0 kg in 7 days"]))  # a weight-gain case
            self._record(row, "escalate", severity="caution", case_id=cid, flags="scale_check_rejected")

    # ---- ECG: every recording is checked ----------------------------------
    def _ecg_recorded(self, p, row):
        """Did the patient record an ECG today? From the platform feed when it says so, otherwise the
        recording protocol in config.yaml (weekly, on symptom days, and when the agent asks on alert days)."""
        if not p.get("ecg_kit"):
            return False
        rec = getattr(row, "ecg_recorded", None)
        if rec is not None and not pd.isna(rec):
            return bool(rec)
        pr = self.cfg.get("ecg_protocol", {})
        every = pr.get("routine_every_days")
        return bool((every and row.day % every == 0)
                    or (pr.get("on_symptoms", True) and any(bool(getattr(row, s, False)) for s in SYMPTOMS))
                    or (pr.get("on_alert", True) and row.alert_level > 0))

    def _read_ecg(self, p, row):
        key = (p["patient_id"], row.day)
        truth = self._truth.get(key, False)
        hr = getattr(row, "hr", None)
        out, gt = ecg_output(p, truth, self.cfg, self.rng, self.ecg_pool,
                             demo=p["patient_id"] == "DEMO-MARY" and truth,
                             rhythm={"af": "af", "new_af": "af", "rate": "other"}.get(self._rhythm_truth.get(key)),
                             hr=None if hr is None or pd.isna(hr) else float(hr))
        ecg = assess_ecg(out, self.cfg)
        return ecg, {"ecg_checked": True, "ecg_reliable": ecg["reliable"], "ecg_action": ecg["action"],
                     "ecg_abnormal": ecg["abnormal"], "ecg_rhythm": ecg["rhythm"], "ecg_hr": ecg["hr"],
                     "gt_ecg_fault": gt.get("fault", "none"), "gt_ecg_rhythm": gt.get("rhythm")}

    def _name_findings(self, p, st, read):
        """Name what a reliable ECG shows. Atrial fibrillation: "known" (in the enrollment record),
        "reported" (a new-AF case was already opened for this patient) or "new". Bradycardia or
        tachycardia: the ECG heart rate outside the patient's own limits."""
        ecg, log = read
        ecg["af"] = ecg["rate"] = None
        if ecg.get("reliable"):
            if ecg.get("rhythm") == "af":
                known = p.get("atrial_fibrillation")
                known = bool(known) and not (isinstance(known, float) and np.isnan(known))
                ecg["af"] = "known" if known else "reported" if st.af else "new"
            if ecg.get("hr") is not None:
                lim = patient_limits(self.cfg["alert_rules"], p)
                if ecg["hr"] < lim["hr_low"]:
                    ecg["rate"], ecg["hr_limit"] = "bradycardia", lim["hr_low"]
                elif ecg["hr"] > lim["hr_high"]:
                    ecg["rate"], ecg["hr_limit"] = "tachycardia", lim["hr_high"]
        log.update(ecg_af=ecg["af"], ecg_rate=ecg["rate"])
        return read

    @staticmethod
    def _rate_text(ecg):
        return f"{ecg['rate']} on the ECG ({ecg['hr']:.0f} bpm, patient's limit {ecg['hr_limit']:g})"

    @staticmethod
    def _af_to_report(st, ecg, symptoms):
        """New atrial fibrillation the care team has not been told about, or was told about without
        symptoms and the patient now reports one."""
        if ecg is None or not ecg.get("af") or ecg["af"] == "known":
            return False
        return ecg["af"] == "new" or bool(symptoms and st.af and st.af["severity"] != "critical")

    def _add_to_open_case(self, p, st, open_id, row, now, level, fired, flags, severity):
        """More evidence for a case the clinician has not closed; raise its severity when it is worse."""
        self.tr.add_evidence(open_id, now, {"day": row.day, "rules": fired, "flags": flags})
        for loop in st.loops:  # the outcome check also covers rules added while the case was open
            if loop["case_id"] == open_id and loop["rules"] is not None:
                loop["rules"] |= {rule_key(f) for f in fired}
                loop["level"] = max(loop["level"], level)
        case = self.tr.get(open_id)
        if RANK[severity] > RANK[case["severity"]] and case["state"] == "created":
            self.tr.escalate_severity(open_id, now, severity, now + self._ack_window(p, severity))
            for g in self.gt_cases:
                if g["case_id"] == open_id:
                    g["standard_ack_deadline_h"] = now + self.cfg["followup"]["ack_deadline_h"][severity]

    def _new_af(self, p, st, row, today, now, ecg, log, flags, notes):
        """New atrial fibrillation on a reliable ECG: critical with a symptom the same day, caution
        without. It does not wait for a weight or cuff problem to be sorted out."""
        R = self.cfg["ecg_rhythm"]
        symptoms = [s for s in SYMPTOMS if today.get(s)]
        # other rules are added only when today's readings are not in doubt (no data fault, no scale check)
        doubt = bool(DATA_FAULTS & set(flags)) or "step_change" in flags or st.verify is not None \
            or st.pending_step is not None
        level, fired = self._corrected_level(st, today, p) if row.alert_level > 0 and not doubt else (0, [])
        severity = R["new_af_with_symptoms"] if symptoms else R["new_af_without_symptoms"]
        if level and RANK[SEV[level]] > RANK[severity]:
            severity = SEV[level]
        what = "new atrial fibrillation on a reliable ECG (not in the enrollment record)"
        if symptoms:
            what += " with " + ", ".join(SYMPTOM_TEXT[s] for s in symptoms)
        fired = [what] + ([self._rate_text(ecg)] if ecg.get("rate") else []) + fired
        flags = flags + ["ecg_new_af"]
        level = max(level, RANK[severity])
        open_id = self._open_case(st, now)
        if open_id:
            self._add_to_open_case(p, st, open_id, row, now, level, fired, flags, severity)
            cid, decision = open_id, "escalate_update"
        else:
            review = build_review(p, row.day, fired, st.hist, today, severity, flags, notes, ecg=ecg, ack_h="{ack}",
                                  selfcare=st.selfcare)
            cid, _ = self._create_case(p, st, now, "clinical", severity, review, trigger=(level, fired),
                                       rng=self.rng_extra)
            decision = "escalate"
        st.af = {"case_id": cid, "severity": severity, "day": row.day}
        return self._record(row, decision if row.alert_level > 0 else "ecg_new_af", severity=severity, case_id=cid,
                            flags=";".join(flags), notes=what, **log)

    def _ecg_without_alert(self, p, st, row, today, now, ecg, log):
        """A recording on a day with no platform alert (routine or symptom day)."""
        if not ecg["reliable"]:
            return self._record(row, "ecg_repeat", flags="ecg_unreliable",
                                notes="ECG not usable (" + ", ".join(ecg["reasons"]) + "); patient asked to "
                                      "record it again", **log)
        if self._af_to_report(st, ecg, any(today.get(s) for s in SYMPTOMS)):
            return self._new_af(p, st, row, today, now, ecg, log, [], [])
        if ecg.get("rate"):
            open_id = self._open_case(st, now)
            if open_id:
                self.tr.add_evidence(open_id, now, {"day": row.day, "ecg": self._rate_text(ecg)})
            return self._record(row, "ecg_noted", case_id=open_id, flags=f"ecg_{ecg['rate']}",
                                notes=self._rate_text(ecg) + " with no alert: added to the chart for the next "
                                      "review", **log)
        if ecg.get("af"):
            why = "in the enrollment record" if ecg["af"] == "known" else "already reported to the care team"
            return self._record(row, "ecg_known_af", flags="ecg_known_af",
                                notes=f"atrial fibrillation on a reliable ECG, {why}: not a new finding", **log)
        if ecg["abnormal"]:
            open_id = self._open_case(st, now)
            if open_id:
                self.tr.add_evidence(open_id, now, {"day": row.day, "ecg": "reliable abnormal recording"})
            return self._record(row, "ecg_noted", case_id=open_id, flags="ecg_abnormal_no_alert",
                                notes="reliable abnormal ECG with no alert: added to the chart for the next "
                                      "review", **log)
        return self._record(row, "ecg_ok", **log)

    # ---- emergency -------------------------------------------------------
    def _emergency(self, p, st, row, today, now, em, ecg, ecg_log, flags, notes):
        """Open an emergency case, message the patient at once, and wait for the clinician's video call."""
        E, cfg = self.cfg["emergency"], self.cfg
        named = {"new": " (new atrial fibrillation)", "reported": " (atrial fibrillation, already reported)",
                 "known": " (atrial fibrillation, known since enrollment)"}.get(ecg.get("af"), "")
        fired = [SYMPTOM_TEXT[s] for s in em] + ["reliable abnormal ECG at the same time" + named]
        if ecg.get("rate"):
            fired.append(self._rate_text(ecg))
        ack_h = self._ack_window(p, "emergency")
        res_h = cfg["followup"]["resolve_deadline_h"]["emergency"]
        review = build_review(p, row.day, fired, st.hist, today, "emergency", flags, notes, ecg=ecg,
                              ack_h=f"{ack_h * 60:g}", selfcare=st.selfcare)
        cid = self.tr.create(p["patient_id"], "clinical", "emergency", now, ack_h, res_h, review)
        self.tr.notify_patient(cid, now, E["patient_message"])
        open_id = self._open_case(st, now)
        if open_id:
            self.tr.add_evidence(open_id, now, {"day": row.day, "emergency_case": cid})

        f = followup_times("emergency", p["remoteness"], ack_h, res_h, cfg, self.rng, reminders=True)
        self.gt_cases.append({"case_id": cid, "gt_stalled": bool(f.get("forgot"))})
        for t, to in f["reminders"]:
            if now + t <= self.horizon:
                self.tr.remind(cid, now + t, to)
        who = "nurse_backup" if any(to == "backup" for _, to in f["reminders"]) else "nurse_primary"
        # Simulated clinician decision (simulation only; in a deployment the clinician records it).
        key = (p["patient_id"], row.day)
        if self._truth.get(key, False) or self._rhythm_truth.get(key) in ("new_af", "rate"):
            decision, reason = "call_911", "symptoms ongoing on the video call with an abnormal ECG"
        else:
            decision, reason = "cancel", ("symptoms settled on the video call; ECG matches the known baseline; "
                                          "review at the next clinic contact")
        resolved_h, made = None, None
        if now + f["ack"] <= self.horizon:
            self.tr.act(cid, "acknowledge", now + f["ack"], who, "clinician", note="video call started")
            if now + f["resolve"] <= self.horizon:
                self.tr.emergency_decision(cid, now + f["resolve"], who, "clinician", decision, reason)
                resolved_h, made = now + f["resolve"], decision
        st.case_id, st.case_resolved_h = cid, resolved_h
        if ecg.get("af") in ("new", "reported"):
            st.af = {"case_id": cid, "severity": "critical", "day": row.day}  # the care team now knows
        return self._record(row, "emergency", severity="emergency", case_id=cid,
                            flags=";".join(flags + ["emergency"]), emergency_decision=made,
                            notes="patient told: " + E["patient_message"], **ecg_log)

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
                st.selfcare = selfcare_summary(st.hist, today, self.cfg)
                self._check_loops(p, st, row, now, today)
                ecg = self._name_findings(p, st, self._read_ecg(p, row)) if self._ecg_recorded(p, row) else None
                if row.alert_level > 0:
                    self._handle_alert(p, st, row, today, now, ecg)
                else:
                    if ecg is not None:
                        self._ecg_without_alert(p, st, row, today, now, *ecg)
                    self._sleep_check(p, st, row, today, now)
                st.hist.append({k: today[k] for k in HIST_KEYS})
                st.hist = st.hist[-14:]
        return pd.DataFrame(self.decisions)

    def _sleep_check(self, p, st, row, today, now):
        """Sleep worse than the patient's own usual on a day with no alert: a low-priority check-in
        call for the nurse (or a note on the open case). At most one per patient in repeat_after_days."""
        s = (st.selfcare or {}).get("sleep")
        if not s or not s["worse"]:
            return
        every = self.cfg["selfcare"]["sleep"]["repeat_after_days"]
        if st.sleep_day is not None and row.day - st.sleep_day < every:
            return
        st.sleep_day = row.day
        open_id = self._open_case(st, now)
        if open_id:
            self.tr.add_evidence(open_id, now, {"day": row.day, "sleep": s["text"]})
            return self._record(row, "sleep_noted", case_id=open_id, flags="sleep_worse",
                                notes=s["text"] + "; added to the open case")
        review = build_review(p, row.day, [s["text"]], st.hist, today, "low", ["sleep_worse"], [],
                              kind="sleep_check", ack_h="{ack}", selfcare=st.selfcare)
        cid, _ = self._create_case(p, st, now, "sleep_check", "low", review,
                                   resolve_note="Nurse called the patient about their sleep; documented.",
                                   rng=self.rng_extra)
        self._record(row, "sleep_check", severity="low", case_id=cid, flags="sleep_worse", notes=s["text"])

    def _handle_alert(self, p, st, row, today, now, ecg_read=None):
        r = self.cfg["reliability"]
        flags, notes = assess_reading(st.hist, today, self.cfg, 0.0)
        ecg, ecg_log = ecg_read if ecg_read else (None, {})

        # 0. emergency: chest pain or fainting AND a reliable abnormal ECG at the same time. Checked
        #    before anything else, so a weight or cuff problem never delays it.
        em = [s for s in self.cfg["emergency"]["symptoms"] if today.get(s)]
        if em and ecg is not None:
            if not ecg["reliable"]:  # ask for a repeat recording straight away
                ecg2, log2 = self._name_findings(p, st, self._read_ecg(p, row))
                ecg_log = {**ecg_log, "ecg_repeated": True}
                if ecg2["reliable"]:
                    ecg, ecg_log = ecg2, {**log2, "ecg_repeated": True}
            if ecg["reliable"] and ecg["abnormal"] and (
                    ecg.get("af") != "known" or self.cfg["ecg_rhythm"].get("known_af_triggers_emergency", True)):
                return self._emergency(p, st, row, today, now, em, ecg, ecg_log, flags, notes)
        # without a reliable abnormal ECG, chest pain or fainting stays critical (platform rule)

        # 0b. new atrial fibrillation on a reliable ECG goes to the care team whatever the other
        #     readings look like (it does not depend on the scale or the cuff)
        symptoms = any(today.get(s) for s in SYMPTOMS)
        if self._af_to_report(st, ecg, symptoms):
            return self._new_af(p, st, row, today, now, ecg, ecg_log, flags, notes)

        # 1. one-off measurement faults -> ask for a repeat, no case
        if DATA_FAULTS & set(flags):
            return self._record(row, "repeat", flags=";".join(flags), notes=" | ".join(notes))

        # 2. scale-change memory: a jump that persists with no symptoms may be a new scale. A nurse
        #    verifies it before the baseline changes; until then weight-only alerts wait in that case.
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

        # 4. evidence from today's ECG recording, if the patient has a device (read in run())
        if ecg is not None and not ecg["reliable"]:
            flags.append("ecg_unreliable")
        severity = SEV[level]
        good_ecg = bool(ecg and ecg["reliable"])
        af = ecg.get("af") if good_ecg else None      # here "known" or "reported": new AF left at step 0b
        rate = ecg.get("rate") if good_ecg else None
        if af:
            flags.append("ecg_known_af")
            notes.append("ECG shows atrial fibrillation, " + ("known since enrollment" if af == "known" else
                                                              "already reported to the care team")
                         + ": not a new finding")
        if rate:
            flags.append(f"ecg_{rate}")
            fired = fired + [self._rate_text(ecg)]
        if severity == "caution" and good_ecg and symptoms:
            if rate:  # the ECG confirms a heart rate outside the patient's limits
                severity = "critical"
                fired = fired + [f"{rate} on a reliable ECG with symptoms"]
            elif ecg["abnormal"] and (not af or self.cfg["ecg_rhythm"].get("known_af_upgrades_with_symptoms")):
                # an abnormal ECG that known AF does not explain (or the care team chose to count it)
                severity = "critical"
                fired = fired + ["reliable abnormal ECG with symptoms"]

        # 5. escalate: update the open case or open a new one
        open_id = self._open_case(st, now)
        if open_id:
            self._add_to_open_case(p, st, open_id, row, now, level, fired, flags, severity)
            return self._record(row, "escalate_update", severity=severity, case_id=open_id, flags=";".join(flags),
                                **ecg_log)

        # 5b. the care team's plan is being followed: an alert that is not worse waits for the outcome check
        for loop in st.loops:
            if loop["check_h"] is not None and now < loop["check_h"] and not self._worse(loop, level, fired) \
                    and RANK[severity] <= RANK[loop["severity"]]:
                self.tr.add_evidence(loop["case_id"], now, {"day": row.day, "rules": fired, "flags": flags,
                                                            "during_patient_loop": True})
                return self._record(row, "loop_update", severity=severity, case_id=loop["case_id"],
                                    flags=";".join(flags), notes="not worse than when the case started; "
                                    "the outcome check after the action decides", **ecg_log)

        review = build_review(p, row.day, fired, st.hist, today, severity, flags, notes, ecg=ecg, ack_h="{ack}",
                              selfcare=st.selfcare)
        cid, resolved_h = self._create_case(p, st, now, "clinical", severity, review, trigger=(level, fired))
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


def checkin_from_sleep_log(tr, cfg, patient, hist, today, now):
    """A sleep entry the patient just logged in the app. hist: earlier days (platform readings with
    the logged days merged in, oldest first); today: this morning's entry.

    Sleep worse than the patient's own usual -> a low-priority check-in call, at most one per
    repeat_after_days. "Slept propped up" or "woke up breathless" are symptoms the platform alerts
    on at caution level, so the check-in is caution. With a case already open the entry is added to
    it. Returns (what, case_id, summary): what is "checkin", "noted" or None."""
    sc = selfcare_summary(hist, today, cfg)
    s = (sc or {}).get("sleep") or {}
    symptoms = [sym for key, sym in SLEEP_SYMPTOMS.items() if today.get(key)]
    why = ([s["text"]] if s.get("worse") else []) + [SYMPTOM_TEXT[x] + " (reported in the sleep log)"
                                                     for x in symptoms]
    if not why:
        return None, None, sc
    pid, day = patient["patient_id"], today["day"]
    open_case = tr.open_case(pid)
    if open_case:
        tr.add_evidence(open_case["case_id"], now, {"day": day, "sleep": "; ".join(why)})
        tr.commit()
        return "noted", open_case["case_id"], sc
    every_h = cfg["selfcare"]["sleep"]["repeat_after_days"] * 24
    recent = [c for c in tr.cases() if c["patient_id"] == pid and c["kind"] == "sleep_check"
              and now - c["created_h"] < every_h]
    if recent and not symptoms:
        return None, None, sc
    severity = "caution" if symptoms else "low"
    ack_h = cfg["followup"]["ack_deadline_h"][severity]
    review = build_review(patient, day, why, [h for h in hist if "weight_kg" in h], None, severity,
                          (["sleep_worse"] if s.get("worse") else []) + (["sleep_log_symptom"] if symptoms else []),
                          [], kind="sleep_check", ack_h=f"{ack_h:g}", selfcare=sc)
    review["symptoms"] = symptoms
    cid = tr.create(pid, "sleep_check", severity, now, ack_h, cfg["followup"]["resolve_deadline_h"][severity], review)
    tr.commit()
    return "checkin", cid, sc


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
