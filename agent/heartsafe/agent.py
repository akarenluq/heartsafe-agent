"""HeartSafe agent: sits between Medly alerts and the clinical team.

For every Medly alert it runs its tools (data-quality check, scale-change memory,
ECG reliability, MRI reliability), decides accept / repeat measurement / escalate
with a deterministic policy from config.yaml, writes an explainable review, and
opens or updates a tracked case. Missed readings create outreach tasks. The
clinical decision always stays with the clinical team.

`run_baseline` is the comparison: Medly alone, every alert goes to the nurse
queue with no reliability check and no follow-up tracking.
"""
import numpy as np
import pandas as pd

from .clinicians import followup_times
from .medly import rule_levels
from .reliability import assess_ecg, assess_mri, assess_reading
from .review import build_review
from .signals import ecg_output, mri_output

SEV = {1: "caution", 2: "critical"}
RANK = {"low": 0, "caution": 1, "critical": 2}
DATA_FAULTS = {"implausible_weight", "implausible_bp", "implausible_hr", "unit_error", "bp_artifact"}


class PatientState:
    def __init__(self):
        self.hist, self.offset, self.pending_step, self.missed = [], 0.0, None, 0
        self.case_id, self.case_resolved_h = None, None


class HeartSafeAgent:
    def __init__(self, cfg, patients, tracker, rng, horizon_h, ecg_pool=None, mri_pool=None, safeguards=None,
                 truth=None):
        self.cfg, self.tr, self.rng, self.horizon = cfg, tracker, rng, horizon_h
        self.patients = patients.set_index("patient_id", drop=False).to_dict("index")
        self.ecg_pool, self.mri_pool = ecg_pool, mri_pool
        self.safeguards = [s for s in (safeguards or []) if s.get("approved")]
        # Ground truth is used ONLY by the simulated world (what a repeat ECG or MRI would show),
        # never by the agent's decisions.
        self._truth = truth or {}
        self.decisions = []

    # ---- helpers -------------------------------------------------------
    def _ack_window(self, patient, severity):
        h = self.cfg["followup"]["ack_deadline_h"][severity]
        for s in self.safeguards:
            if s["type"] == "tighter_ack_deadline" and str(patient.get(s["field"])) == str(s["value"]):
                h *= s["factor"]
        return h

    def _open_case(self, st, now):
        return st.case_id if st.case_id and (st.case_resolved_h is None or st.case_resolved_h > now) else None

    def _create_case(self, patient, st, now, kind, severity, review):
        ack_h = self._ack_window(patient, severity)
        res_h = self.cfg["followup"]["resolve_deadline_h"][severity]
        review["next_step"] = review["next_step"].replace("{ack}", f"{ack_h:g}")
        cid = self.tr.create(patient["patient_id"], kind, severity, now, ack_h, res_h, review)
        f = followup_times(severity, patient["remoteness"], ack_h, res_h, self.cfg, self.rng, reminders=True,
                           force_forget=patient["patient_id"] == "DEMO-MARY" and kind == "clinical")
        for t, to in f["reminders"]:
            if now + t <= self.horizon:
                self.tr.remind(cid, now + t, to)
        who = "nurse_backup" if any(to == "backup" for _, to in f["reminders"]) else "nurse_primary"
        resolved_h = None
        if now + f["ack"] <= self.horizon:
            self.tr.act(cid, "acknowledge", now + f["ack"], who, "clinician")
            if now + f["ack"] + 0.5 <= self.horizon:
                self.tr.act(cid, "review", now + f["ack"] + 0.5, who, "clinician")
            if now + f["resolve"] <= self.horizon:
                self.tr.act(cid, "resolve", now + f["resolve"], who, "clinician", note="Patient contacted; plan documented.")
                resolved_h = now + f["resolve"]
        if kind == "clinical":
            st.case_id, st.case_resolved_h = cid, resolved_h
        return cid, resolved_h

    def _record(self, row, decision, **kw):
        self.decisions.append({"patient_id": row.patient_id, "day": row.day, "medly_level": row.medly_level,
                               "medly_rules": row.medly_rules, "decision": decision, **kw})

    @staticmethod
    def _today(row, offset=0.0):
        return {"day": row.day, "weight_kg": row.weight_kg - offset, "raw_weight_kg": row.weight_kg,
                "sbp": row.sbp, "hr": row.hr, "sob": bool(row.sob), "swelling": bool(row.swelling)}

    def _corrected_level(self, st, today):
        days = np.arange(today["day"] - 14, today["day"] + 1)
        arr = {k: np.full(len(days), np.nan) for k in ("w", "sbp", "hr")}
        sym = {k: np.zeros(len(days), bool) for k in ("sob", "swelling")}
        for h in st.hist + [today]:
            i = h["day"] - days[0]
            if 0 <= i < len(days):
                arr["w"][i], arr["sbp"][i], arr["hr"][i] = h["weight_kg"], h["sbp"], h["hr"]
                sym["sob"][i], sym["swelling"][i] = h["sob"], h["swelling"]
        lv, fired = rule_levels(arr["w"], arr["sbp"], arr["hr"], sym["sob"], sym["swelling"], self.cfg["medly_rules"])
        return int(lv[-1]), fired[-1]

    # ---- main loop -----------------------------------------------------
    def run(self, feed):
        r = self.cfg["reliability"]
        for pid, g in feed.groupby("patient_id", sort=False):
            p, st = self.patients[pid], PatientState()
            for row in g.sort_values("day").itertuples():
                now = row.day * 24 + 9.0
                if not row.submitted:
                    st.missed += 1
                    if st.missed == r["missing_outreach_days"]:
                        review = build_review(p, row.day, [f"no readings for {st.missed} days"], st.hist, None,
                                              "low", ["missed_readings"], [], kind="outreach", ack_h="{ack}")
                        cid, _ = self._create_case(p, st, now, "outreach", "low", review)
                        self._record(row, "outreach", severity="low", case_id=cid, flags="missed_readings")
                    continue
                st.missed = 0
                today = self._today(row, st.offset)
                if row.medly_level > 0:
                    self._handle_alert(p, st, row, today, now)
                st.hist.append({k: today[k] for k in ("day", "weight_kg", "sbp", "hr", "sob", "swelling")})
                st.hist = st.hist[-14:]
        return pd.DataFrame(self.decisions)

    def _handle_alert(self, p, st, row, today, now):
        r = self.cfg["reliability"]
        flags, notes = assess_reading(st.hist, today, self.cfg, 0.0)

        # 1. one-off measurement faults -> ask for a repeat, no case
        if DATA_FAULTS & set(flags):
            return self._record(row, "repeat", flags=";".join(flags), notes=" | ".join(notes))

        # 2. scale-change memory: a jump that persists with no symptoms is a new device, not fluid
        symptoms = today["sob"] or today["swelling"]
        if st.pending_step is not None:
            step_day, step, pre = st.pending_step
            if abs((today["weight_kg"] - pre) - step) <= r["step_persist_tolerance_kg"]:
                st.offset += step
                for h in st.hist:
                    if h["day"] >= step_day:
                        h["weight_kg"] -= step
                today = self._today(row, st.offset)
                st.pending_step = None
                notes.append(f"scale change of {step:+.1f} kg confirmed on repeat; readings re-baselined")
                flags = [f for f in flags if f != "step_change"] + ["device_change_confirmed"]
            else:
                st.pending_step = None
        elif "step_change" in flags:
            pre = float(np.median([h["weight_kg"] for h in st.hist[-3:]]))
            st.pending_step = (row.day, today["weight_kg"] - pre, pre)
            return self._record(row, "repeat", flags=";".join(flags), notes=" | ".join(notes))

        # 3. does the alert still stand on corrected data?
        level, fired = self._corrected_level(st, today)
        if level == 0:
            return self._record(row, "filtered", flags=";".join(flags) or "corrected_data_normal",
                                notes=" | ".join(notes) or "alert not supported after data-quality correction")

        # 4. evidence from the ECG device, if the patient has one
        ecg = None
        if p["ecg_kit"]:
            truth = self._truth.get((p["patient_id"], row.day), False)
            ecg = assess_ecg(ecg_output(p, truth, self.cfg, self.rng, self.ecg_pool), self.cfg)
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
            return self._record(row, "escalate_update", severity=severity, case_id=open_id, flags=";".join(flags))

        review = build_review(p, row.day, fired, st.hist, today, severity, flags, notes, ecg=ecg, ack_h="{ack}")
        cid, resolved_h = self._create_case(p, st, now, "clinical", severity, review)
        self._record(row, "escalate", severity=severity, case_id=cid, flags=";".join(flags))

        # 6. imaging follow-up for some critical cases: MRI results get their own reliability check
        if severity == "critical" and resolved_h and (
                p["patient_id"] == "DEMO-MARY" or self.rng.random() < self.cfg["simulation"]["mri_referral_prob"]):
            t_mri = resolved_h + 72
            if t_mri <= self.horizon:
                truth = self._truth.get((p["patient_id"], row.day), False)
                mri = assess_mri(mri_output(truth, self.rng, self.mri_pool))
                if not mri["reliable"] or p["patient_id"] == "DEMO-MARY":
                    rv = build_review(p, int(t_mri // 24), ["cardiac MRI result returned"], st.hist, None,
                                      "caution", mri["flags"], [], mri=mri, kind="imaging", ack_h="{ack}")
                    mcid, _ = self._create_case(p, st, t_mri, "imaging", "caution", rv)
                    self.decisions.append({"patient_id": p["patient_id"], "day": int(t_mri // 24), "medly_level": 0,
                                           "medly_rules": "", "decision": "imaging_review", "severity": "caution",
                                           "case_id": mcid, "flags": ";".join(mri["flags"])})


def run_baseline(feed, patients, cfg, rng, horizon_h):
    """Medly alone: each alert episode goes to the nurse queue; no reliability check, no tracking."""
    pats = patients.set_index("patient_id").to_dict("index")
    rows = []
    for pid, g in feed[feed.medly_level > 0].groupby("patient_id", sort=False):
        busy_until = -1.0
        for row in g.sort_values("day").itertuples():
            now = row.day * 24 + 9.0
            if now < busy_until:
                continue
            sev = SEV[row.medly_level]
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
