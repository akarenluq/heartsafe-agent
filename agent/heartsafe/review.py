"""Structured, explainable review for the clinical team.

The review is built from facts only (platform readings, reliability checks, model
outputs) and written from templates, so it needs no LLM: plain-language ECG and MRI
report summaries, and the follow-up status from the follow-up watch (watch.py). A
Claude-written version (orchestrator.py) is optional future work, off by default.
"""
from .monitoring import SYMPTOM_TEXT, SYMPTOMS

LVEF_BANDS = {"reduced_max": 40, "mildly_reduced_max": 49}  # % (universal definition of heart failure)

NEXT_STEP = {
    "emergency": ("Start a video call with the patient within {ack} minutes, then record your decision: "
                  "call 911 or cancel the alarm, with a reason. HeartSafe has already told the patient: "
                  "\"If you feel worse, call 911.\" HeartSafe never calls 911 itself."),
    "critical": ("Contact the patient within {ack} hours and assess per the heart-failure program protocol. "
                 "If oxygen saturation, heart rate or blood pressure have changed, consider transfer for "
                 "assessment and an echocardiogram."),
    "caution": "Contact the patient within {ack} hours; review the trend and symptoms.",
    "low": "Reach the patient within {ack} hours to restore daily readings (device, connectivity or support).",
}
FOLLOWUP_STEP = ("Reach the patient, their caregiver or the community health worker within {ack} hours: "
                 "the care team's instruction was not confirmed or not completed.")
SCALE_CHECK_STEP = ("Call the patient within {ack} hours to confirm whether they started using a different "
                    "scale. Readings are re-baselined only after you confirm it.")
ENROLLMENT = ("nyha", "days_since_discharge", "caregiver", "beta_blocker", "pacemaker_icd", "atrial_fibrillation",
              "baseline_sbp")


def build_review(patient, day, rules_fired, hist, today, severity, flags, notes, ecg=None, mri=None,
                 kind="clinical", ack_h=None):
    trend = [{"day": h["day"], "weight_kg": round(h["weight_kg"], 1), "sbp": h["sbp"], "hr": h["hr"]}
             for h in hist[-7:]] + ([{"day": today["day"], "weight_kg": round(today["weight_kg"], 1),
                                      "sbp": today["sbp"], "hr": today["hr"]}] if today else [])
    caveats = list(notes)
    if ecg and not ecg["reliable"]:
        what = "repeat ECG requested" if ecg.get("action", "repeat") == "repeat" else "ECG left for clinician review"
        caveats.append("ECG not used as evidence (" + ", ".join(ecg["reasons"]) + "); " + what)
    if mri and not mri["reliable"]:
        caveats.append("MRI measurements need expert review (" + ", ".join(mri.get("reasons") or mri["flags"]) + ")")
    if mri and mri.get("caveats"):
        caveats += mri["caveats"]
    return {
        "kind": kind, "patient_id": patient["patient_id"], "day": day, "severity": severity,
        "context": {k: patient[k] for k in ("remoteness", "sex", "age", "device")},
        "enrollment": {k: patient[k] for k in ENROLLMENT if k in patient},
        "why_flagged": rules_fired,
        "symptoms": [s for s in SYMPTOMS if today and today.get(s)],
        "trend": trend,
        "ecg": None if not ecg else {k: ecg.get(k) for k in ("reliable", "action", "abnormal", "prob", "quality",
                                                             "uncertainty", "reasons")},
        "mri": mri,
        "reliability_flags": flags,
        "caveats": caveats,
        "next_step": {"scale_check": SCALE_CHECK_STEP, "patient_followup": FOLLOWUP_STEP}.get(
            kind, NEXT_STEP[severity]).format(ack=ack_h),
        "decision_owner": "clinical team",
    }


def _band(x, good=0.8, fair=0.5):
    return "good" if x >= good else "fair" if x >= fair else "poor"


def summarize_ecg(e):
    """The single-lead ECG report in one or two plain sentences."""
    if not e:
        return None
    read = "abnormal" if e["abnormal"] else "no abnormality"
    s = (f"Single-lead ECG: the model reads {read} (probability abnormal {e['prob']:.2f}); signal quality "
         f"{_band(e['quality'])} ({e['quality']:.2f}), model uncertainty {e['uncertainty']:.2f}.")
    if e["reliable"]:
        return s + " Reliable enough to use as evidence."
    why = ", ".join(e.get("reasons") or []) or "the ECG pipeline did not accept it"
    then = "a repeat ECG was requested" if e.get("action", "repeat") == "repeat" else "left for clinician review"
    return s + f" Not used as evidence ({why}); {then}."


def summarize_mri(m, bands=None):
    """The cardiac MRI report in one or two plain sentences, with the ejection fraction band."""
    if not m:
        return None
    b = bands or LVEF_BANDS
    ef = m.get("lvef")
    if ef is None:
        part = "no ejection fraction measured"
    else:
        band = ("reduced" if ef <= b["reduced_max"] else "mildly reduced" if ef <= b["mildly_reduced_max"]
                else "preserved")
        part = f"left ventricular ejection fraction {ef:.0f}% ({band})"
    where = f" on a {m['scanner']} scanner" if m.get("scanner") else ""
    unc = f", segmentation uncertainty {m['uncertainty']:.3f}" if m.get("uncertainty") is not None else ""
    s = f"Cardiac MRI{where}: {part}{unc}."
    if m["reliable"]:
        s += " Accepted by the imaging checks."
    else:
        s += " Measurements need expert review (" + ", ".join(m.get("reasons") or m.get("flags") or []) + ")."
    return s + "".join(f" {c.rstrip('.')}." for c in m.get("caveats") or [])


def patient_evidence(decisions, reviews, bands=None):
    """All ECG checks and MRI exams of one patient in a few lines (decisions = that patient's rows)."""
    lines = []
    if "ecg_checked" in decisions:
        e = decisions[decisions.ecg_checked.fillna(False).astype(bool)]
        if len(e):
            ok = e.ecg_reliable.astype(bool)
            ab = e[ok].ecg_abnormal.astype(bool).sum()
            lines.append(f"ECG: {len(e)} recording(s) checked, {int(ok.sum())} reliable ({int(ab)} abnormal), "
                         f"{int((~ok).sum())} not used as evidence.")
    for r in reviews:
        if r.get("mri"):
            lines.append(f"Day {r['day']}: " + summarize_mri(r["mri"], bands))
    return lines


def render(review, followup=None, lvef_bands=None):
    """The template review. followup: lines from watch.followup_status."""
    r = review
    lines = [f"**{r['severity'].upper()} - {r['kind']} review for {r['patient_id']} (day {r['day']})**",
             f"Context: {r['context']['age']} y, {r['context']['sex']}, {r['context']['remoteness']} community, "
             f"{r['context']['device'].replace('_', ' ')}."]
    en = r.get("enrollment") or {}
    if en:
        bits = [f"NYHA {en['nyha']}" if en.get("nyha") else None,
                "beta-blocker" if en.get("beta_blocker") else None,
                "pacemaker/ICD" if en.get("pacemaker_icd") else None,
                "atrial fibrillation" if en.get("atrial_fibrillation") else None,
                "has a caregiver" if en.get("caregiver") else None]
        lines.append("Enrollment: " + ", ".join(b for b in bits if b) + ".")
    if r["why_flagged"]:
        lines.append("Why flagged: " + "; ".join(r["why_flagged"]) + ".")
    if r["symptoms"]:
        lines.append("Reported symptoms: " + ", ".join(SYMPTOM_TEXT[s] for s in r["symptoms"]) + ".")
    if r["trend"]:
        w = [t["weight_kg"] for t in r["trend"]]
        lines.append(f"Weight trend (last {len(w)} readings): {w[0]:.1f} -> {w[-1]:.1f} kg; "
                     f"latest BP {r['trend'][-1]['sbp']:.0f}, HR {r['trend'][-1]['hr']:.0f}.")
    for s in (summarize_ecg(r["ecg"]), summarize_mri(r["mri"], lvef_bands)):
        if s:
            lines.append(s)
    # the ECG / MRI summaries already state their own reliability; keep the other caveats
    caveats = [c for c in r["caveats"] if not c.startswith(("ECG not used", "MRI measurements need"))
               and c not in ((r["mri"] or {}).get("caveats") or [])]
    if caveats:
        lines.append("Reliability caveats: " + "; ".join(caveats) + ".")
    if followup:
        lines.append("Follow-up status: " + " ".join(followup))
    lines.append("Suggested next step: " + r["next_step"])
    lines.append("HeartSafe does not make the clinical decision; the clinical team does.")
    return "\n\n".join(lines)
