"""Structured, explainable review for the clinical team.

The review is built from facts only (platform readings, reliability checks, model
outputs). An LLM may reword it for readability, but it gets the same facts and
is told not to add any; the template text is always available as the fallback.
"""
import json
import os

NEXT_STEP = {
    "critical": "Contact the patient within {ack} hours and assess per the heart-failure program protocol.",
    "caution": "Contact the patient within {ack} hours; review the trend and symptoms.",
    "low": "Reach the patient within {ack} hours to restore daily readings (device, connectivity or support).",
}


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
        "why_flagged": rules_fired,
        "symptoms": [s for s in ("sob", "swelling") if today and today.get(s)],
        "trend": trend,
        "ecg": None if not ecg else {k: ecg.get(k) for k in ("reliable", "action", "abnormal", "prob", "quality",
                                                             "uncertainty", "reasons")},
        "mri": mri,
        "reliability_flags": flags,
        "caveats": caveats,
        "next_step": NEXT_STEP[severity].format(ack=ack_h),
        "decision_owner": "clinical team",
    }


def render(review):
    r = review
    lines = [f"**{r['severity'].upper()} - {r['kind']} review for {r['patient_id']} (day {r['day']})**",
             f"Context: {r['context']['age']} y, {r['context']['sex']}, {r['context']['remoteness']} community, "
             f"{r['context']['device'].replace('_', ' ')}."]
    if r["why_flagged"]:
        lines.append("Why flagged: " + "; ".join(r["why_flagged"]) + ".")
    if r["symptoms"]:
        lines.append("Reported symptoms: " + ", ".join({"sob": "shortness of breath", "swelling": "swelling"}[s]
                                                       for s in r["symptoms"]) + ".")
    if r["trend"]:
        w = [t["weight_kg"] for t in r["trend"]]
        lines.append(f"Weight trend (last {len(w)} readings): {w[0]:.1f} -> {w[-1]:.1f} kg; "
                     f"latest BP {r['trend'][-1]['sbp']:.0f}, HR {r['trend'][-1]['hr']:.0f}.")
    if r["ecg"]:
        e = r["ecg"]
        if e["reliable"]:
            lines.append(f"ECG (quality {e['quality']:.2f}): {'abnormal' if e['abnormal'] else 'no abnormality'} "
                         f"(model probability {e['prob']:.2f}).")
    if r["mri"]:
        m = r["mri"]
        where = f" ({m['scanner']} scanner)" if m.get("scanner") else ""
        unc = f", segmentation uncertainty {m['uncertainty']:.3f}" if m.get("uncertainty") is not None else ""
        lines.append(f"Cardiac MRI{where}: LVEF {m['lvef']}%{unc}" + ("" if m["reliable"] else " (flagged)") + ".")
    if r["caveats"]:
        lines.append("Reliability caveats: " + "; ".join(r["caveats"]) + ".")
    lines.append("Suggested next step: " + r["next_step"])
    lines.append("HeartSafe does not make the clinical decision; the clinical team does.")
    return "\n\n".join(lines)


def llm_rewrite(review, model):
    """Optional: reword the template for readability. Returns template text on any failure."""
    text = render(review)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return text, "template"
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=model, max_tokens=500,
            system=("You rewrite clinical monitoring summaries for nurses. Use ONLY the facts given. "
                    "Do not add diagnoses, medications, numbers or recommendations that are not in the facts. "
                    "Keep the suggested next step and the caveats. Under 150 words."),
            messages=[{"role": "user", "content": "Facts (JSON):\n" + json.dumps(review, default=str)
                       + "\n\nDraft:\n" + text}])
        return msg.content[0].text, model
    except Exception as e:  # network, key or package problems fall back to the template
        return text + f"\n\n_(LLM unavailable: {type(e).__name__}; template shown)_", "template"
