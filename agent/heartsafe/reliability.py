"""Reliability checks: is a Medly alert backed by trustworthy data and model outputs?"""
import numpy as np


def _median(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.median(xs)) if xs else float("nan")


def assess_reading(hist, today, cfg, offset=0.0):
    """Data-quality flags for today's reading given recent history.

    hist: list of earlier dicts {day, weight_kg, sbp, hr, sob, swelling} with readings (last 14 days),
          weights already corrected for any confirmed scale change.
    offset: weight correction from a confirmed scale change, applied to today's raw weight.
    Returns (flags, notes): flags drive the decision; notes explain it.
    """
    r = cfg["reliability"]
    flags, notes = [], []
    w = today["weight_kg"] - offset
    sbp, hr = today["sbp"], today["hr"]
    symptoms = bool(today["sob"] or today["swelling"])

    if not (r["weight_plausible_kg"][0] <= today["weight_kg"] <= r["weight_plausible_kg"][1]):
        flags.append("implausible_weight")
    if not (r["sbp_plausible"][0] <= sbp <= r["sbp_plausible"][1]):
        flags.append("implausible_bp")
    if not (r["hr_plausible"][0] <= hr <= r["hr_plausible"][1]):
        flags.append("implausible_hr")

    recent = [h for h in hist if h["day"] >= today["day"] - 7]
    last3 = [h["weight_kg"] for h in hist[-3:]]
    med_w = _median(last3)
    med_sbp = _median([h["sbp"] for h in recent])
    med_hr = _median([h["hr"] for h in recent])

    if not np.isnan(med_w):
        ratio = w / med_w
        if r["unit_error_ratio"][0] <= ratio <= r["unit_error_ratio"][1]:
            flags.append("unit_error")
            notes.append(f"weight {today['weight_kg']:.1f} is about 2.2x the recent {med_w:.1f} kg (entered in lb?)")
        else:
            step = w - med_w
            hr_stable = np.isnan(med_hr) or abs(hr - med_hr) <= r["hr_stable_bpm"]
            if abs(step) >= r["step_change_kg"] and not symptoms and hr_stable:
                flags.append("step_change")
                notes.append(f"weight jumped {step:+.1f} kg overnight with no symptoms and stable heart rate (new scale?)")
    if not np.isnan(med_sbp) and sbp - med_sbp >= r["cuff_jump_mmhg"] and not symptoms \
            and (np.isnan(med_hr) or abs(hr - med_hr) <= r["hr_stable_bpm"]):
        flags.append("bp_artifact")
        notes.append(f"systolic BP {sbp:.0f} vs usual {med_sbp:.0f} with no other change (cuff placement?)")
    if len(recent) < r["stale_min_readings_7d"]:
        flags.append("sparse_data")
        notes.append(f"only {len(recent)} readings in the past 7 days")
    return flags, notes


FLAG_TEXT = {
    # ECG pipeline
    "low_quality": "poor signal quality", "noisy_signal": "noisy signal", "baseline_wander": "baseline wander",
    "implausible_heart_rate": "implausible heart rate", "device_shift": "recording unlike the training ECGs",
    # imaging pipeline
    "scanner_shift": "scanner differs from training scans", "low_snr": "low image quality",
    "severe_scanner_shift": "image far outside the validated range", "lv_slice_gap": "gap in LV coverage",
    "lv_coverage": "incomplete LV coverage", "lvef_out_of_range": "EF outside the plausible range",
    "esv_not_below_edv": "ESV not below EDV", "lv_edv_out_of_range": "LV volume outside the plausible range",
    "rv_edv_out_of_range": "RV volume outside the plausible range", "lv_fragmented": "fragmented LV segmentation",
    # both
    "high_uncertainty": "model uncertain",
}


def assess_ecg(o, cfg):
    """Is an ECG model output reliable enough to use as evidence?

    A real pipeline output carries its own decision (suggested_action), made with
    thresholds calibrated on held-out ECGs; the agent uses it as is. Simulated outputs
    are checked against the thresholds in config.yaml."""
    r = cfg["reliability"]
    q = o.get("quality", {}).get("sqi", o.get("quality", {}).get("score", 1.0))
    unc = o.get("uncertainty", 0.0)
    prob = o.get("probability", o.get("prediction", {}).get("abnormal_prob", 0.0))
    if "suggested_action" in o:
        action = o["suggested_action"]
        reasons = [FLAG_TEXT.get(f, f.replace("_", " ")) for f in o.get("flags", []) if f != "safeguard"]
        if action != "accept" and not reasons:
            reasons = ["ECG pipeline did not accept the result"]
    else:
        reasons = []
        if q < r["ecg_min_quality"]:
            reasons.append(f"signal quality {q:.2f}")
        if unc > r["ecg_max_uncertainty"]:
            reasons.append(f"model uncertainty {unc:.2f}")
        if o.get("shift_score") is not None and "device_shift" in o.get("flags", []):
            reasons.append("device shift")
        action = "accept" if not reasons else ("repeat" if q < r["ecg_min_quality"] else "escalate")
    return {"reliable": action == "accept", "action": action, "abnormal": prob >= r["ecg_abnormal_prob"],
            "prob": prob, "quality": q, "uncertainty": unc, "reasons": list(dict.fromkeys(reasons))}


def assess_mri(o):
    flags = o.get("flags", [])
    meta = o.get("metadata") or {}
    return {"reliable": o.get("suggested_action", "accept") == "accept", "action": o.get("suggested_action", "accept"),
            "flags": flags, "reasons": [FLAG_TEXT.get(f, f.replace("_", " ")) for f in flags],
            "caveats": o.get("caveats", []), "lvef": o.get("prediction", {}).get("lvef_pct"),
            "uncertainty": o.get("uncertainty"), "scanner": meta.get("vendor_name"), "exam_id": o.get("exam_id")}
