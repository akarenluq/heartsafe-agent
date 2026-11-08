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


def assess_ecg(o, cfg):
    """Is an ECG model output reliable enough to use as evidence?"""
    r = cfg["reliability"]
    q = o.get("quality", {}).get("sqi", o.get("quality", {}).get("score", 1.0))
    unc = o.get("uncertainty", 0.0)
    prob = o.get("probability", o.get("prediction", {}).get("abnormal_prob", 0.0))
    reasons = []
    if q < r["ecg_min_quality"]:
        reasons.append(f"signal quality {q:.2f}")
    if unc > r["ecg_max_uncertainty"]:
        reasons.append(f"model uncertainty {unc:.2f}")
    if o.get("shift_score") is not None and "device_shift" in o.get("flags", []):
        reasons.append("device shift")
    return {"reliable": not reasons, "abnormal": prob >= r["ecg_abnormal_prob"], "prob": prob,
            "quality": q, "uncertainty": unc, "reasons": reasons}


def assess_mri(o):
    flags = o.get("flags", [])
    return {"reliable": o.get("suggested_action", "accept") == "accept", "flags": flags,
            "lvef": o.get("prediction", {}).get("lvef_pct")}
