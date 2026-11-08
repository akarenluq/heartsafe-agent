"""Simulated Medly-style cohort: daily weight, blood pressure, heart rate and symptoms.

Medly data is not public, so this generates a large cohort that follows Medly's
published inputs. Ground truth (true decompensation episodes, faulty readings)
is kept in columns starting with "gt_" and is used only by the evaluation.
"""
import numpy as np
import pandas as pd

from . import age_band

REMOTENESS = ["urban", "rural", "remote"]


def _patient_days(pid, attrs, n_days, cfg, rng, script=None):
    sc = cfg["simulation"]
    rem, device = attrs["remoteness"], attrs["device"]
    w0, s0, h0 = attrs["w0"], attrs["s0"], attrs["h0"]
    script = script or {}

    ep_start = script.get("episode_start")
    if ep_start is None and rng.random() < sc["episode_prob"]:
        ep_start = int(rng.integers(6, n_days - 6))
    ep_len = script.get("episode_len", int(rng.integers(8, 15)))
    sc_day = script.get("scale_change_day")
    if sc_day is None and "scale_change_day" not in script and rng.random() < sc["scale_change_prob"][device]:
        sc_day = int(rng.integers(3, n_days - 3))
    sc_off = script.get("scale_offset", rng.choice([-1, 1]) * rng.uniform(2.0, 4.0))
    missing_days = set(script.get("missing_days", []))

    rows, drift = [], 0.0
    for d in range(n_days):
        in_ep = ep_start is not None and ep_start <= d < ep_start + ep_len
        ramp = min(d - ep_start + 1, 7) / 7 if in_ep else 0.0
        drift = 0.9 * drift + rng.normal(0, 0.08)
        w = w0 + drift + 3.2 * ramp + rng.normal(0, 0.3)
        sbp = s0 - 4 * ramp + rng.normal(0, 7)
        hr = h0 + 12 * ramp + rng.normal(0, 4)
        base = 0.0 if script else sc["symptom_base_rate"]
        sob = rng.random() < base + 0.65 * ramp
        swell = rng.random() < base + 0.55 * ramp
        fault = "none"
        clean_w, clean_sbp, clean_hr = w, sbp, hr
        if sc_day is not None and d >= sc_day:
            w += sc_off
            fault = "scale_change"
        if script.get("unit_error_day") == d or (not script and rng.random() < sc["unit_error_rate"]):
            w *= 2.2046
            fault = "unit_error"
        if script.get("cuff_error_day") == d or (not script and rng.random() < sc["cuff_error_rate"]):
            sbp += rng.normal(38, 6)
            hr += rng.normal(0, 2)
            fault = "cuff_error"
        submitted = d not in missing_days and (bool(script) or rng.random() >= sc["missing_rate"][rem])
        rows.append({
            "patient_id": pid, "day": d, "submitted": submitted,
            "weight_kg": round(w, 1) if submitted else np.nan,
            "sbp": round(sbp) if submitted else np.nan,
            "hr": round(hr) if submitted else np.nan,
            "sob": bool(sob) if submitted else False,
            "swelling": bool(swell) if submitted else False,
            "gt_episode": in_ep, "gt_fault": fault if submitted else "none",
            "gt_weight_kg": round(clean_w, 1) if submitted else np.nan,
            "gt_sbp": round(clean_sbp) if submitted else np.nan, "gt_hr": round(clean_hr) if submitted else np.nan,
        })
    return rows


def simulate_cohort(cfg, n_patients=None, n_days=None, seed=None, demo=True):
    co, sc = cfg["cohort"], cfg["simulation"]
    n_patients = n_patients or co["patients"]
    n_days = n_days or co["days"]
    rng = np.random.default_rng(co["seed"] if seed is None else seed)
    pats, rows = [], []
    for i in range(n_patients):
        rem = rng.choice(REMOTENESS, p=sc["remoteness_mix"])
        sex = rng.choice(["F", "M"], p=[0.45, 0.55])
        age = int(np.clip(rng.normal(70, 11), 35, 95))
        attrs = {
            "patient_id": f"P{i:05d}", "remoteness": rem, "sex": sex, "age": age, "age_band": age_band(age),
            "device": "own_scale" if rng.random() < sc["own_scale_share"][rem] else "provided_kit",
            "ecg_kit": bool(rng.random() < sc["ecg_kit_share"]),
            "w0": rng.normal(84 if sex == "M" else 72, 12), "s0": rng.normal(122, 14), "h0": rng.normal(72, 8),
        }
        pats.append(attrs)
        rows += _patient_days(attrs["patient_id"], attrs, n_days, cfg, rng)

    if demo:
        # Scripted demo patient for the presentation: remote community, new scale on
        # day 20 (should be filtered), real decompensation from day 50 (should escalate).
        attrs = {"patient_id": "DEMO-MARY", "remoteness": "remote", "sex": "F", "age": 68, "age_band": age_band(68),
                 "device": "own_scale", "ecg_kit": True, "w0": 71.0, "s0": 128.0, "h0": 74.0}
        pats.append(attrs)
        rows += _patient_days("DEMO-MARY", attrs, n_days, cfg, np.random.default_rng(42), script={
            "scale_change_day": 20, "scale_offset": 2.6, "episode_start": 50, "episode_len": 12,
            "missing_days": [44, 45, 46], "cuff_error_day": 30})

    patients = pd.DataFrame(pats).drop(columns=["w0", "s0", "h0"])
    return patients, pd.DataFrame(rows)
