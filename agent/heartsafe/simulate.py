"""Simulated remote-monitoring cohort (modelled on Medly).

Platform data is not public, so this generates a large cohort that follows the
published inputs of Medly, the main reference platform: daily weight, blood
pressure, heart rate and symptoms, plus the extra inputs cardiologists asked for
(diastolic BP, oxygen saturation, more symptoms, diuretic taken, post-discharge
labs). Candidates are screened with the inclusion criteria in config.yaml and
only eligible patients are enrolled. Ground truth (true decompensation episodes,
faulty readings) is kept in columns starting with "gt_" and is used only by the
evaluation.
"""
import numpy as np
import pandas as pd

from . import age_band

REMOTENESS = ["urban", "rural", "remote"]
EXCLUSIONS = {"dementia": "dementia or uncontrolled psychiatric illness",
              "long_term_care": "lives in long-term care",
              "palliative": "life expectancy under 1 year / palliative care",
              "transplant_or_lvad": "transplant list or ventricular assist device",
              "dialysis": "on dialysis"}


def screen(p, inc):
    """Reasons a candidate is not eligible (empty list = enrolled)."""
    why = []
    if p["age"] < inc["min_age"]:
        why.append(f"under {inc['min_age']}")
    if p["hf_admission_h"] < inc["min_hf_admission_h"]:
        why.append(f"heart-failure admission under {inc['min_hf_admission_h']} h")
    if p["days_since_discharge"] > inc["max_days_since_discharge"]:
        why.append(f"discharged more than {inc['max_days_since_discharge']} days ago")
    if p["remoteness"] not in inc["remoteness"]:
        why.append("outside the target communities")
    if p["nyha"] not in inc["nyha"]:
        why.append(f"NYHA class {p['nyha']}")
    if not p["can_self_measure"] and not p["caregiver"]:
        why.append("cannot take readings and has no caregiver")
    why += [EXCLUSIONS[k] for k in inc["exclude"] if p.get(k)]
    return why


def _candidate(i, rng, sc):
    pr = sc["patient"]
    rem = rng.choice(REMOTENESS, p=sc["remoteness_mix"])
    sex = rng.choice(["F", "M"], p=[0.45, 0.55])
    age = int(np.clip(rng.normal(70, 11), 35, 95))
    p = {
        "patient_id": f"P{i:05d}", "remoteness": rem, "sex": sex, "age": age, "age_band": age_band(age),
        "device": "own_scale" if rng.random() < sc["own_scale_share"][rem] else "provided_kit",
        "ecg_kit": bool(rng.random() < sc["ecg_kit_share"]),
        # enrollment record (NYHA is assessed at the bedside; no echocardiogram needed)
        "nyha": rng.choice(["I", "II", "III", "IV"], p=pr["nyha_mix"]),
        "hf_admission_h": round(float(rng.lognormal(np.log(pr["median_admission_h"]), 0.6))),
        "days_since_discharge": int(rng.integers(0, pr["max_days_since_discharge_seen"] + 1)),
        "caregiver": bool(rng.random() < pr["caregiver_share"]),
        "can_self_measure": bool(rng.random() < pr["can_self_measure_share"]),
        "beta_blocker": bool(rng.random() < pr["beta_blocker_share"]),
        "pacemaker_icd": bool(rng.random() < pr["pacemaker_share"]),
        "atrial_fibrillation": bool(rng.random() < pr["af_share"]),
        "creatinine_baseline_umol": round(float(np.clip(rng.normal(105, 25), 50, 250))),
        "w0": rng.normal(84 if sex == "M" else 72, 12), "s0": rng.normal(122, 14), "h0": rng.normal(72, 8),
        "d0": rng.normal(74, 8), "o0": float(np.clip(rng.normal(96.5, 1.0), 94.5, 99)), "k0": rng.normal(4.3, 0.3),
    }
    p["baseline_sbp"] = round(p["s0"])
    for k, share in pr["exclusion_rates"].items():
        p[k] = bool(rng.random() < share)
    return p


def _episode_start(rng, sc, n_days):
    """Decompensation is most likely in the first weeks after discharge."""
    for _ in range(100):
        d = 6 + int(rng.exponential(sc["episode_early_scale_days"]))
        if d < n_days - 6:
            return d
    return int(rng.integers(6, n_days - 6))


def _patient_days(pid, attrs, n_days, cfg, rng, script=None):
    sc = cfg["simulation"]
    rem, device = attrs["remoteness"], attrs["device"]
    w0, s0, h0 = attrs["w0"], attrs["s0"], attrs["h0"]
    d0, o0, k0 = attrs.get("d0", 74.0), attrs.get("o0", 96.5), attrs.get("k0", 4.3)
    creat0 = attrs.get("creatinine_baseline_umol", 105)
    dsd = attrs.get("days_since_discharge", 0)
    script = script or {}

    ep_start = script.get("episode_start")
    if ep_start is None and "episode_start" not in script and rng.random() < sc["episode_prob"]:
        ep_start = _episode_start(rng, sc, n_days)
    ep_len = script.get("episode_len", int(rng.integers(8, 15)))
    sc_day = script.get("scale_change_day")
    if sc_day is None and "scale_change_day" not in script and rng.random() < sc["scale_change_prob"][device]:
        sc_day = int(rng.integers(3, n_days - 3))
    sc_off = script.get("scale_offset", rng.choice([-1, 1]) * rng.uniform(2.0, 4.0))
    missing_days = set(script.get("missing_days", []))
    lab_days = {d - dsd for d in sc["lab_days_after_discharge"]}

    rows, drift = [], 0.0
    for d in range(n_days):
        in_ep = ep_start is not None and ep_start <= d < ep_start + ep_len
        ramp = min(d - ep_start + 1, 7) / 7 if in_ep else 0.0
        pre_ep = ep_start is not None and ep_start - 5 <= d < ep_start
        drift = 0.9 * drift + rng.normal(0, 0.08)
        w = w0 + drift + 3.2 * ramp + rng.normal(0, 0.3)
        sbp = s0 - 4 * ramp + rng.normal(0, 7)
        dbp = d0 - 2 * ramp + rng.normal(0, 5)
        hr = h0 + 12 * ramp + rng.normal(0, 4)
        spo2 = float(np.clip(o0 - 3.5 * ramp + rng.normal(0, 0.7), 80, 100))
        base = 0.0 if script else sc["symptom_base_rate"]
        sob = rng.random() < base + 0.65 * ramp
        swell = rng.random() < base + 0.55 * ramp
        orthopnea = rng.random() < (0 if script else sc["orthopnea_base_rate"]) + 0.45 * ramp
        pnd = rng.random() < (0 if script else sc["pnd_base_rate"]) + 0.3 * ramp
        chest = rng.random() < (0 if script else sc["chest_pain_base_rate"]) + 0.03 * ramp
        syncope = rng.random() < (0 if script else sc["syncope_base_rate"]) + 0.015 * ramp
        miss = 0.0 if script else sc["diuretic_miss_rate"] + (sc["diuretic_miss_before_episode"] if pre_ep else 0)
        diuretic = rng.random() >= miss
        fault = "none"
        clean_w, clean_sbp, clean_dbp, clean_hr = w, sbp, dbp, hr
        if sc_day is not None and d >= sc_day:
            w += sc_off
            fault = "scale_change"
        if script.get("unit_error_day") == d or (not script and rng.random() < sc["unit_error_rate"]):
            w *= 2.2046
            fault = "unit_error"
        if script.get("cuff_error_day") == d or (not script and rng.random() < sc["cuff_error_rate"]):
            sbp += rng.normal(38, 6)
            dbp += rng.normal(15, 4)
            hr += rng.normal(0, 2)
            fault = "cuff_error"
        submitted = d not in missing_days and (bool(script) or rng.random() >= sc["missing_rate"][rem])
        lab = submitted and d in lab_days
        na = lambda v, nd=0: (round(v, nd) if nd else round(v)) if submitted else np.nan  # noqa: E731
        rows.append({
            "patient_id": pid, "day": d, "submitted": submitted,
            "weight_kg": na(w, 1), "sbp": na(sbp), "dbp": na(dbp), "hr": na(hr), "spo2": na(spo2),
            "sob": bool(sob) and submitted, "swelling": bool(swell) and submitted,
            "orthopnea": bool(orthopnea) and submitted, "pnd": bool(pnd) and submitted,
            "chest_pain": bool(chest) and submitted, "syncope": bool(syncope) and submitted,
            "diuretic_taken": bool(diuretic) or not submitted,
            "potassium": round(float(rng.normal(k0, 0.25)), 1) if lab else np.nan,
            "creatinine_umol": round(creat0 + 30 * ramp + rng.normal(0, 6)) if lab else np.nan,
            "gt_episode": in_ep, "gt_fault": fault if submitted else "none",
            "gt_weight_kg": round(clean_w, 1) if submitted else np.nan,
            "gt_sbp": round(clean_sbp) if submitted else np.nan, "gt_dbp": round(clean_dbp) if submitted else np.nan,
            "gt_hr": round(clean_hr) if submitted else np.nan,
        })
    return rows


def simulate_cohort(cfg, n_patients=None, n_days=None, seed=None, demo=True):
    """Screen candidates until n_patients are enrolled. The screening log is in patients.attrs["screening"]."""
    co, sc, inc = cfg["cohort"], cfg["simulation"], cfg["inclusion"]
    n_patients = n_patients or co["patients"]
    n_days = n_days or co["days"]
    rng = np.random.default_rng(co["seed"] if seed is None else seed)
    pats, rows, reasons, n_cand = [], [], {}, 0
    while len(pats) < n_patients:
        cand = _candidate(n_cand, rng, sc)
        n_cand += 1
        why = screen(cand, inc)
        if why:
            for r in why:
                reasons[r] = reasons.get(r, 0) + 1
            continue
        pats.append(cand)
        rows += _patient_days(cand["patient_id"], cand, n_days, cfg, rng)

    if demo:
        # Scripted demo patient for the presentation: remote community, discharged 3 days ago,
        # new scale on day 20 (nurse verifies it, then it is filtered), real decompensation
        # from day 50 (should escalate).
        attrs = {"patient_id": "DEMO-MARY", "remoteness": "remote", "sex": "F", "age": 68, "age_band": age_band(68),
                 "device": "own_scale", "ecg_kit": True, "nyha": "III", "hf_admission_h": 120,
                 "days_since_discharge": 3, "caregiver": True, "can_self_measure": True, "beta_blocker": True,
                 "pacemaker_icd": False, "atrial_fibrillation": False, "creatinine_baseline_umol": 98,
                 "baseline_sbp": 128, **{k: False for k in EXCLUSIONS},
                 "w0": 71.0, "s0": 128.0, "h0": 74.0, "d0": 76.0, "o0": 97.0, "k0": 4.2}
        pats.append(attrs)
        rows += _patient_days("DEMO-MARY", attrs, n_days, cfg, np.random.default_rng(42), script={
            "scale_change_day": 20, "scale_offset": 2.6, "episode_start": 50, "episode_len": 12,
            "missing_days": [44, 45, 46], "cuff_error_day": 30})

    patients = pd.DataFrame(pats).drop(columns=["w0", "s0", "h0", "d0", "o0", "k0"])
    patients.attrs["screening"] = {"candidates": n_cand, "enrolled": n_patients,
                                   "not_eligible": n_cand - n_patients,
                                   "reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])),
                                   "criteria": inc}
    return patients, pd.DataFrame(rows)
