"""Simulated remote-monitoring cohort (modelled on Medly).

Platform data is not public, so this generates a large cohort that follows the
published inputs of Medly, the main reference platform: daily weight, blood
pressure, heart rate and symptoms, plus the extra inputs cardiologists asked for
(diastolic BP, oxygen saturation, more symptoms, diuretic taken, post-discharge
labs). Candidates are screened with the inclusion criteria in config.yaml and
only eligible patients are enrolled. Ground truth (true decompensation episodes,
faulty readings, heart rhythm) is kept in columns starting with "gt_" and is used
only by the evaluation.

Heart rhythm (known and new atrial fibrillation) and the patient's self-care log
(last night's sleep, breathing-exercise sessions) are added after the readings, each
from its own random stream per patient, so they never change the readings above.
"""
import zlib

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


def _personal_limits(p, cfg):
    """Enrollment: personal limits for a patient whose usual BP or heart rate sits near a default limit."""
    rules, m = cfg["alert_rules"], cfg["simulation"]["patient"]["personal_limit_margin"]
    if p["s0"] - m["sbp"] < rules["sbp_low"]:
        p["sbp_low"] = round(p["s0"] - m["sbp"])
    if p["s0"] + m["sbp"] > rules["sbp_high"]:
        p["sbp_high"] = round(p["s0"] + m["sbp"])
    if p["h0"] - m["hr"] < rules["hr_low"]:
        p["hr_low"] = round(p["h0"] - m["hr"])
    if p["h0"] + m["hr"] > rules["hr_high"]:
        p["hr_high"] = round(p["h0"] + m["hr"])


def _own_rng(seed, pid, stream):
    """A random stream of its own for one patient and one topic."""
    return np.random.default_rng([int(seed), zlib.crc32(pid.encode()), stream])


def _add_rhythm(feed, pats, n_days, cfg, seed):
    """True heart rhythm on every day: gt_rhythm ("af" / "sinus") and gt_new_af.

    Patients with atrial fibrillation at enrollment are in AF on every recording (permanent) or
    on some days (paroxysmal). A few patients without it develop new AF for a spell: the heart
    beats faster and they are more often short of breath."""
    rc = cfg["simulation"].get("rhythm")
    n = len(pats)
    af, new = np.zeros((n, n_days), bool), np.zeros((n, n_days), bool)
    if rc:
        grid = lambda c, t: feed[c].to_numpy(t).reshape(n, n_days).copy()  # noqa: E731
        hr, gt_hr, sob, sub = grid("hr", float), grid("gt_hr", float), grid("sob", bool), grid("submitted", bool)
        lo, hi = rc["new_af_days"]
        for i, p in enumerate(pats):
            if p["patient_id"] == "DEMO-MARY":
                continue
            rng = _own_rng(seed, p["patient_id"], 1)
            u = rng.random()
            if p.get("atrial_fibrillation"):
                af[i] = True if u < rc["permanent_af_share"] else rng.random(n_days) < rc["paroxysmal_af_day_prob"]
            elif u < rc["new_af_prob"] and n_days > lo + 6:
                start = int(rng.integers(5, n_days - lo))
                d = slice(start, min(start + int(rng.integers(lo, hi + 1)), n_days))
                af[i, d] = new[i, d] = True
                rise = np.round(rng.normal(rc["new_af_hr_rise"], 6, n_days))
                hr[i, d] += rise[d]
                gt_hr[i, d] += rise[d]
                sob[i, d] |= (rng.random(n_days) < rc["new_af_breathless"])[d] & sub[i, d]
        feed = feed.assign(hr=hr.ravel(), gt_hr=gt_hr.ravel(), sob=sob.ravel())
    return feed.assign(gt_rhythm=np.where(af.ravel(), "af", "sinus"), gt_new_af=new.ravel())


def _add_rate_spells(feed, pats, n_days, cfg, seed):
    """A few days with the heart rate outside the patient's own limits: gt_rate ("slow" / "fast" /
    "none"). Too slow in a patient on a beta-blocker without a pacemaker, too fast in any patient;
    on some of these days the patient is short of breath."""
    rc = cfg["simulation"].get("heart_rate_spells")
    n = len(pats)
    kind = np.full((n, n_days), "none", dtype=object)
    if rc and n_days > rc["days"][1] + 6:
        rules = cfg["alert_rules"]
        grid = lambda c, t: feed[c].to_numpy(t).reshape(n, n_days).copy()  # noqa: E731
        hr, gt_hr, sob, sub = grid("hr", float), grid("gt_hr", float), grid("sob", bool), grid("submitted", bool)
        lo, hi = rc["days"]
        for i, p in enumerate(pats):
            if p["patient_id"] == "DEMO-MARY":
                continue
            rng = _own_rng(seed, p["patient_id"], 3)
            u = rng.random()
            slow = u < rc["slow_prob"] and p.get("beta_blocker") and not p.get("pacemaker_icd")
            if not slow and u < 1 - rc["fast_prob"]:
                continue
            start = int(rng.integers(5, n_days - hi))
            d = slice(start, start + int(rng.integers(lo, hi + 1)))
            m = d.stop - d.start
            v = p.get("hr_low", rules["hr_low"]) - rng.integers(2, 10, m) if slow else \
                p.get("hr_high", rules["hr_high"]) + rng.integers(3, 26, m)
            hr[i, d] = gt_hr[i, d] = np.where(sub[i, d], v, np.nan)
            sob[i, d] |= (rng.random(m) < rc["breathless"]) & sub[i, d]
            kind[i, d] = "slow" if slow else "fast"
        feed = feed.assign(hr=hr.ravel(), gt_hr=gt_hr.ravel(), sob=sob.ravel())
    return feed.assign(gt_rate=kind.ravel())


def _add_selfcare(feed, pats, n_days, cfg, seed):
    """What the patient logs in the daily check-in: hours slept last night, times woken, and
    breathing-exercise sessions done. Sleep is shorter during a decompensation, on nights with
    breathlessness, and in a spell of poor sleep (most likely soon after discharge)."""
    sc = cfg["simulation"].get("selfcare")
    if not sc:
        return feed
    n = len(pats)
    plan = cfg.get("selfcare", {}).get("breathing", {}).get("sessions_per_day", 3)
    grid = lambda c: feed[c].to_numpy(bool).reshape(n, n_days)  # noqa: E731
    ep, sub, breathless = grid("gt_episode"), grid("submitted"), grid("orthopnea") | grid("pnd")
    hours, wakes, sess = (np.full((n, n_days), np.nan) for _ in range(3))
    poor = np.zeros((n, n_days), bool)
    lo, hi = sc["poor_sleep_days"]
    for i, p in enumerate(pats):
        demo = p["patient_id"] == "DEMO-MARY"
        rng = _own_rng(seed, p["patient_id"], 2)
        usual = float(np.clip(rng.normal(*sc["usual_sleep_h"]), 5.0, 9.0))
        wake0 = float(np.clip(rng.gamma(2.0, 0.6), 0.2, 3.0))
        adherence = float(rng.beta(*sc["breathing_adherence_beta"]))
        has_spell = rng.random() < sc["poor_sleep_prob"]
        start = 3 + int(rng.exponential(sc["poor_sleep_early_scale_days"]))
        length = int(rng.integers(lo, hi + 1))
        if demo:  # the demo patient sleeps badly in her second week at home
            usual, wake0, adherence, has_spell, start, length = 6.8, 1.0, 0.7, True, 8, 6
        if has_spell and start < n_days:
            poor[i, start:start + length] = True
        run = np.zeros(n_days)  # days into the decompensation: sleep worsens over the first week
        for d in range(n_days):
            run[d] = (run[d - 1] + 1 if d else 1) if ep[i, d] else 0
        ramp = np.minimum(run, 7) / 7
        h = usual + rng.normal(0, sc["night_sd_h"], n_days) - sc["episode_sleep_loss_h"] * ramp             - sc["breathless_night_loss_h"] * breathless[i] - sc["poor_sleep_loss_h"] * poor[i]
        w = rng.poisson(wake0 + 2.0 * ramp + 1.5 * breathless[i] + 2.0 * poor[i])
        b = rng.binomial(plan, np.clip(adherence * (1 - 0.4 * ramp), 0, 1))
        ok = sub[i]
        hours[i, ok] = np.round(np.clip(h, 1.5, 11.0) * 2)[ok] / 2  # patients answer in half hours
        wakes[i, ok], sess[i, ok] = w[ok], b[ok]
    return feed.assign(sleep_hours=hours.ravel(), sleep_awakenings=wakes.ravel(),
                       breathing_sessions=sess.ravel(), gt_poor_sleep=poor.ravel())


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
        _personal_limits(cand, cfg)
        pats.append(cand)
        rows += _patient_days(cand["patient_id"], cand, n_days, cfg, rng)

    if demo:
        # Scripted demo patient for the presentation: remote community, discharged 3 days ago,
        # new scale on day 20 (nurse verifies it, then it is filtered), real decompensation
        # on days 50-54 (should escalate; the care team's plan works and her readings recover).
        attrs = {"patient_id": "DEMO-MARY", "remoteness": "remote", "sex": "F", "age": 68, "age_band": age_band(68),
                 "device": "own_scale", "ecg_kit": True, "nyha": "III", "hf_admission_h": 120,
                 "days_since_discharge": 3, "caregiver": True, "can_self_measure": True, "beta_blocker": True,
                 "pacemaker_icd": False, "atrial_fibrillation": False, "creatinine_baseline_umol": 98,
                 "baseline_sbp": 128, **{k: False for k in EXCLUSIONS},
                 "w0": 71.0, "s0": 128.0, "h0": 74.0, "d0": 76.0, "o0": 97.0, "k0": 4.2}
        pats.append(attrs)
        rows += _patient_days("DEMO-MARY", attrs, n_days, cfg, np.random.default_rng(42), script={
            "scale_change_day": 20, "scale_offset": 2.6, "episode_start": 50, "episode_len": 5,
            "missing_days": [44, 45, 46], "cuff_error_day": 30})

    patients = pd.DataFrame(pats).drop(columns=["w0", "s0", "h0", "d0", "o0", "k0"])
    patients.attrs["screening"] = {"candidates": n_cand, "enrolled": n_patients,
                                   "not_eligible": n_cand - n_patients,
                                   "reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])),
                                   "criteria": inc}
    own_seed = co["seed"] if seed is None else seed
    feed = _add_rhythm(pd.DataFrame(rows), pats, n_days, cfg, own_seed)
    feed = _add_rate_spells(feed, pats, n_days, cfg, own_seed)
    return patients, _add_selfcare(feed, pats, n_days, cfg, own_seed)
