"""ECG and cardiac MRI inputs for the reliability layer.

Real model outputs plug in through the shared HeartSafe JSON format
({prediction, probability, uncertainty, quality, shift_score, metadata, flags,
suggested_action}); the ECG and imaging pipelines write those files. Without them,
outputs are simulated so the coordination layer can be run and demoed end to end.
"""
import json
import random
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def ecg_quality_index(signal, fs):
    """0-1 quality score for a single-lead ECG: flatline, clipping, noise and heart-rate plausibility."""
    x = np.asarray(signal, float)
    if x.size < fs * 5 or not np.isfinite(x).all():
        return 0.0, ["too short or invalid"]
    issues = []
    dx = np.abs(np.diff(x))
    flat = float(np.mean(dx < 1e-6))
    if flat > 0.2:
        issues.append(f"flat {flat:.0%}")
    rng = np.ptp(x)
    clip = float(np.mean((x >= x.max() - 0.01 * rng) | (x <= x.min() + 0.01 * rng)))
    if clip > 0.05:
        issues.append(f"clipped {clip:.0%}")
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(x.size, 1 / fs)
    band = spec[(f >= 0.5) & (f <= 40)].sum()
    hf = spec[f > 40].sum()
    noise = float(hf / (band + hf + 1e-12))
    if noise > 0.3:
        issues.append(f"high-frequency noise {noise:.0%}")
    peaks, _ = find_peaks(x, distance=int(0.3 * fs), prominence=0.5 * np.std(x))
    hr = 60 * len(peaks) / (x.size / fs)
    if not 30 <= hr <= 200:
        issues.append(f"heart rate {hr:.0f} implausible")
    score = max(0.0, 1 - 2 * flat - 4 * clip - noise - (0.4 if not 30 <= hr <= 200 else 0))
    return round(score, 3), issues


def load_outputs(folders):
    """Shared-format JSON outputs from one or more pipeline folders (list or comma-separated),
    e.g. ../ecg/results/clean/exams,../ecg/results/faulty/exams."""
    if not folders:
        return []
    if isinstance(folders, (str, Path)):
        folders = [f for f in str(folders).split(",") if f]
    out = []
    for f in folders:
        p = Path(f)
        if p.exists():
            out += [json.load(open(x)) for x in sorted(p.glob("*.json"))]
    return out


def _pick(pool, rng, *filters):
    """Random item, narrowed by each filter in turn (a filter that leaves nothing is skipped)."""
    cands = pool
    for f in filters:
        c = [o for o in cands if f(o)]
        if c:
            cands = c
    return dict(random.Random(int(rng.integers(1e9))).choice(cands))


def _split_truth(o):
    """Separate ground truth (simulation and evaluation only) from what the agent may see."""
    gt = o.pop("_eval", {}) or {}
    return o, gt


def ecg_output(patient, true_episode, cfg, rng, pool=None, demo=False):
    """One ECG model output for a patient-day -> (output the agent sees, ground truth).

    With real outputs, the simulated world picks a recording whose label matches the
    patient's true state, from the fault-injected set at a remoteness-dependent rate.
    The demo patient gets a clean recording the pipeline accepted (the demo script)."""
    if pool:
        sc = cfg["simulation"]
        faulty = (not demo) and bool(rng.random() < sc["ecg_fault_share"][patient["remoteness"]])
        label = "abnormal" if true_episode else "normal"
        o = _pick(pool, rng, lambda o: o.get("_eval", {}).get("label") == label,
                  lambda o: (o.get("_eval", {}).get("fault", "none") != "none") == faulty,
                  (lambda o: o.get("suggested_action") == "accept") if demo else (lambda o: True))
        o["source"] = "ecg_model"
        return _split_truth(o)
    a, b = cfg["simulation"]["ecg_quality_beta"][patient["remoteness"]]
    quality = float(rng.beta(a, b))
    p_true = 0.8 if true_episode else 0.12
    abnormal = rng.random() < p_true
    prob = float(np.clip(rng.normal(0.8 if abnormal else 0.2, 0.12), 0.01, 0.99))
    # Poor signals make the model less certain and more often wrong.
    if quality < 0.5 and rng.random() < 0.4:
        prob = float(np.clip(1 - prob + rng.normal(0, 0.1), 0.01, 0.99))
    unc = float(np.clip(0.1 + (1 - quality) * 0.5 + rng.normal(0, 0.05), 0, 1))
    return ({"source": "simulated", "modality": "ecg_single_lead",
             "prediction": {"abnormal_prob": round(prob, 3)}, "probability": round(prob, 3),
             "uncertainty": round(unc, 3), "quality": {"sqi": round(quality, 3)}},
            {"fault": "low_quality" if quality < 0.5 else "none"})


def mri_output(patient, true_episode, cfg, rng, pool=None, demo=False):
    """One cardiac MRI result -> (output the agent sees, ground truth).

    With real outputs, rural and remote patients are more often scanned at a regional
    centre whose scanner the model never saw (M&Ms vendors C/D), and some exams come
    from the fault-injected set. The demo patient gets a regional scan the pipeline flagged."""
    if pool:
        sc = cfg["simulation"]
        seen = set(sc.get("imaging_seen_vendors", ["A", "B"]))
        regional = demo or bool(rng.random() < sc["regional_scanner_share"][patient["remoteness"]])
        faulty = (not demo) and bool(rng.random() < sc["mri_fault_share"][patient["remoteness"]])
        o = _pick(pool, rng,
                  lambda o: ((o.get("metadata") or {}).get("vendor") not in seen) == regional,
                  lambda o: (o.get("_eval", {}).get("fault", "none") != "none") == faulty,
                  # demo: a plausible result the pipeline flagged as uncertain
                  (lambda o: "high_uncertainty" in o.get("flags", []) and not any(
                      "range" in f or "fragment" in f for f in o.get("flags", []))) if demo else (lambda o: True),
                  lambda o: ((o.get("_eval", {}).get("gt_lvef_pct") or 60) < 50) == bool(true_episode))
        o["source"] = "imaging_pipeline"
        o, gt = _split_truth(o)
        bad = (gt.get("lvef_abs_err") or 0) > 10 or (gt.get("dice_lv") or 1) < 0.8
        return o, {"fault": gt.get("fault", "none"), "unreliable": bad or gt.get("fault", "none") != "none",
                   "regional": regional}
    lvef = float(np.clip(rng.normal(35 if true_episode else 52, 8), 10, 75))
    unc = float(np.clip(rng.normal(0.18, 0.07), 0.02, 0.6))
    flags = ["high_uncertainty"] if unc > 0.27 or demo else []
    return ({"source": "simulated", "modality": "cardiac_mri", "prediction": {"lvef_pct": round(lvef, 1)},
             "uncertainty": round(unc, 3), "flags": flags,
             "suggested_action": "escalate" if flags else "accept"},
            {"fault": "none", "unreliable": bool(flags), "regional": False})
