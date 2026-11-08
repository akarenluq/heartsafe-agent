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


def load_outputs(folder):
    """Shared-format JSON outputs from a pipeline (e.g. imaging/results/clean/exams)."""
    folder = Path(folder) if folder else None
    if not folder or not folder.exists():
        return []
    return [json.load(open(p)) for p in sorted(folder.glob("*.json"))]


def ecg_output(patient, true_episode, cfg, rng, pool=None):
    """One ECG model output for a patient-day, from real outputs if given, else simulated."""
    if pool:
        label = "abnormal" if true_episode else "normal"
        cands = [o for o in pool if o.get("_eval", {}).get("label") == label] or pool
        o = dict(random.Random(int(rng.integers(1e9))).choice(cands))
        o["source"] = "ecg_model"
        return o
    a, b = cfg["simulation"]["ecg_quality_beta"][patient["remoteness"]]
    quality = float(rng.beta(a, b))
    p_true = 0.8 if true_episode else 0.12
    abnormal = rng.random() < p_true
    prob = float(np.clip(rng.normal(0.8 if abnormal else 0.2, 0.12), 0.01, 0.99))
    # Poor signals make the model less certain and more often wrong.
    if quality < 0.5 and rng.random() < 0.4:
        prob = float(np.clip(1 - prob + rng.normal(0, 0.1), 0.01, 0.99))
    unc = float(np.clip(0.1 + (1 - quality) * 0.5 + rng.normal(0, 0.05), 0, 1))
    return {"source": "simulated", "modality": "ecg_single_lead",
            "prediction": {"abnormal_prob": round(prob, 3)}, "probability": round(prob, 3),
            "uncertainty": round(unc, 3), "quality": {"sqi": round(quality, 3)}}


def mri_output(true_episode, rng, pool=None):
    """One cardiac MRI result, from the imaging pipeline's exam JSONs if available."""
    if pool:
        o = dict(random.Random(int(rng.integers(1e9))).choice(pool))
        o.pop("_eval", None)
        o["source"] = "imaging_pipeline"
        return o
    lvef = float(np.clip(rng.normal(35 if true_episode else 52, 8), 10, 75))
    unc = float(np.clip(rng.normal(0.18, 0.07), 0.02, 0.6))
    flags = ["high_uncertainty"] if unc > 0.27 else []
    return {"source": "simulated", "modality": "cardiac_mri", "prediction": {"lvef_pct": round(lvef, 1)},
            "uncertainty": round(unc, 3), "flags": flags,
            "suggested_action": "escalate" if flags else "accept"}
