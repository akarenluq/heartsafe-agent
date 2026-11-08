"""Shared helpers for the HeartSafe ECG path (single lead I)."""
from math import gcd

import numpy as np
import yaml
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

LABEL_NAMES = {0: "normal", 1: "abnormal", -1: "unknown"}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resample(x, fs_in, fs_out):
    x = np.asarray(x, np.float64)
    if int(fs_in) == int(fs_out):
        return x.astype(np.float32)
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, int(fs_out) // g, int(fs_in) // g).astype(np.float32)


def fix_length(x, n, fs):
    """Crop to n samples (skipping the first 2 s of long recordings) or tile short ones."""
    if len(x) >= n:
        start = min(int(2 * fs), len(x) - n)
        return x[start:start + n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def model_input(X, fs):
    """Band-pass 0.5-40 Hz and z-score each recording. X: (N, T) raw mV."""
    b, a = butter(3, [0.5, 40], btype="band", fs=fs)
    Y = filtfilt(b, a, np.asarray(X, np.float64), axis=1)
    Y = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-6)
    return np.nan_to_num(np.clip(Y, -10, 10)).astype(np.float32)


def signal_quality(signal, fs, hr_range=(30, 200)):
    """0-1 signal quality index. Same scoring as agent/heartsafe/signals.ecg_quality_index,
    so the agent and this pipeline agree on what 'poor quality' means."""
    x = np.asarray(signal, float)
    if x.size < fs * 5 or not np.isfinite(x).all():
        return {"sqi": 0.0, "issues": ["too short or invalid"], "heart_rate_bpm": None}
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
    hr_ok = hr_range[0] <= hr <= hr_range[1]
    if not hr_ok:
        issues.append(f"heart rate {hr:.0f} implausible")
    score = max(0.0, 1 - 2 * flat - 4 * clip - noise - (0 if hr_ok else 0.4))
    return {"sqi": round(score, 3), "issues": issues, "heart_rate_bpm": round(hr, 1)}


FAULTS = ("noise", "flatline", "clipping", "baseline_wander")


def inject_fault(x, fs, kind, rng):
    """Known, realistic recording faults for evaluating the reliability checks."""
    x = np.asarray(x, np.float64).copy()
    n, s, ptp = len(x), np.std(x) + 1e-6, np.ptp(x) + 1e-6
    if kind == "noise":            # loose electrode / muscle noise
        x += rng.normal(0, rng.uniform(0.5, 1.0) * s, n)
    elif kind == "flatline":       # lead-off for 3-6 s
        L = int(rng.uniform(3, 6) * fs)
        st = int(rng.integers(0, n - L))
        x[st:st + L] = x[st]
    elif kind == "clipping":       # amplifier saturation
        q = rng.uniform(5, 15)
        x = np.clip(x, np.percentile(x, q), np.percentile(x, 100 - q))
    elif kind == "baseline_wander":  # breathing / movement
        t = np.arange(n) / fs
        x += rng.uniform(1.0, 2.5) * ptp * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6.28))
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def age_band(age):
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if np.isnan(a):
        return "unknown"
    return "<40" if a < 40 else ("40-59" if a < 60 else "60+")
