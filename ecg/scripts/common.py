"""Shared helpers for the HeartSafe ECG path (single lead I)."""
from math import gcd

import numpy as np
import yaml
from scipy.signal import butter, filtfilt, find_peaks, resample_poly, welch
from scipy.stats import kurtosis, skew

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


def r_peaks(xf, fs):
    """R peaks on a band-passed ECG, polarity-free (energy envelope)."""
    k = max(1, int(0.08 * fs))
    env = np.convolve(xf ** 2, np.ones(k) / k, mode="same")
    peaks, _ = find_peaks(env, distance=int(0.3 * fs), height=0.3 * np.percentile(env, 99))
    return peaks


def quality_components(signal, fs):
    """Signal-quality measures used for repeat-measurement decisions.
      template_corr  mean correlation of each beat with the average beat (noise lowers it)
      baseline_power share of 0-40 Hz power below 1 Hz (baseline wander raises it)
      qrs_power      share of 5-40 Hz power in the QRS band 5-15 Hz (noise lowers it)
      hf_power       share of power above 40 Hz (muscle / electrode noise raises it)
    plus the agent's own index (flatline, clipping) and a robust heart rate."""
    x = np.asarray(signal, float)
    agent = signal_quality(x, fs)
    if not np.isfinite(x).all() or np.ptp(x) == 0:
        return {"agent_sqi": agent["sqi"], "agent_issues": agent["issues"], "template_corr": 0.0,
                "baseline_power": 1.0, "qrs_power": 0.0, "hf_power": 0.0, "heart_rate_bpm": 0.0}
    b, a = butter(3, [0.5, 40], btype="band", fs=fs)
    xf = filtfilt(b, a, x)
    pk = r_peaks(xf, fs)
    w = int(0.25 * fs)
    beats = np.array([xf[p - w:p + w] for p in pk if p - w >= 0 and p + w <= len(xf)])
    if len(beats) >= 3:
        t = beats.mean(0)
        c = [np.corrcoef(bt, t)[0, 1] for bt in beats]
        tc = float(np.nanmean(c)) if np.isfinite(c).any() else 0.0
    else:
        tc = 0.0
    f, P = welch(x - x.mean(), fs=fs, nperseg=min(len(x), 4 * fs))
    band = P[f <= 40].sum() + 1e-12
    return {"agent_sqi": agent["sqi"], "agent_issues": agent["issues"],
            "template_corr": round(tc, 4),
            "baseline_power": round(float(P[f < 1].sum() / band), 4),
            "qrs_power": round(float(P[(f >= 5) & (f <= 15)].sum() / (P[(f >= 5) & (f <= 40)].sum() + 1e-12)), 4),
            "hf_power": round(float(P[f > 40].sum() / (P.sum() + 1e-12)), 4),
            "heart_rate_bpm": round(60 * len(pk) / (len(x) / fs), 1)}


def recording_features(signal, fs):
    """Acquisition characteristics that differ between ECG devices (amplitude scale,
    spectral shape, smoothness): the input to the device-shift score."""
    x = np.asarray(signal, float)
    s = x.std() + 1e-6
    f, P = welch(x - x.mean(), fs=fs, nperseg=min(len(x), 2 * fs))
    Pt = P.sum() + 1e-12
    bands = [(0, 0.5), (0.5, 5), (5, 15), (15, 40), (40, fs)]
    return np.array([np.log(np.ptp(x) + 1e-6), np.log(s),
                     np.log(np.abs(np.diff(x)).mean() / s + 1e-6),
                     *[np.log(P[(f >= lo) & (f < hi)].sum() / Pt + 1e-8) for lo, hi in bands],
                     float(np.sum(f * P) / Pt), float(skew(x)), float(np.log(kurtosis(x) + 3 + 1e-6))])


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
