"""Monitoring-platform adapter and alert rules.

HeartSafe sits on top of an existing remote-monitoring platform; it does not
replace the platform's alerts. The main reference is Medly (UHN), which is used in
remote and Indigenous communities; other examples are TELUS Home Health Monitoring
and Cloud DX. These platforms are not open source, so `PlatformAdapter` reads a
generic daily export (CSV with one row per patient-day) and `platform_levels`
approximates a normal / caution / critical alert, modelled on Medly, with common
heart-failure telemonitoring rules. A real deployment would swap in the
platform's own export and alert level.

The rules were reviewed with practising cardiologists (Nov 2026): diastolic BP,
oxygen saturation, more symptoms, diuretic adherence and lab results were added,
and BP / heart-rate limits can be set per patient at enrollment.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = {0: "normal", 1: "caution", 2: "critical"}
REQUIRED = ["patient_id", "day", "submitted", "weight_kg", "sbp", "hr", "sob", "swelling"]
# Optional columns: a platform that does not export them gets these defaults (rule never fires).
OPTIONAL = {"dbp": np.nan, "spo2": np.nan, "chest_pain": False, "syncope": False, "orthopnea": False,
            "pnd": False, "diuretic_taken": True, "potassium": np.nan, "creatinine_umol": np.nan,
            # the patient's self-care log (no alert rule uses these; see selfcare.py)
            "sleep_hours": np.nan, "sleep_awakenings": np.nan, "breathing_sessions": np.nan}
SYMPTOMS = ["sob", "swelling", "orthopnea", "pnd", "chest_pain", "syncope"]
SYMPTOM_TEXT = {"sob": "shortness of breath", "swelling": "swelling", "orthopnea": "breathless lying flat",
                "pnd": "woke up breathless at night", "chest_pain": "chest pain", "syncope": "fainting"}
LIMIT_KEYS = ["sbp_low", "sbp_high", "dbp_high", "hr_low", "hr_high"]


class PlatformAdapter:
    """Reads a platform daily export. Ground-truth columns (gt_*) are kept apart."""

    def __init__(self, feed: pd.DataFrame):
        missing = [c for c in REQUIRED if c not in feed.columns]
        if missing:
            raise ValueError(f"Monitoring feed is missing columns {missing}")
        feed = feed.copy()
        for c, v in OPTIONAL.items():
            if c not in feed.columns:
                feed[c] = v
        self.feed = feed.sort_values(["patient_id", "day"]).reset_index(drop=True)

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(Path(path)))

    def observable(self):
        """What the agent may see: no ground truth."""
        return self.feed[[c for c in self.feed.columns if not c.startswith("gt_")]]


def patient_limits(rules, patient=None):
    """BP / heart-rate limits for one patient: the clinician's values from enrollment, else the defaults."""
    lim = {k: rules[k] for k in LIMIT_KEYS}
    for k in LIMIT_KEYS:
        v = (patient or {}).get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            lim[k] = float(v)
    return lim


def _gains(weights):
    s = pd.Series(weights)
    ref2 = s.shift(1).rolling(2, min_periods=1).min()
    ref7 = s.shift(1).rolling(7, min_periods=1).min()
    return (s - ref2).to_numpy(), (s - ref7).to_numpy()


def _missed_run(taken):
    """Consecutive days without the diuretic, ending on each day."""
    run, out = 0, []
    for t in taken:
        run = run + 1 if not t else 0
        out.append(run)
    return np.array(out)


def rule_levels(weight, sbp, hr, sob, swelling, rules, extra=None, limits=None, creatinine_baseline=None):
    """Alert level and the rules that fired, for one patient's day-ordered arrays (NaN = no reading).

    extra: optional dict of arrays (dbp, spo2, orthopnea, pnd, chest_pain, syncope, diuretic_taken,
    potassium, creatinine_umol). limits: per-patient BP / HR limits (see patient_limits)."""
    n = len(weight)
    ex = extra or {}
    get = lambda k, fill: np.asarray(ex[k]) if k in ex else np.full(n, fill)  # noqa: E731
    dbp, spo2 = get("dbp", np.nan).astype(float), get("spo2", np.nan).astype(float)
    k_, creat = get("potassium", np.nan).astype(float), get("creatinine_umol", np.nan).astype(float)
    sym = {"sob": np.asarray(sob, bool), "swelling": np.asarray(swelling, bool)}
    for s in ("orthopnea", "pnd", "chest_pain", "syncope"):
        sym[s] = get(s, False).astype(bool)
    missed = _missed_run(get("diuretic_taken", True).astype(bool))
    lim = limits or patient_limits(rules)
    g2, g7 = _gains(weight)
    levels = np.zeros(n, dtype=int)
    fired = [[] for _ in range(n)]
    for i in range(n):
        if np.isnan(weight[i]):
            continue
        f, crit = fired[i], []
        if g2[i] > rules["weight_gain_2d_kg"]:
            f.append(f"weight +{g2[i]:.1f} kg in 2 days")
        if g7[i] > rules["weight_gain_7d_kg"]:
            f.append(f"weight +{g7[i]:.1f} kg in 7 days")
        weight_rule = g2[i] > rules["weight_gain_2d_kg"] or g7[i] > rules["weight_gain_7d_kg"]
        for s in ("sob", "swelling", "orthopnea", "pnd"):
            if sym[s][i]:
                f.append(SYMPTOM_TEXT[s])
        for s in ("chest_pain", "syncope"):
            if sym[s][i]:
                crit.append(SYMPTOM_TEXT[s])
        if missed[i] >= rules["missed_diuretic_days"]:
            f.append(f"diuretic not taken for {missed[i]} days")
        if not np.isnan(spo2[i]) and spo2[i] < rules["spo2_caution"]:
            (crit if spo2[i] < rules["spo2_critical"] else f).append(f"oxygen saturation {spo2[i]:.0f}%")
        if not np.isnan(k_[i]):
            if k_[i] < rules["potassium_critical"][0] or k_[i] > rules["potassium_critical"][1]:
                crit.append(f"potassium {k_[i]:.1f} mmol/L")
            elif k_[i] < rules["potassium_caution"][0] or k_[i] > rules["potassium_caution"][1]:
                f.append(f"potassium {k_[i]:.1f} mmol/L")
        if not np.isnan(creat[i]) and creatinine_baseline is not None and not np.isnan(creatinine_baseline) \
                and creat[i] - creatinine_baseline >= rules["creatinine_rise_umol"]:
            f.append(f"creatinine {creat[i]:.0f} umol/L (baseline {creatinine_baseline:.0f})")
        if g7[i] > rules["weight_gain_critical_kg"]:
            crit.append(f"weight +{g7[i]:.1f} kg in 7 days")
        if sbp[i] < lim["sbp_low"] or sbp[i] > lim["sbp_high"]:
            crit.append(f"systolic BP {sbp[i]:.0f}")
        if not np.isnan(dbp[i]) and dbp[i] > lim["dbp_high"]:
            crit.append(f"diastolic BP {dbp[i]:.0f}")
        if hr[i] < lim["hr_low"] or hr[i] > lim["hr_high"]:
            crit.append(f"heart rate {hr[i]:.0f}")
        breathless = sym["sob"][i] or sym["orthopnea"][i] or sym["pnd"][i]
        if crit or (breathless and weight_rule):
            levels[i] = 2
            f.extend(c for c in crit if c not in f)
        elif f:
            levels[i] = 1
    return levels, fired


def feed_arrays(g):
    """Rule inputs from one patient's day-ordered feed rows."""
    extra = {c: g[c].to_numpy() for c in OPTIONAL if c in g}
    return (g.weight_kg.to_numpy(float), g.sbp.to_numpy(float), g.hr.to_numpy(float),
            g.sob.to_numpy(bool), g.swelling.to_numpy(bool), extra)


def platform_levels(feed, rules, patients=None):
    """Adds alert_level (0/1/2) and alert_rules to a feed, per patient in day order.

    patients (optional DataFrame): per-patient limits and creatinine baseline from enrollment."""
    pats = patients.set_index("patient_id").to_dict("index") if patients is not None else {}
    for c, v in OPTIONAL.items():
        if c not in feed.columns:
            feed = feed.assign(**{c: v})
    out = []
    for pid, g in feed.groupby("patient_id", sort=False):
        g = g.sort_values("day")
        p = pats.get(pid, {})
        w, sbp, hr, sob, sw, extra = feed_arrays(g)
        lv, fired = rule_levels(w, sbp, hr, sob, sw, rules, extra, patient_limits(rules, p),
                                p.get("creatinine_baseline_umol"))
        out.append(g.assign(alert_level=lv, alert_rules=["; ".join(x) for x in fired]))
    return pd.concat(out, ignore_index=True)
