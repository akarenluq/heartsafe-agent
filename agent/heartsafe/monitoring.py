"""Monitoring-platform adapter and alert rules.

HeartSafe sits on top of an existing remote-monitoring platform; it does not
replace the platform's alerts. The main reference is Medly (UHN), which is used in
remote and Indigenous communities; other examples are TELUS Home Health Monitoring
and Cloud DX. These platforms are not open source, so `PlatformAdapter` reads a
generic daily export (CSV with one row per patient-day) and `platform_levels`
approximates a normal / caution / critical alert, modelled on Medly, with common
heart-failure telemonitoring rules. A real deployment would swap in the
platform's own export and alert level.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = {0: "normal", 1: "caution", 2: "critical"}
REQUIRED = ["patient_id", "day", "submitted", "weight_kg", "sbp", "hr", "sob", "swelling"]


class PlatformAdapter:
    """Reads a platform daily export. Ground-truth columns (gt_*) are kept apart."""

    def __init__(self, feed: pd.DataFrame):
        missing = [c for c in REQUIRED if c not in feed.columns]
        if missing:
            raise ValueError(f"Monitoring feed is missing columns {missing}")
        self.feed = feed.sort_values(["patient_id", "day"]).reset_index(drop=True)

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(Path(path)))

    def observable(self):
        """What the agent may see: no ground truth."""
        return self.feed[[c for c in self.feed.columns if not c.startswith("gt_")]]


def _gains(weights):
    s = pd.Series(weights)
    ref2 = s.shift(1).rolling(2, min_periods=1).min()
    ref7 = s.shift(1).rolling(7, min_periods=1).min()
    return (s - ref2).to_numpy(), (s - ref7).to_numpy()


def rule_levels(weight, sbp, hr, sob, swelling, rules):
    """Alert level and the rules that fired, for one patient's day-ordered arrays (NaN = no reading)."""
    g2, g7 = _gains(weight)
    levels = np.zeros(len(weight), dtype=int)
    fired = [[] for _ in weight]
    for i in range(len(weight)):
        if np.isnan(weight[i]):
            continue
        f = fired[i]
        if g2[i] > rules["weight_gain_2d_kg"]:
            f.append(f"weight +{g2[i]:.1f} kg in 2 days")
        if g7[i] > rules["weight_gain_7d_kg"]:
            f.append(f"weight +{g7[i]:.1f} kg in 7 days")
        if sob[i]:
            f.append("shortness of breath")
        if swelling[i]:
            f.append("swelling")
        crit = []
        if g7[i] > rules["weight_gain_critical_kg"]:
            crit.append(f"weight +{g7[i]:.1f} kg in 7 days")
        if sbp[i] < rules["sbp_low"] or sbp[i] > rules["sbp_high"]:
            crit.append(f"systolic BP {sbp[i]:.0f}")
        if hr[i] < rules["hr_low"] or hr[i] > rules["hr_high"]:
            crit.append(f"heart rate {hr[i]:.0f}")
        weight_rule = g2[i] > rules["weight_gain_2d_kg"] or g7[i] > rules["weight_gain_7d_kg"]
        if crit or (sob[i] and weight_rule):
            levels[i] = 2
            f.extend(c for c in crit if c not in f)
        elif f:
            levels[i] = 1
    return levels, fired


def platform_levels(feed, rules):
    """Adds alert_level (0/1/2) and alert_rules to a feed, per patient in day order."""
    out = []
    for pid, g in feed.groupby("patient_id", sort=False):
        g = g.sort_values("day")
        lv, fired = rule_levels(g.weight_kg.to_numpy(float), g.sbp.to_numpy(float), g.hr.to_numpy(float),
                                g.sob.to_numpy(bool), g.swelling.to_numpy(bool), rules)
        out.append(g.assign(alert_level=lv, alert_rules=["; ".join(x) for x in fired]))
    return pd.concat(out, ignore_index=True)
