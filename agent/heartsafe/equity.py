"""Equity monitor: does HeartSafe work equally well across communities and patient groups?

Uses only what a deployed system can observe (no ground truth): missed-reading rate,
data-quality repeat rate, time to acknowledge and acknowledgement within deadline.
A subgroup is flagged when its whole bootstrap 95% CI is worse than the overall
estimate. Flags become safeguard proposals that an administrator must approve.
"""
import numpy as np
import pandas as pd

# metric: (unit of analysis, column, higher_is_worse, label)
METRICS = {
    "missed_day_rate": ("patient", "missed_rate", True, "Share of days with no reading"),
    "repeat_rate": ("patient", "repeat_rate", True, "Share of platform alerts needing a repeat or correction"),
    "ecg_unreliable_rate": ("patient", "ecg_unreliable_rate", True, "Share of ECG checks not reliable enough to use"),
    "hours_to_ack": ("case", "hours_to_ack", True, "Hours from case creation to acknowledgement"),
    "ack_within_deadline": ("case", "ack_in_time", False, "Share of cases acknowledged within deadline"),
    "patient_action_done": ("case", "patient_acted", False,
                            "Share of resolved cases where the patient side completed the instruction"),
}


def _boot(x, n, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tables(feed, patients, decisions, cases):
    obs = feed[[c for c in feed.columns if not c.startswith("gt_")]]
    per_p = obs.groupby("patient_id").agg(days=("day", "size"), missed=("submitted", lambda s: (~s).sum()),
                                          alerts=("alert_level", lambda s: (s > 0).sum()))
    rep = decisions[decisions.decision.isin(["repeat", "filtered", "nurse_verify", "nurse_verify_update"])].groupby("patient_id").size()
    per_p["missed_rate"] = per_p.missed / per_p.days
    per_p["repeat_rate"] = (rep.reindex(per_p.index).fillna(0) / per_p.alerts.replace(0, np.nan))
    per_p["ecg_unreliable_rate"] = np.nan
    if "ecg_checked" in decisions:
        e = decisions[decisions.ecg_checked.fillna(False).astype(bool)]
        if len(e):
            per_p["ecg_unreliable_rate"] = (1 - e.ecg_reliable.astype(float)).groupby(e.patient_id).mean() \
                .reindex(per_p.index)
    per_p = per_p.join(patients.set_index("patient_id"))
    c = cases[cases.kind == "clinical"].merge(patients, on="patient_id")
    c["hours_to_ack"] = c.acknowledged_h - c.created_h
    c["ack_in_time"] = (c.acknowledged_h <= c.ack_deadline_h).astype(float)
    c.loc[c.acknowledged_h.isna(), "ack_in_time"] = np.nan
    c["patient_acted"] = np.nan
    if "loop_state" in c:
        reason = c.get("loop_reason", pd.Series("", index=c.index)).fillna("")
        acted = c.loop_state.isin(["action_done", "closed"]) | reason.str.startswith("readings did not improve")
        failed = reason.isin(["patient could not be reached", "instruction not completed"])
        c.loc[acted, "patient_acted"] = 1.0
        c.loc[failed, "patient_acted"] = 0.0
    return per_p.reset_index(), c


def audit(feed, patients, decisions, cases, cfg, seed=0):
    eq = cfg["equity"]
    rng = np.random.default_rng(seed)
    per_p, per_c = tables(feed, patients, decisions, cases)
    rows = []
    for key, (unit, col, worse_high, label) in METRICS.items():
        df = per_p if unit == "patient" else per_c
        mean, lo, hi = _boot(df[col], eq["n_boot"], rng)
        rows.append({"metric": key, "label": label, "group": "overall", "value": "all", "n": int(df[col].notna().sum()),
                     "mean": mean, "ci_low": lo, "ci_high": hi, "gap": False})
        for g in eq["groups"]:
            for v, sub in df.groupby(df[g].astype(str)):
                m, l, h = _boot(sub[col], eq["n_boot"], rng)
                n = int(sub[col].notna().sum())
                gap = n >= eq["min_n"] and ((l > mean) if worse_high else (h < mean))
                rows.append({"metric": key, "label": label, "group": g, "value": v, "n": n,
                             "mean": m, "ci_low": l, "ci_high": h, "gap": bool(gap)})
    res = pd.DataFrame(rows)

    proposals = []
    for r in res[res.gap].itertuples():
        if r.metric in ("hours_to_ack", "ack_within_deadline"):
            proposals.append({"id": f"tighter_ack_deadline:{r.group}={r.value}", "type": "tighter_ack_deadline",
                              "field": r.group, "value": r.value, "factor": eq["tighter_deadline_factor"],
                              "evidence": f"{r.label}: {r.mean:.2f} (95% CI {r.ci_low:.2f}-{r.ci_high:.2f}, n={r.n})",
                              "effect": "earlier reminders and backup routing for this group; adds reminders",
                              "approved": False})
        elif r.metric == "patient_action_done":
            proposals.append({"id": f"earlier_patient_followup:{r.group}={r.value}", "type": "earlier_patient_followup",
                              "field": r.group, "value": r.value,
                              "factor": cfg["patient_loop"]["safeguard_factor"],
                              "evidence": f"{r.label}: {r.mean:.2f} (95% CI {r.ci_low:.2f}-{r.ci_high:.2f}, n={r.n})",
                              "effect": "contact the caregiver or community health worker earlier for this group; "
                                        "adds follow-up calls", "approved": False})
        elif r.metric == "ecg_unreliable_rate":
            proposals.append({"id": f"ecg_support:{r.group}={r.value}", "type": "recommendation",
                              "field": r.group, "value": r.value,
                              "evidence": f"{r.label}: {r.mean:.2f} (95% CI {r.ci_low:.2f}-{r.ci_high:.2f}, n={r.n})",
                              "effect": "recommend ECG recording coaching and device checks for this group "
                                        "(program decision, not automated)", "approved": False})
        else:
            proposals.append({"id": f"support_outreach:{r.group}={r.value}", "type": "recommendation",
                              "field": r.group, "value": r.value,
                              "evidence": f"{r.label}: {r.mean:.2f} (95% CI {r.ci_low:.2f}-{r.ci_high:.2f}, n={r.n})",
                              "effect": "recommend device, connectivity and measurement-support outreach "
                                        "(program decision, not automated)", "approved": False})
    # one proposal per id
    seen, uniq = set(), []
    for p in proposals:
        if p["id"] not in seen:
            seen.add(p["id"])
            uniq.append(p)
    return res, uniq
