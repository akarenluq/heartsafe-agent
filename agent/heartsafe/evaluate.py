"""HeartSafe vs Medly alone, on the simulated cohort's ground truth."""
import numpy as np
import pandas as pd

from .medly import medly_levels


def episodes(feed):
    """True decompensation episodes that Medly alerted on at least once."""
    out = []
    for pid, g in feed.groupby("patient_id"):
        g = g.sort_values("day")
        ep = g[g.gt_episode]
        if ep.empty:
            continue
        start, end = int(ep.day.min()), int(ep.day.max())
        alerts = g[(g.day >= start) & (g.day <= end) & (g.medly_level > 0)]
        out.append({"patient_id": pid, "start": start, "end": end,
                    "first_alert_h": alerts.day.min() * 24 + 9.0 if len(alerts) else np.nan})
    return pd.DataFrame(out)


def _in_episode(feed_idx, pid, day, before=1, after=3):
    return any(feed_idx.get((pid, d), False) for d in range(day - before, day + after + 1))


def fault_driven(feed, rules):
    """Alert days that would NOT have alerted on the fault-free readings (and no true episode)."""
    clean = feed.drop(columns=["medly_level", "medly_rules"]).copy()
    clean[["weight_kg", "sbp", "hr"]] = clean[["gt_weight_kg", "gt_sbp", "gt_hr"]].to_numpy()
    clean = medly_levels(clean, rules)[["patient_id", "day", "medly_level"]].rename(columns={"medly_level": "clean_level"})
    f = feed.merge(clean, on=["patient_id", "day"])
    return f[(f.medly_level > 0) & (f.clean_level == 0) & (~f.gt_episode)]


def evaluate(feed, patients, decisions, cases, baseline, horizon_h, rules=None):
    patient_months = len(patients) * (feed.day.max() + 1) / 30.0
    ep_idx = {(r.patient_id, r.day): r.gt_episode for r in feed[["patient_id", "day", "gt_episode"]].itertuples()}
    eps = episodes(feed)
    eps = eps[eps.first_alert_h.notna()]
    cases = cases.copy()
    clinical = cases[cases.kind == "clinical"]
    rem = patients.set_index("patient_id").remoteness

    # Episode-level: did a clinician acknowledge, and how fast after Medly's first alert?
    rows = []
    esc = decisions[decisions.decision.isin(["escalate", "escalate_update"])]
    for e in eps.itertuples():
        d = esc[(esc.patient_id == e.patient_id) & (esc.day >= e.start - 1) & (esc.day <= e.end)]
        ack_a = clinical[clinical.case_id.isin(d.case_id)].acknowledged_h.min() if len(d) else np.nan
        b = baseline[(baseline.patient_id == e.patient_id) & (baseline.day >= e.start - 1) & (baseline.day <= e.end)]
        ack_b = b.acknowledged_h.min() if len(b) else np.nan
        rows.append({"patient_id": e.patient_id, "remoteness": rem[e.patient_id], "escalated": len(d) > 0,
                     "ttack_agent": ack_a - e.first_alert_h if pd.notna(ack_a) else np.nan,
                     "ttack_medly": ack_b - e.first_alert_h if pd.notna(ack_b) else np.nan})
    er = pd.DataFrame(rows)

    # False alarms reaching the nurse queue.
    false_agent = sum(not _in_episode(ep_idx, r.patient_id, int(r.created_h // 24)) for r in clinical.itertuples())
    false_medly = sum(not _in_episode(ep_idx, r.patient_id, r.day) for r in baseline.itertuples())

    # Alerts caused by faulty readings (no true episode): how many did HeartSafe stop?
    fa = fault_driven(feed, rules)[["patient_id", "day"]] if rules else \
        feed[(feed.medly_level > 0) & (feed.gt_fault != "none") & (~feed.gt_episode)][["patient_id", "day"]]
    fa = fa.merge(decisions[["patient_id", "day", "decision"]], on=["patient_id", "day"], how="left")
    stopped = fa.decision.isin(["repeat", "filtered"]).mean() if len(fa) else np.nan

    def share_within(x, h):
        return float(np.mean(np.nan_to_num(x, nan=np.inf) <= h)) if len(x) else np.nan

    res = {
        "cohort": {"patients": int(len(patients)), "days": int(feed.day.max() + 1),
                   "patient_days": int(len(feed)), "medly_alert_days": int((feed.medly_level > 0).sum()),
                   "true_episodes_alerted": int(len(er))},
        "true_episodes": {
            "escalated_by_heartsafe": round(float(er.escalated.mean()), 3) if len(er) else None,
            "acknowledged_by_end": {"heartsafe": round(float(er.ttack_agent.notna().mean()), 3),
                                    "medly_alone": round(float(er.ttack_medly.notna().mean()), 3)},
            "median_hours_first_alert_to_ack": {"heartsafe": round(float(er.ttack_agent.median()), 1),
                                                "medly_alone": round(float(er.ttack_medly.median()), 1)},
            "acknowledged_within_24h": {"heartsafe": round(share_within(er.ttack_agent, 24), 3),
                                        "medly_alone": round(share_within(er.ttack_medly, 24), 3)},
            "mean_hours_first_alert_to_ack": {"heartsafe": round(float(er.ttack_agent.mean()), 1),
                                              "medly_alone": round(float(er.ttack_medly.mean()), 1)},
            "hours_to_ack_by_remoteness": {
                k: {"heartsafe_mean": round(float(g.ttack_agent.mean()), 1),
                    "medly_alone_mean": round(float(g.ttack_medly.mean()), 1),
                    "heartsafe_median": round(float(g.ttack_agent.median()), 1),
                    "medly_alone_median": round(float(g.ttack_medly.median()), 1), "n": int(len(g))}
                for k, g in er.groupby("remoteness")},
        },
        "fault_driven_alerts": {"n": int(len(fa)), "stopped_by_heartsafe": round(float(stopped), 3),
                                "stopped_by_medly_alone": 0.0},
        "nurse_queue_per_100_patient_months": {
            "heartsafe_clinical_cases": round(100 * len(clinical) / patient_months, 1),
            "medly_alone_alert_items": round(100 * len(baseline) / patient_months, 1),
            "false_alarms_heartsafe": round(100 * false_agent / patient_months, 1),
            "false_alarms_medly_alone": round(100 * false_medly / patient_months, 1)},
        "agent_actions": decisions.decision.value_counts().to_dict(),
        "followup": {
            "cases": int(len(cases)), "reminders_sent": int(cases.reminders.sum()),
            "clinical_cases_ack_within_deadline": round(float(
                (clinical.acknowledged_h <= clinical.ack_deadline_h).mean()), 3) if len(clinical) else None,
            "medly_alone_alert_items_never_acknowledged": int(baseline.lost.sum()),
            "open_at_end": int((cases.state != "resolved").sum())},
    }
    return res, er
