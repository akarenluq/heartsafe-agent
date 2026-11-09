"""HeartSafe vs the monitoring platform alone, on the simulated cohort's ground truth."""
import numpy as np
import pandas as pd

from .monitoring import platform_levels


def episodes(feed):
    """True decompensation episodes that the platform alerted on at least once."""
    out = []
    for pid, g in feed.groupby("patient_id"):
        g = g.sort_values("day")
        ep = g[g.gt_episode]
        if ep.empty:
            continue
        start, end = int(ep.day.min()), int(ep.day.max())
        alerts = g[(g.day >= start) & (g.day <= end) & (g.alert_level > 0)]
        out.append({"patient_id": pid, "start": start, "end": end,
                    "first_alert_h": alerts.day.min() * 24 + 9.0 if len(alerts) else np.nan})
    return pd.DataFrame(out)


def _in_episode(feed_idx, pid, day, before=1, after=3):
    return any(feed_idx.get((pid, d), False) for d in range(day - before, day + after + 1))


def fault_driven(feed, rules, patients=None):
    """Alert days that would NOT have alerted on the fault-free readings (and no true episode)."""
    clean = feed.drop(columns=["alert_level", "alert_rules"]).copy()
    cols = [c for c in ("weight_kg", "sbp", "hr", "dbp") if f"gt_{c}" in clean]
    clean[cols] = clean[[f"gt_{c}" for c in cols]].to_numpy()
    clean = platform_levels(clean, rules, patients)[["patient_id", "day", "alert_level"]].rename(columns={"alert_level": "clean_level"})
    f = feed.merge(clean, on=["patient_id", "day"])
    return f[(f.alert_level > 0) & (f.clean_level == 0) & (~f.gt_episode)]


def _sens_spec(flagged, positive):
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    return {"n": int(len(flagged)), "n_positive": int(positive.sum()),
            "sensitivity": round(float(flagged[positive].mean()), 3) if positive.any() else None,
            "specificity": round(float(1 - flagged[~positive].mean()), 3) if (~positive).any() else None}


def resolution(clinical, baseline, horizon_h, followup_cfg):
    """Follow-up completion (resolved within the resolve deadline) and time to resolution.
    Items still open at the end count as not resolved; only items whose deadline fell
    inside the simulated window are scored for completion."""
    out = {}
    if followup_cfg is None:
        return out
    dl = followup_cfg["resolve_deadline_h"]
    a = clinical.assign(deadline=clinical.created_h + clinical.severity.map(dl))
    b = baseline.assign(deadline=baseline.created_h + baseline.severity.map(dl))
    for name, d in (("heartsafe", a), ("platform_alone", b)):
        due = d[d.deadline <= horizon_h]
        hours = (d.resolved_h - d.created_h)
        out[name] = {"items": int(len(d)),
                     "resolved_within_deadline": round(float((due.resolved_h <= due.deadline).mean()), 3) if len(due) else None,
                     "resolved_by_end": round(float(d.resolved_h.notna().mean()), 3) if len(d) else None,
                     "median_hours_to_resolution": round(float(hours.median()), 1) if hours.notna().any() else None}
    return out


def stalled(clinical, baseline):
    """Cases that would have stalled (clinician forgot) vs how many HeartSafe caught with a reminder."""
    if "gt_stalled" not in clinical:
        return {}
    s = clinical[clinical.gt_stalled.fillna(False).astype(bool)]
    return {"heartsafe": {"simulated_to_stall": int(len(s)), "flagged_by_reminder": int((s.reminders > 0).sum()),
                          "resolved_by_end": int(s.resolved_h.notna().sum())},
            "platform_alone": {"simulated_to_stall": int(baseline.lost.sum()), "flagged": 0,
                            "note": "no tracking: a forgotten alert is never found"}}


def model_evidence(decisions):
    """How the real (or simulated) ECG and MRI results were judged, against injected faults."""
    out = {}
    if "ecg_checked" in decisions:
        e = decisions[decisions.ecg_checked.fillna(False).astype(bool)]
        if len(e):
            out["ecg"] = {**_sens_spec(~e.ecg_reliable.astype(bool), e.gt_ecg_fault.fillna("none") != "none"),
                          "unreliable_share": round(float((~e.ecg_reliable.astype(bool)).mean()), 3),
                          "actions": e.ecg_action.value_counts().to_dict()}
    if "mri_reliable" in decisions:
        m = decisions[decisions.mri_reliable.notna()]
        if len(m):
            out["mri"] = {**_sens_spec(~m.mri_reliable.astype(bool), m.gt_mri_unreliable.astype(bool)),
                          "review_cases": int((m.decision == "imaging_review").sum()),
                          "accepted": int((m.decision == "imaging_accept").sum()),
                          "regional_scanner_share": round(float(m.gt_mri_regional.astype(bool).mean()), 3),
                          "note": "positive = injected fault, LVEF error > 10 points or LV Dice < 0.8"}
    return out


def evaluate(feed, patients, decisions, cases, baseline, horizon_h, rules=None, followup_cfg=None):
    patient_months = len(patients) * (feed.day.max() + 1) / 30.0
    ep_idx = {(r.patient_id, r.day): r.gt_episode for r in feed[["patient_id", "day", "gt_episode"]].itertuples()}
    eps = episodes(feed)
    eps = eps[eps.first_alert_h.notna()]
    cases = cases.copy()
    clinical = cases[cases.kind == "clinical"]
    rem = patients.set_index("patient_id").remoteness

    # Episode-level: did a clinician acknowledge, and how fast after the platform's first alert?
    rows = []
    esc = decisions[decisions.decision.isin(["escalate", "escalate_update"])]
    for e in eps.itertuples():
        d = esc[(esc.patient_id == e.patient_id) & (esc.day >= e.start - 1) & (esc.day <= e.end)]
        ack_a = clinical[clinical.case_id.isin(d.case_id)].acknowledged_h.min() if len(d) else np.nan
        b = baseline[(baseline.patient_id == e.patient_id) & (baseline.day >= e.start - 1) & (baseline.day <= e.end)]
        ack_b = b.acknowledged_h.min() if len(b) else np.nan
        rows.append({"patient_id": e.patient_id, "remoteness": rem[e.patient_id], "escalated": len(d) > 0,
                     "ttack_agent": ack_a - e.first_alert_h if pd.notna(ack_a) else np.nan,
                     "ttack_platform": ack_b - e.first_alert_h if pd.notna(ack_b) else np.nan})
    er = pd.DataFrame(rows)

    # False alarms reaching the nurse queue.
    false_agent = sum(not _in_episode(ep_idx, r.patient_id, int(r.created_h // 24)) for r in clinical.itertuples())
    false_platform = sum(not _in_episode(ep_idx, r.patient_id, r.day) for r in baseline.itertuples())

    # Alerts caused by faulty readings (no true episode): how many did HeartSafe stop?
    fa = fault_driven(feed, rules, patients)[["patient_id", "day"]] if rules else \
        feed[(feed.alert_level > 0) & (feed.gt_fault != "none") & (~feed.gt_episode)][["patient_id", "day"]]
    # one decision per alert day: leave out records that are not a reply to the day's alert
    # (nurse-confirmed scale, imaging results), which can share a day with it
    alert_dec = decisions[~decisions.decision.isin(["scale_verified", "imaging_accept", "imaging_review"])]
    alert_dec = alert_dec.drop_duplicates(["patient_id", "day"], keep="last")
    fa = fa.merge(alert_dec[["patient_id", "day", "decision"]], on=["patient_id", "day"], how="left")
    stopped = fa.decision.isin(["repeat", "filtered"]).mean() if len(fa) else np.nan
    held = fa.decision.isin(["nurse_verify", "nurse_verify_update"]).mean() if len(fa) else np.nan
    scale_checks = cases[cases.kind == "scale_check"]

    def share_within(x, h):
        return float(np.mean(np.nan_to_num(x, nan=np.inf) <= h)) if len(x) else np.nan

    res = {
        "cohort": {"patients": int(len(patients)), "days": int(feed.day.max() + 1),
                   "patient_days": int(len(feed)), "platform_alert_days": int((feed.alert_level > 0).sum()),
                   "true_episodes_alerted": int(len(er))},
        "true_episodes": {
            "escalated_by_heartsafe": round(float(er.escalated.mean()), 3) if len(er) else None,
            "acknowledged_by_end": {"heartsafe": round(float(er.ttack_agent.notna().mean()), 3),
                                    "platform_alone": round(float(er.ttack_platform.notna().mean()), 3)},
            "median_hours_first_alert_to_ack": {"heartsafe": round(float(er.ttack_agent.median()), 1),
                                                "platform_alone": round(float(er.ttack_platform.median()), 1)},
            "acknowledged_within_24h": {"heartsafe": round(share_within(er.ttack_agent, 24), 3),
                                        "platform_alone": round(share_within(er.ttack_platform, 24), 3)},
            "mean_hours_first_alert_to_ack": {"heartsafe": round(float(er.ttack_agent.mean()), 1),
                                              "platform_alone": round(float(er.ttack_platform.mean()), 1)},
            "hours_to_ack_by_remoteness": {
                k: {"heartsafe_mean": round(float(g.ttack_agent.mean()), 1),
                    "platform_alone_mean": round(float(g.ttack_platform.mean()), 1),
                    "heartsafe_median": round(float(g.ttack_agent.median()), 1),
                    "platform_alone_median": round(float(g.ttack_platform.median()), 1), "n": int(len(g))}
                for k, g in er.groupby("remoteness")},
        },
        "fault_driven_alerts": {"n": int(len(fa)), "stopped_by_heartsafe": round(float(stopped), 3),
                                "held_for_nurse_verification": round(float(held), 3),
                                "stopped_by_platform_alone": 0.0},
        "scale_checks": {"nurse_verifications": int(len(scale_checks)),
                         "confirmed_new_scale": int((decisions.decision == "scale_verified").sum()),
                         "real_gain_escalated": int((decisions.get("flags", pd.Series(dtype=str))
                                                     == "scale_check_rejected").sum())},
        "nurse_queue_per_100_patient_months": {
            "heartsafe_clinical_cases": round(100 * len(clinical) / patient_months, 1),
            "heartsafe_nurse_verifications": round(100 * len(scale_checks) / patient_months, 1),
            "platform_alone_alert_items": round(100 * len(baseline) / patient_months, 1),
            "false_alarms_heartsafe": round(100 * false_agent / patient_months, 1),
            "false_alarms_platform_alone": round(100 * false_platform / patient_months, 1)},
        "agent_actions": decisions.decision.value_counts().to_dict(),
        "resolution": resolution(clinical, baseline, horizon_h, followup_cfg),
        "stalled_cases": stalled(clinical, baseline),
        "model_evidence": model_evidence(decisions),
        "followup": {
            "cases": int(len(cases)), "reminders_sent": int(cases.reminders.sum()),
            "clinical_cases_ack_within_deadline": round(float(
                (clinical.acknowledged_h <= clinical.ack_deadline_h).mean()), 3) if len(clinical) else None,
            "platform_alone_alert_items_never_acknowledged": int(baseline.lost.sum()),
            "open_at_end": int((cases.state != "resolved").sum())},
    }
    return res, er
