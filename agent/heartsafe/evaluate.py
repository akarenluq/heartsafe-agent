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


def emergencies(decisions, cases, ep_idx):
    """Emergency level: chest pain or fainting with a reliable abnormal ECG. Minutes to the
    clinician's video call, and what the clinician decided (call 911 or cancel the alarm)."""
    e = cases[cases.severity == "emergency"]
    if not len(e):
        return {"cases": 0}
    mins = (e.acknowledged_h - e.created_h) * 60
    d = decisions[decisions.decision == "emergency"]
    true_ep = [_in_episode(ep_idx, r.patient_id, int(r.day), before=0, after=0) for r in d.itertuples()]
    return {"cases": int(len(e)),
            "median_minutes_to_video_call": round(float(mins.median()), 1) if mins.notna().any() else None,
            "video_call_within_15_min": round(float((mins <= 15).mean()), 3),
            "video_call_within_30_min": round(float((mins <= 30).mean()), 3),
            "reminders_sent": int(e.reminders.sum()),
            "decisions": d.emergency_decision.fillna("pending").value_counts().to_dict()
            if "emergency_decision" in d else {},
            "during_true_episode": round(float(np.mean(true_ep)), 3) if len(true_ep) else None,
            "chest_pain_or_fainting_kept_critical": int(((decisions.decision.isin(["escalate", "escalate_update"]))
                                                         & decisions.alert_rules.fillna("").str.contains(
                                                             "chest pain|fainting")).sum())}


def ecg_recordings(decisions):
    """Every ECG recording is checked, with or without a platform alert."""
    if "ecg_checked" not in decisions:
        return {}
    e = decisions[decisions.ecg_checked.fillna(False).astype(bool)]
    no_alert = e[e.decision.isin(["ecg_ok", "ecg_repeat", "ecg_noted", "ecg_new_af", "ecg_known_af"])]
    return {"recordings_checked": int(len(e)), "on_days_without_alert": int(len(no_alert)),
            "repeat_requested_without_alert": int((no_alert.decision == "ecg_repeat").sum()),
            "abnormal_noted_without_alert": int((no_alert.decision == "ecg_noted").sum()),
            "new_af_reported_without_alert": int((no_alert.decision == "ecg_new_af").sum()),
            "known_af_logged_without_alert": int((no_alert.decision == "ecg_known_af").sum())}


def _spells(feed, col):
    """First and last day of each patient's spell in a true/false ground-truth column."""
    f = feed[feed[col].fillna(False).astype(bool)]
    return f.groupby("patient_id").day.agg(start="min", end="max").reset_index()


def rhythm(feed, patients, decisions):
    """Named ECG findings: new atrial fibrillation reported to the care team, known atrial
    fibrillation logged, and heart rates on the ECG outside the patient's limits."""
    if "ecg_af" not in decisions or "ecg_checked" not in decisions:
        return {}
    e = decisions[decisions.ecg_checked.fillna(False).astype(bool)]
    ok = e[e.ecg_reliable.astype(bool)]
    new = e[e.ecg_af == "new"]
    flags = e["flags"].fillna("") if "flags" in e else pd.Series("", index=e.index)
    out = {"reliable_recordings": int(len(ok)),
           "recordings_named_af": int((ok.ecg_rhythm == "af").sum()),
           "known_af_recordings_not_escalated_as_new": int((e.ecg_af == "known").sum()),
           "new_af_reports": int(len(new)),
           "new_af_report_severity": new.severity.value_counts().to_dict(),
           "raised_to_critical_when_symptoms_followed": int(((e.ecg_af == "reported")
                                                             & flags.str.contains("ecg_new_af")).sum()),
           "later_af_recordings_already_reported": int((e.ecg_af == "reported").sum()),
           "bradycardia_on_ecg": int((e.ecg_rate == "bradycardia").sum()),
           "tachycardia_on_ecg": int((e.ecg_rate == "tachycardia").sum())}
    if "emergency_decision" in e:  # the unchanged emergency rule in patients whose ECG always shows AF
        em = e[(e.decision == "emergency") & (e.ecg_af == "known")]
        out["emergencies_where_the_ecg_finding_was_known_af"] = {
            "cases": int(len(em)), "of_all_emergencies": int((e.decision == "emergency").sum()),
            "clinician_cancelled": int((em.emergency_decision == "cancel").sum())}
    known_sym = e[(e.ecg_af == "known") & e.decision.isin(["escalate", "escalate_update"])
                  & (e.severity == "caution")]
    out["caution_alerts_with_known_af_on_ecg_not_raised"] = int(len(known_sym))
    if "gt_ecg_rhythm" in e:
        true_af = ok.gt_ecg_rhythm == "af"
        out["af_named_on_reliable_af_recordings"] = round(float((ok.ecg_rhythm[true_af] == "af").mean()), 3) \
            if true_af.any() else None
        out["new_af_reports_on_a_recording_without_af"] = int((new.gt_ecg_rhythm != "af").sum())
    rated = e[e.ecg_rate.notna() & e.decision.isin(["escalate", "escalate_update", "emergency"])]
    out["alerts_with_the_rate_confirmed_on_the_ecg"] = {"alerts": int(len(rated)),
                                                        "critical_or_emergency": int(rated.severity.isin(
                                                            ["critical", "emergency"]).sum())}
    if "gt_rate" in feed:
        kit = set(patients[patients.ecg_kit.astype(bool)].patient_id)
        f = feed[feed.gt_rate.isin(["slow", "fast"]) & feed.patient_id.isin(kit)]
        sp = f.groupby("patient_id").agg(start=("day", "min"), end=("day", "max"), kind=("gt_rate", "first"))
        seen = e[e.ecg_rate.notna()]
        hit = [bool(((seen.patient_id == pid) & (seen.day >= s.start) & (seen.day <= s.end)).any())
               for pid, s in sp.iterrows()]
        out["heart_rate_spells_in_patients_with_ecg_kit"] = sp.kind.value_counts().to_dict()
        out["heart_rate_spells_confirmed_on_the_ecg"] = round(float(np.mean(hit)), 3) if hit else None
    if "gt_new_af" in feed:
        sp = _spells(feed, "gt_new_af")
        sp = sp[sp.patient_id.isin(patients[patients.ecg_kit.astype(bool)].patient_id)]
        days = []
        for s in sp.itertuples():
            d = new[(new.patient_id == s.patient_id) & (new.day >= s.start) & (new.day <= s.end)].day
            days.append(d.min() - s.start if len(d) else np.nan)
        days = pd.Series(days, dtype=float)
        out.update({"new_af_spells_in_patients_with_ecg_kit": int(len(sp)),
                    "spells_reported_to_care_team": round(float(days.notna().mean()), 3) if len(days) else None,
                    "median_days_onset_to_report": round(float(days.median()), 1) if days.notna().any() else None})
    return out


def selfcare(feed, decisions, cases, patient_months, ep_idx, cfg=None):
    """Sleep log and breathing exercises: check-in calls opened for worse sleep, what was behind
    them in the simulated world, and breathing sessions against the care team's plan."""
    if "sleep_hours" not in feed or not feed.sleep_hours.notna().any():
        return {}
    d = decisions[decisions.decision.isin(["sleep_check", "sleep_noted"])]
    sc = cases[cases.kind == "sleep_check"]
    out = {"nights_logged": int(feed.sleep_hours.notna().sum()),
           "median_sleep_h": round(float(feed.sleep_hours.median()), 1),
           "sleep_checkins": int((d.decision == "sleep_check").sum()),
           "notes_added_to_open_case": int((d.decision == "sleep_noted").sum()),
           "patients_with_a_checkin": int(d.patient_id.nunique()),
           "checkins_per_100_patient_months": round(100 * len(sc) / patient_months, 1),
           "median_hours_to_checkin_call": round(float((sc.acknowledged_h - sc.created_h).median()), 1)
           if sc.acknowledged_h.notna().any() else None}
    if "gt_poor_sleep" in feed:
        poor = {(r.patient_id, r.day) for r in feed[feed.gt_poor_sleep.fillna(False).astype(bool)].itertuples()}

        def why(r):
            nights = [(r.patient_id, x) for x in range(int(r.day) - 2, int(r.day) + 1)]
            return "poor-sleep spell" if any(n in poor for n in nights) else \
                "decompensation or new atrial fibrillation" if any(ep_idx.get(n, False) for n in nights) else \
                "a few short nights in a row"
        out["what_was_behind_them"] = pd.Series([why(r) for r in d.itertuples()], dtype=object) \
            .value_counts().to_dict()
        sp = _spells(feed, "gt_poor_sleep")
        hit = [bool(((d.patient_id == s.patient_id) & (d.day >= s.start) & (d.day <= s.end + 2)).any())
               for s in sp.itertuples()]
        out["poor_sleep_spells"] = int(len(sp))
        out["spells_with_checkin_or_note"] = round(float(np.mean(hit)), 3) if hit else None
    if "breathing_sessions" in feed and feed.breathing_sessions.notna().any():
        plan = ((cfg or {}).get("selfcare") or {}).get("breathing", {}).get("sessions_per_day")
        b = feed[feed.breathing_sessions.notna()]
        per_patient = b.groupby("patient_id").breathing_sessions.mean()
        out["breathing"] = {"plan_sessions_per_day": plan,
                            "mean_sessions_per_day": round(float(b.breathing_sessions.mean()), 2)}
        if plan:
            out["breathing"]["days_meeting_plan"] = round(float((b.breathing_sessions >= plan).mean()), 3)
            out["breathing"]["patients_below_half_plan"] = round(float((per_patient < plan / 2).mean()), 3)
    return out


def patient_loop(loops, patients, horizon_h):
    """After the clinician's decision: did the patient confirm, act, and did readings improve?
    Platform alone = one message and no tracking: completed only if the patient reads it and acts
    on their own (gt_self_completed)."""
    if loops is None or not len(loops):
        return {}
    lp = loops.merge(patients[["patient_id", "remoteness"]], on="patient_id")
    lp = lp[lp.notified_h <= horizon_h - 7 * 24]  # loops with at least a week to play out
    if not len(lp):
        return {}
    acted = lp.action_done_h.notna()
    hours = lp.action_done_h - lp.notified_h

    def summ(d):
        a = d.action_done_h.notna()
        return {"n": int(len(d)), "heartsafe_action_done": round(float(a.mean()), 3),
                "platform_alone_action_done": round(float(d.gt_self_completed.fillna(False).astype(bool).mean()), 3),
                "median_hours_to_action": round(float((d.action_done_h - d.notified_h).median()), 1)
                if a.any() else None}
    return {
        **summ(lp),
        "confirmed_by": lp.confirmed_by.fillna("not reached").value_counts().to_dict(),
        "helper_contacts": int(lp.contacts.sum()),
        "closed_readings_improved": int((lp.state == "closed").sum()),
        "reopened": int((lp.state == "reopened").sum()),
        "reopen_reasons": lp[lp.state == "reopened"].outcome.str.split(" \\(").str[0].value_counts().to_dict(),
        "still_open_at_end": int((~lp.state.isin(["closed", "reopened"])).sum()),
        "acted_within_48h": round(float((hours <= 48).mean()), 3) if acted.any() else None,
        "by_remoteness": {k: summ(g) for k, g in lp.groupby("remoteness")},
    }


def evaluate(feed, patients, decisions, cases, baseline, horizon_h, rules=None, followup_cfg=None, loops=None,
             cfg=None):
    patient_months = len(patients) * (feed.day.max() + 1) / 30.0
    # a true event = a decompensation episode, a first spell of atrial fibrillation, or a heart rate
    # truly outside the patient's limits
    event = feed.gt_episode.astype(bool) | (feed.gt_new_af.fillna(False).astype(bool) if "gt_new_af" in feed
                                            else False)
    if "gt_rate" in feed:
        event = event | feed.gt_rate.isin(["slow", "fast"])
    ep_idx = dict(zip(zip(feed.patient_id, feed.day), event))
    eps = episodes(feed)
    eps = eps[eps.first_alert_h.notna()]
    cases = cases.copy()
    clinical = cases[cases.kind == "clinical"]
    rem = patients.set_index("patient_id").remoteness

    # Episode-level: did a clinician acknowledge, and how fast after the platform's first alert?
    rows = []
    esc = decisions[decisions.decision.isin(["escalate", "escalate_update", "emergency", "ecg_new_af"])]
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
    alert_dec = decisions[~decisions.decision.isin(["scale_verified", "imaging_accept", "imaging_review",
                                                    "sleep_check", "sleep_noted"])]
    alert_dec = alert_dec.drop_duplicates(["patient_id", "day"], keep="last")
    fa = fa.merge(alert_dec[["patient_id", "day", "decision"]], on=["patient_id", "day"], how="left")
    stopped = fa.decision.isin(["repeat", "filtered"]).mean() if len(fa) else np.nan
    held = fa.decision.isin(["nurse_verify", "nurse_verify_update"]).mean() if len(fa) else np.nan
    scale_checks = cases[cases.kind == "scale_check"]
    followups = cases[cases.kind == "patient_followup"]
    sleep_checks = cases[cases.kind == "sleep_check"]

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
            "heartsafe_patient_followups": round(100 * len(followups) / patient_months, 1),
            "heartsafe_sleep_checkins": round(100 * len(sleep_checks) / patient_months, 1),
            "platform_alone_alert_items": round(100 * len(baseline) / patient_months, 1),
            "false_alarms_heartsafe": round(100 * false_agent / patient_months, 1),
            "false_alarms_platform_alone": round(100 * false_platform / patient_months, 1)},
        "agent_actions": decisions.decision.value_counts().to_dict(),
        "resolution": resolution(clinical, baseline, horizon_h, followup_cfg),
        "stalled_cases": stalled(clinical, baseline),
        "model_evidence": model_evidence(decisions),
        "emergencies": emergencies(decisions, cases, ep_idx),
        "ecg_recordings": ecg_recordings(decisions),
        "rhythm": rhythm(feed, patients, decisions),
        "selfcare": selfcare(feed, decisions, cases, patient_months, ep_idx, cfg),
        "patient_loop": patient_loop(loops, patients, horizon_h),
        "followup": {
            "cases": int(len(cases)), "reminders_sent": int(cases.reminders.sum()),
            "clinical_cases_ack_within_deadline": round(float(
                (clinical.acknowledged_h <= (clinical.standard_ack_deadline_h.fillna(clinical.ack_deadline_h)
                                             if "standard_ack_deadline_h" in clinical else clinical.ack_deadline_h)
                 ).mean()), 3) if len(clinical) else None,
            "platform_alone_alert_items_never_acknowledged": int(baseline.lost.sum()),
            "open_at_end": int((cases.state != "resolved").sum())},
    }
    return res, er
