import json

import numpy as np
import pandas as pd
import pytest

from heartsafe import load_config
from heartsafe.agent import HeartSafeAgent, checkin_from_sleep_log, rule_key, run_baseline
from heartsafe.equity import audit
from heartsafe.evaluate import evaluate
from heartsafe.monitoring import PlatformAdapter, patient_limits, platform_levels, rule_levels
from heartsafe.reliability import assess_ecg, assess_reading
from heartsafe.review import render, summarize_ecg, summarize_mri
from heartsafe.selfcare import SelfCareLog, breathing_status, sleep_status, with_logged
from heartsafe.selfcare import describe as describe_selfcare
from heartsafe.signals import ecg_quality_index
from heartsafe.simulate import simulate_cohort
from heartsafe.tracker import PermissionDenied, Tracker
from heartsafe.watch import Watcher, followup_status

CFG = load_config()


def day(d, w, sbp=120, hr=70, sob=False, swelling=False):
    return {"day": d, "weight_kg": w, "sbp": sbp, "hr": hr, "sob": sob, "swelling": swelling}


def test_alert_rules_weight_gain_and_critical():
    w = np.array([70, 70.1, 70, 71.8, 72.5, 74.2], float)
    lv, fired = rule_levels(w, np.full(6, 120.0), np.full(6, 70.0), np.zeros(6, bool), np.zeros(6, bool), CFG["alert_rules"])
    assert lv[3] == 1 and "2 days" in fired[3][0]
    assert lv[5] == 2                          # > 3.5 kg in 7 days
    lv, _ = rule_levels(np.full(3, 70.0), np.array([120, 85, 120.0]), np.full(3, 70.0), np.zeros(3, bool),
                        np.zeros(3, bool), CFG["alert_rules"])
    assert list(lv) == [0, 2, 0]               # low BP is critical


def _levels(n=1, limits=None, creat0=None, **extra):
    z = np.zeros(n, bool)
    return rule_levels(np.full(n, 70.0), np.full(n, 120.0), np.full(n, 70.0), z, z, CFG["alert_rules"],
                       {k: np.asarray(v) for k, v in extra.items()}, limits, creat0)


def test_rules_added_after_cardiologist_review():
    assert _levels(dbp=[115])[0][0] == 2                       # diastolic > 110
    assert list(_levels(3, spo2=[96, 91, 88])[0]) == [0, 1, 2]  # oxygen saturation
    assert _levels(chest_pain=[True])[0][0] == 2 and _levels(syncope=[True])[0][0] == 2
    assert _levels(orthopnea=[True])[0][0] == 1 and _levels(pnd=[True])[0][0] == 1
    assert list(_levels(3, diuretic_taken=[False, False, True])[0]) == [0, 1, 0]
    assert list(_levels(3, potassium=[4.2, 5.8, 6.3])[0]) == [0, 1, 2]
    assert _levels(creatinine_umol=[140], creat0=105)[0][0] == 1
    # the clinician can set a patient's own limits at enrollment (e.g. low resting heart rate)
    lim = patient_limits(CFG["alert_rules"], {"hr_low": 40})
    lv, _ = rule_levels(np.full(1, 70.0), np.full(1, 120.0), np.full(1, 42.0), np.zeros(1, bool), np.zeros(1, bool),
                        CFG["alert_rules"], limits=lim)
    assert lv[0] == 0


def test_enrollment_follows_inclusion_criteria():
    patients, _ = simulate_cohort(CFG, n_patients=80, n_days=30, seed=5, demo=False)
    inc = CFG["inclusion"]
    assert len(patients) == 80 and (patients.hf_admission_h >= inc["min_hf_admission_h"]).all()
    assert (patients.days_since_discharge <= inc["max_days_since_discharge"]).all()
    assert patients.nyha.isin(inc["nyha"]).all() and not patients[inc["exclude"]].any().any()
    assert (patients.can_self_measure | patients.caregiver).all()
    s = patients.attrs["screening"]
    assert s["candidates"] > s["enrolled"] == 80 and s["reasons"]
    # a patient whose usual BP is near a default limit gets a personal limit at enrollment
    low = patients[patients.baseline_sbp < CFG["alert_rules"]["sbp_low"] + 15]
    assert len(low) and (low.sbp_low < CFG["alert_rules"]["sbp_low"]).all()


def test_loop_reopens_only_if_the_starting_problem_remains():
    loop = {"level": 1, "rules": {rule_key("weight +2.6 kg in 7 days"), rule_key("swelling")}}
    nope = HeartSafeAgent._not_improved
    assert nope(loop, 1, ["weight +2.8 kg in 7 days"])       # the same rule still fires (other number)
    assert not nope(loop, 0, [])                               # readings normal
    assert not nope(loop, 1, ["diuretic not taken for 2 days"])  # a different caution: a new alert, not a reopening
    assert nope(loop, 2, ["systolic BP 84"])                   # worse: a higher alert level
    assert HeartSafeAgent._worse(loop, 2, ["systolic BP 84"])
    crit = {"level": 2, "rules": {rule_key("systolic BP 85")}}
    assert not nope(crit, 1, ["swelling"]) and nope(crit, 2, ["chest pain"])  # new critical reason = worse
    assert nope({"level": None, "rules": None}, 1, ["swelling"])  # no record of the trigger: any alert counts


def test_reliability_flags_unit_cuff_and_scale_change():
    hist = [day(d, 70 + 0.1 * (d % 2)) for d in range(7)]
    assert "unit_error" in assess_reading(hist, day(7, 154.3), CFG)[0]
    assert "bp_artifact" in assess_reading(hist, day(7, 70, sbp=165), CFG)[0]
    assert "step_change" in assess_reading(hist, day(7, 72.8), CFG)[0]
    # the same jump with symptoms is not called a device change
    assert "step_change" not in assess_reading(hist, day(7, 72.8, sob=True), CFG)[0]


def test_ecg_reliability_and_quality_index():
    good = assess_ecg({"probability": 0.9, "uncertainty": 0.1, "quality": {"sqi": 0.9}}, CFG)
    bad = assess_ecg({"probability": 0.9, "uncertainty": 0.6, "quality": {"sqi": 0.3}}, CFG)
    assert good["reliable"] and good["abnormal"] and not bad["reliable"]
    fs = 250
    t = np.arange(10 * fs) / fs
    beats = np.arange(0.4, 10, 60 / 72)                # 72 bpm
    ecg = sum(np.exp(-((t - b) / 0.02) ** 2) for b in beats) + 0.02 * np.random.default_rng(0).normal(size=t.size)
    good_q, issues = ecg_quality_index(ecg, fs)
    assert good_q > 0.6, issues
    assert ecg_quality_index(np.zeros_like(ecg), fs)[0] < 0.2


def test_tracker_only_clinicians_close_and_audit():
    tr = Tracker()
    cid = tr.create("P1", "clinical", "critical", 10, 4, 48, {"x": 1})
    with pytest.raises(PermissionDenied):
        tr.act(cid, "acknowledge", 11, "heartsafe", "agent")
    with pytest.raises(ValueError):
        tr.act(cid, "resolve", 11, "n", "clinician", "done")      # must acknowledge first
    tr.act(cid, "acknowledge", 11, "n", "clinician")
    tr.act(cid, "review", 12, "n", "clinician")
    with pytest.raises(ValueError):
        tr.act(cid, "resolve", 13, "n", "clinician", "  ")        # note required
    tr.act(cid, "resolve", 13, "n", "clinician", "called patient")
    assert tr.get(cid)["state"] == "resolved"
    assert [a["action"] for a in tr.audit(cid)] == ["create", "acknowledge", "review", "resolve"]
    assert tr.due_reminders(100) == []


def test_followup_watch_finds_overdue_cases_and_unconfirmed_loops():
    tr = Tracker()
    late = tr.create("P1", "clinical", "critical", 0, 4, 48, {"x": 1})       # never acknowledged
    done = tr.create("P2", "clinical", "caution", 0, 24, 120, {"x": 1})
    for a in ("acknowledge", "review"):
        tr.act(done, a, 1, "n", "clinician")
    tr.act(done, "resolve", 2, "n", "clinician", "extra diuretic")
    tr.start_loop(done, 2, "plan", 12)                                       # confirmation due at 14 h
    w = Watcher(tr, CFG)
    assert w.tick(3) == [] and w.tick(4) is None                             # not due again until 2 h later
    found = {s["case_id"]: s for s in w.tick(20, act=True)}
    assert found[late]["issues"] == ["ack_overdue"] and found[done]["issues"] == ["patient_not_confirmed"]
    acts = [a["action"] for a in tr.audit()]
    assert "reminder" in acts and "patient_contact" in acts                  # reminds and contacts a helper ...
    assert tr.get(late)["state"] == "created" and tr.loop(done)["state"] == "notified"  # ... never acts for them
    assert any("caregiver" in line for line in followup_status(tr.get(done), tr, 21)["lines"])


def test_ecg_and_mri_summaries_without_llm():
    ecg = {"reliable": False, "action": "repeat", "abnormal": True, "prob": 0.91, "quality": 0.3,
           "uncertainty": 0.2, "reasons": ["noisy signal"]}
    s = summarize_ecg(ecg)
    assert "poor" in s and "Not used as evidence (noisy signal)" in s and "repeat ECG" in s
    mri = {"reliable": True, "lvef": 38.0, "scanner": "GE", "uncertainty": 0.04, "flags": [], "caveats": []}
    assert "38% (reduced)" in summarize_mri(mri) and "GE scanner" in summarize_mri(mri)
    assert "(preserved)" in summarize_mri({**mri, "lvef": 60.0})
    review = {"kind": "clinical", "patient_id": "P1", "day": 3, "severity": "caution",
              "context": {"age": 70, "sex": "F", "remoteness": "remote", "device": "own_scale"},
              "why_flagged": ["swelling"], "symptoms": [], "trend": [], "ecg": ecg, "mri": mri,
              "caveats": ["ECG not used as evidence (noisy signal); repeat ECG requested"], "next_step": "Call."}
    text = render(review, ["Waiting for acknowledgement; deadline in 3.0 h."])
    assert "Follow-up status: Waiting" in text and "Reliability caveats" not in text  # no duplicate ECG caveat


def test_patient_loop_only_patient_side_confirms():
    tr = Tracker()
    cid = tr.create("P1", "clinical", "caution", 0, 24, 120, {"x": 1})
    with pytest.raises(ValueError):
        tr.start_loop(cid, 1, "plan", 12)                       # not resolved yet
    for a in ("acknowledge", "review"):
        tr.act(cid, a, 1, "n", "clinician")
    tr.act(cid, "resolve", 2, "n", "clinician", "extra diuretic for 3 days")
    tr.start_loop(cid, 2, "plan", 12)
    with pytest.raises(PermissionDenied):
        tr.patient_step(cid, "confirm", 3, "heartsafe", "agent")  # the agent cannot confirm for the patient
    with pytest.raises(ValueError):
        tr.close_loop(cid, 3, "improved")                         # nothing done yet
    tr.contact(cid, 14, "caregiver")
    tr.patient_step(cid, "confirm", 15, "daughter", "caregiver")
    tr.patient_step(cid, "action_done", 20, "daughter", "caregiver")
    tr.close_loop(cid, 92, "readings improved after the action")
    lp = tr.loop(cid)
    assert lp["state"] == "closed" and lp["confirmed_by"] == "caregiver" and lp["contacts"] == 1
    assert [a["action"] for a in tr.audit(cid)][-4:] == ["patient_contact", "patient_confirm", "patient_action_done",
                                                          "loop_closed"]


def test_emergency_only_a_clinician_decides():
    tr = Tracker()
    cid = tr.create("P1", "clinical", "emergency", 10, 0.25, 1, {"x": 1})
    tr.notify_patient(cid, 10, "If you feel worse, call 911.")
    with pytest.raises(ValueError):
        tr.emergency_decision(cid, 10.1, "n", "clinician", "cancel", "settled")   # video call not started
    tr.act(cid, "acknowledge", 10.1, "n", "clinician", note="video call started")
    with pytest.raises(PermissionDenied):
        tr.emergency_decision(cid, 10.2, "heartsafe", "agent", "call_911", "x")  # the agent never calls 911
    with pytest.raises(ValueError):
        tr.emergency_decision(cid, 10.2, "n", "clinician", "call_911", " ")      # reason required
    with pytest.raises(ValueError):
        tr.emergency_decision(cid, 10.2, "n", "clinician", "ambulance", "x")     # only call_911 / cancel
    tr.emergency_decision(cid, 10.3, "n", "clinician", "call_911", "pain ongoing on video")
    c = tr.get(cid)
    assert c["state"] == "resolved" and c["outcome"].startswith("Video call: clinician called 911")
    assert [a["action"] for a in tr.audit(cid)] == ["create", "patient_message", "acknowledge", "emergency_decision"]


def _one_patient(symptom_day=9, symptom="chest_pain", n_days=12):
    patients, feed = simulate_cohort(CFG, n_patients=1, n_days=n_days, seed=11, demo=False)
    patients["ecg_kit"] = True
    feed = feed.assign(submitted=True, weight_kg=70.0, sbp=120, dbp=75, hr=70, spo2=97.0, diuretic_taken=True,
                       potassium=np.nan, creatinine_umol=np.nan, gt_episode=False, gt_fault="none")
    for c in ("sob", "swelling", "orthopnea", "pnd", "chest_pain", "syncope"):
        feed[c] = False
    feed.loc[feed.day == symptom_day, symptom] = True
    return patients, platform_levels(feed, CFG["alert_rules"], patients)


def _run_with_ecg(patients, feed, reliable, abnormal):
    tr = Tracker()
    agent = HeartSafeAgent(CFG, patients, tr, np.random.default_rng(0), 30 * 24.0)
    ecg = {"reliable": reliable, "action": "accept" if reliable else "repeat", "abnormal": abnormal,
           "prob": 0.9 if abnormal else 0.1, "quality": 0.9 if reliable else 0.2, "uncertainty": 0.1,
           "reasons": [] if reliable else ["low signal quality"]}
    agent._read_ecg = lambda p, row: (dict(ecg), {"ecg_checked": True, "ecg_reliable": reliable,
                                                  "ecg_abnormal": abnormal})
    return agent.run(PlatformAdapter(feed).observable()), tr


@pytest.mark.parametrize("symptom", ["chest_pain", "syncope"])
def test_emergency_needs_symptom_and_reliable_abnormal_ecg(symptom):
    patients, feed = _one_patient(symptom=symptom)
    dec, tr = _run_with_ecg(patients, feed, reliable=True, abnormal=True)
    em = dec[dec.decision == "emergency"]
    assert em.day.tolist() == [9]
    case = tr.get(int(em.case_id.iloc[0]))
    assert case["severity"] == "emergency" and case["ack_deadline_h"] - case["created_h"] == pytest.approx(0.25)
    log = tr.audit(case["case_id"])
    assert log[1]["action"] == "patient_message" and "call 911" in log[1]["detail"]
    assert log[1]["t_h"] == case["created_h"]                       # the patient is told straight away
    assert case["state"] == "resolved" and case["outcome"].startswith("Video call:")
    assert all(a["role"] in ("agent", "clinician") and (a["action"] != "emergency_decision" or
                                                       a["role"] == "clinician") for a in log)
    # the same symptom without an abnormal ECG, or with an ECG that is not reliable, stays critical
    for reliable, abnormal in ((True, False), (False, True)):
        dec, tr = _run_with_ecg(patients, feed, reliable, abnormal)
        assert "emergency" not in dec.decision.values
        assert dec[dec.day == 9].severity.tolist() == ["critical"]


def test_every_ecg_recording_is_checked():
    patients, feed = _one_patient(symptom_day=99)                   # no symptoms, no alerts
    dec, _ = _run_with_ecg(patients, feed, reliable=True, abnormal=False)
    assert dec[dec.decision == "ecg_ok"].day.tolist() == [0, 7]     # weekly routine recordings
    dec, _ = _run_with_ecg(patients, feed, reliable=False, abnormal=False)
    assert dec[dec.decision == "ecg_repeat"].day.tolist() == [0, 7]  # a disconnected ECG -> record again
    assert CFG["patient_loop"]["outcome_check_after_h"] == 24


def _run_with_rhythm(patients, feed, af_from_day=None, ecg_hr=70.0, cfg=CFG):
    """Every ECG is reliable; from af_from_day on it shows atrial fibrillation. ecg_hr: the R-peak rate."""
    tr = Tracker()
    agent = HeartSafeAgent(cfg, patients, tr, np.random.default_rng(0), 30 * 24.0)

    def read(p, row):
        af = af_from_day is not None and row.day >= af_from_day
        other = not af and ecg_hr != 70.0                      # a rate outside the limits is another abnormal rhythm
        ecg = {"reliable": True, "action": "accept", "abnormal": af or other, "prob": 0.9 if af or other else 0.1,
               "quality": 0.9, "uncertainty": 0.1, "reasons": [],
               "rhythm": "af" if af else "other" if other else "normal", "af_prob": 0.9 if af else 0.02,
               "hr": ecg_hr}
        return ecg, {"ecg_checked": True, "ecg_reliable": True, "ecg_abnormal": ecg["abnormal"]}
    agent._read_ecg = read
    return agent.run(PlatformAdapter(feed).observable()), tr


def test_new_af_is_critical_with_symptoms_and_caution_without():
    # no symptoms on the routine recording of day 7: reported once, at caution level
    patients, feed = _one_patient(symptom_day=9, symptom="sob", n_days=16)
    patients["atrial_fibrillation"] = False
    dec, tr = _run_with_rhythm(patients, feed, af_from_day=7)
    first = dec[dec.day == 7].iloc[0]
    assert first.decision == "ecg_new_af" and first.severity == "caution" and "ecg_new_af" in first["flags"]
    case = tr.get(int(first.case_id))
    assert case["kind"] == "clinical" and "new atrial fibrillation" in str(case["review"])
    # a symptom two days later raises the same finding to critical; after that it is not reported again
    later = dec[(dec.day == 9) & dec["flags"].fillna("").str.contains("ecg_new_af")]
    assert later.severity.tolist() == ["critical"]
    again = dec[(dec.day == 14) & (dec.decision == "ecg_known_af")]
    assert len(again) == 1 and "already reported" in again.notes.iloc[0]
    assert not dec[dec.day == 14]["flags"].fillna("").str.contains("ecg_new_af").any()
    # a symptom on the day of the first AF recording: critical straight away
    patients, feed = _one_patient(symptom_day=7, symptom="sob", n_days=10)
    patients["atrial_fibrillation"] = False
    dec, _ = _run_with_rhythm(patients, feed, af_from_day=7)
    new = dec[dec["flags"].fillna("").str.contains("ecg_new_af")]
    assert new.day.tolist() == [7] and new.severity.tolist() == ["critical"]


def test_known_af_is_logged_not_escalated_as_new():
    patients, feed = _one_patient(symptom_day=99)
    patients["atrial_fibrillation"] = True                          # in the enrollment record
    dec, tr = _run_with_rhythm(patients, feed, af_from_day=0)
    assert dec[dec.day.isin([0, 7])].decision.tolist() == ["ecg_known_af", "ecg_known_af"]
    assert "enrollment record" in dec.notes.iloc[0]
    assert not any(c["kind"] == "clinical" for c in tr.cases())


def test_heart_rate_on_the_ecg_uses_the_patients_own_limits():
    patients, feed = _one_patient(symptom_day=9, symptom="orthopnea")   # a caution alert with a symptom
    lim = patient_limits(CFG["alert_rules"], patients.iloc[0].to_dict())
    dec, tr = _run_with_rhythm(patients, feed, ecg_hr=lim["hr_high"] + 20)
    noted = dec[dec.day == 7].iloc[0]                                # routine recording, no alert: on the chart
    assert noted.decision == "ecg_noted" and noted["flags"] == "ecg_tachycardia"
    alert = dec[dec.day == 9].iloc[0]                                # with a symptom the ECG rate makes it critical
    assert alert.decision == "escalate" and alert.severity == "critical"
    assert "tachycardia on the ECG" in str(tr.get(int(alert.case_id))["review"])
    dec, _ = _run_with_rhythm(patients, feed, ecg_hr=lim["hr_low"] - 8)
    assert dec[dec.day == 7]["flags"].tolist() == ["ecg_bradycardia"]
    # a personal limit set at enrollment moves the line: the same rate is then inside the limits
    patients["hr_low"] = lim["hr_low"] - 15
    dec, _ = _run_with_rhythm(patients, feed, ecg_hr=lim["hr_low"] - 8)
    assert not dec["flags"].fillna("").str.contains("bradycardia").any()


def test_known_af_choices_are_config_switches():
    import copy
    # chest pain + a reliable abnormal ECG that shows only the known AF: an emergency unless switched off
    patients, feed = _one_patient(symptom_day=9, symptom="chest_pain")
    patients["atrial_fibrillation"] = True
    dec, _ = _run_with_rhythm(patients, feed, af_from_day=0)
    assert dec[dec.day == 9].decision.tolist() == ["emergency"]
    off = copy.deepcopy(CFG)
    off["ecg_rhythm"]["known_af_triggers_emergency"] = False
    dec, _ = _run_with_rhythm(patients, feed, af_from_day=0, cfg=off)
    assert "emergency" not in dec.decision.values and dec[dec.day == 9].severity.tolist() == ["critical"]
    # a caution alert with a symptom: known AF does not raise it unless the care team chooses so
    patients, feed = _one_patient(symptom_day=9, symptom="orthopnea")
    patients["atrial_fibrillation"] = True
    dec, _ = _run_with_rhythm(patients, feed, af_from_day=0)
    row = dec[dec.day == 9].iloc[0]
    assert row.severity == "caution" and "ecg_known_af" in row["flags"]
    on = copy.deepcopy(CFG)
    on["ecg_rhythm"]["known_af_upgrades_with_symptoms"] = True
    dec, _ = _run_with_rhythm(patients, feed, af_from_day=0, cfg=on)
    assert dec[dec.day == 9].severity.tolist() == ["critical"]


def test_ecg_rhythm_is_named_only_when_likely_enough():
    out = {"probability": 0.9, "uncertainty": 0.1, "quality": {"sqi": 0.9, "heart_rate_bpm": 112},
           "prediction": {"rhythm": {"label": "af", "probs": {"af": 0.42, "normal": 0.3, "other": 0.28}}}}
    e = assess_ecg(out, CFG)
    assert e["rhythm"] == "unclear" and e["hr"] == 112.0 and e["reliable"]
    out["prediction"]["rhythm"]["probs"]["af"] = 0.8
    assert assess_ecg(out, CFG)["rhythm"] == "af"
    assert assess_ecg({"probability": 0.9, "uncertainty": 0.1, "quality": {"sqi": 0.9}}, CFG)["rhythm"] is None


def _nights(hours, first_day=0):
    return [{"day": first_day + i, "sleep_hours": h, "sleep_awakenings": 1.0} for i, h in enumerate(hours)]


def test_sleep_is_compared_with_the_patients_own_usual():
    s = sleep_status(_nights([7.0] * 6 + [5.0, 5.0]), {"day": 8, "sleep_hours": 5.0, "sleep_awakenings": 3.0}, CFG)
    assert s["worse"] and s["usual_h"] == 7.0 and s["recent_h"] == 5.0
    assert "usual 7.0 h" in s["text"] and "woke 1.7 times a night (usual 1)" in s["text"]
    assert not sleep_status(_nights([7.0] * 6 + [6.0] * 3), None, CFG)["worse"]       # 1 h less: not enough
    assert not sleep_status(_nights([7.0] * 6 + [4.0, 4.0, 6.5]), None, CFG)["worse"]  # one good night
    assert sleep_status(_nights([5.0] * 6 + [4.0] * 3), None, CFG)["worse"]           # under 4.5 h a night
    short = sleep_status(_nights([7.0] * 3 + [4.0] * 3), None, CFG)                   # usual not known yet
    assert short["usual_h"] is None and not short["worse"]
    assert sleep_status([{"day": 0, "weight_kg": 70.0}], None, CFG) is None           # nothing logged
    b = breathing_status([{"day": d, "breathing_sessions": n} for d, n in enumerate([3, 2, 1, 2])], None, CFG)
    assert b == {"plan": CFG["selfcare"]["breathing"]["sessions_per_day"], "days": 4, "per_day": 2.0}
    text = describe_selfcare({"sleep": s, "breathing": b})
    assert text.startswith("Sleep worse than usual") and "2 of 3 planned sessions a day" in text


def test_worse_sleep_opens_one_low_priority_checkin():
    patients, feed = _one_patient(symptom_day=99, n_days=20)
    feed["sleep_hours"] = np.where(feed.day.between(10, 16), 4.5, 7.0)
    feed["sleep_awakenings"] = 1.0
    dec, tr = _run_with_ecg(patients, feed, reliable=True, abnormal=False)
    checks = dec[dec.decision == "sleep_check"]
    assert checks.day.tolist() == [12] and checks.severity.tolist() == ["low"]   # third short night; then not again
    case = tr.get(int(checks.case_id.iloc[0]))
    assert case["kind"] == "sleep_check" and case["severity"] == "low"
    assert "sleep worse than usual" in str(case["review"]) and not tr.loops()    # a call, not a clinical case
    assert not any(c["kind"] == "clinical" for c in tr.cases())


def test_selfcare_log_is_for_the_patient_side_only():
    log = SelfCareLog()
    for role in ("agent", "clinician"):
        with pytest.raises(PermissionDenied):
            log.log_sleep("P1", 5, 129.0, "x", role, 6.0)
    with pytest.raises(ValueError):
        log.log_sleep("P1", 5, 129.0, "P1", "patient", 30)
    with pytest.raises(ValueError):
        log.log_breathing("P1", 5, 129.0, "P1", "patient", "Slow breathing", 0)
    log.log_sleep("P1", 5, 129.0, "P1", "patient", 6.0, awakenings=2)
    log.log_sleep("P1", 5, 130.0, "daughter", "caregiver", 4.5, awakenings=3, woke_breathless=True,
                  kept_awake=["cough"])
    for _ in range(2):
        log.log_breathing("P1", 5, 131.0, "P1", "patient", "Slow breathing", 5)
    d = log.days("P1")[5]
    assert d["sleep_hours"] == 4.5 and d["woke_breathless"] and d["breathing_sessions"] == 2   # last sleep entry counts
    assert len(log.entries("P1")) == 4 and len(log.entries("P1", "breathing")) == 2 and log.days("P2") == {}
    merged = with_logged([{"day": 4, "sleep_hours": 7.0, "breathing_sessions": 1.0},
                          {"day": 5, "sleep_hours": 7.0, "breathing_sessions": 1.0}], log.days("P1"))
    assert [m["sleep_hours"] for m in merged] == [7.0, 4.5] and merged[1]["breathing_sessions"] == 3.0
    assert [m["day"] for m in with_logged([], {9: {"day": 9, "sleep_hours": 6.0}})] == [9]


def test_sleep_logged_in_the_app_reaches_the_care_team():
    patients, _ = _one_patient()
    p = patients.iloc[0].to_dict()
    usual = _nights([7.0] * 8)
    good = {"day": 8, "sleep_hours": 7.0, "sleep_awakenings": 1}
    tr = Tracker()
    assert checkin_from_sleep_log(tr, CFG, p, usual, good, 201.0)[:2] == (None, None) and not tr.cases()
    # three short nights: a low-priority check-in call
    bad = usual + _nights([4.5, 4.5], 8)
    what, cid, sc = checkin_from_sleep_log(tr, CFG, p, bad, {"day": 10, "sleep_hours": 4.5, "sleep_awakenings": 3},
                                           249.0)
    case = tr.get(cid)
    assert what == "checkin" and (case["kind"], case["severity"]) == ("sleep_check", "low") and sc["sleep"]["worse"]
    assert case["ack_deadline_h"] - case["created_h"] == CFG["followup"]["ack_deadline_h"]["low"]
    # while that case is open, the next entry is added to it
    what, same, _ = checkin_from_sleep_log(tr, CFG, p, bad, {"day": 10, "sleep_hours": 4.0, "sleep_awakenings": 3},
                                           250.0)
    assert (what, same) == ("noted", cid) and tr.audit(cid)[-1]["action"] == "evidence"
    # after the nurse's call: no second sleep-only check-in inside the cooldown ...
    for a in ("acknowledge", "review"):
        tr.act(cid, a, 260.0, "n", "clinician")
    tr.act(cid, "resolve", 261.0, "n", "clinician", "called; pain at night, plan documented")
    worse_again = {"day": 13, "sleep_hours": 4.0, "sleep_awakenings": 3}
    hist = bad + _nights([4.5, 4.5, 4.5], 10)
    assert checkin_from_sleep_log(tr, CFG, p, hist, worse_again, 321.0)[0] is None
    # ... but breathlessness at night is a symptom: a caution check-in whatever the sleep hours
    what, cid2, _ = checkin_from_sleep_log(tr, CFG, p, hist, {**good, "day": 13, "woke_breathless": True}, 322.0)
    case = tr.get(cid2)
    assert what == "checkin" and cid2 != cid and case["severity"] == "caution"
    review = json.loads(case["review"])
    assert review["symptoms"] == ["pnd"] and "reported in the sleep log" in review["why_flagged"][0]
    assert all(a["role"] != "agent" or a["action"] in ("create", "evidence") for a in tr.audit())


@pytest.fixture(scope="module")
def small_run():
    patients, feed = simulate_cohort(CFG, n_patients=150, n_days=60, seed=3)
    feed = platform_levels(feed, CFG["alert_rules"], patients)
    horizon = 60 * 24.0
    tr = Tracker()
    truth = {(r.patient_id, r.day): bool(r.gt_episode) for r in feed.itertuples()}
    scale_truth = {(r.patient_id, r.day) for r in feed[feed.gt_fault == "scale_change"].itertuples()}
    rhythm_truth = {(r.patient_id, r.day): "new_af" if r.gt_new_af else "af"
                    for r in feed[feed.gt_rhythm == "af"].itertuples()}
    for r in feed[feed.gt_rate.isin(["slow", "fast"])].itertuples():
        rhythm_truth.setdefault((r.patient_id, r.day), "rate")
    agent = HeartSafeAgent(CFG, patients, tr, np.random.default_rng(1), horizon, truth=truth,
                           scale_truth=scale_truth, rhythm_truth=rhythm_truth)
    obs = PlatformAdapter(feed).observable()
    assert not any(c.startswith("gt_") for c in obs.columns)
    dec = agent.run(obs)
    base = run_baseline(obs, patients, CFG, np.random.default_rng(2), horizon)
    run_ = (patients, feed, dec, pd.DataFrame(tr.cases()), base, horizon, tr)
    small_run_loops[:] = [pd.DataFrame(tr.loops()).merge(pd.DataFrame(agent.gt_loops), on="case_id")]
    return run_


small_run_loops = []


def test_demo_patient_story(small_run):
    _, _, dec, _, _, _, _ = small_run
    m = dec[dec.patient_id == "DEMO-MARY"]
    days = lambda lo, hi: m[(m.day >= lo) & (m.day <= hi)].decision  # noqa: E731
    assert days(20, 20).tolist() == ["repeat"] and days(21, 21).tolist() == ["nurse_verify"]  # a nurse must verify
    assert "scale_verified" in days(21, 30).values                   # nurse confirms the new scale
    assert not days(20, 30).isin(["escalate", "escalate_update"]).any()
    assert days(24, 30).eq("filtered").any()                         # re-baselined after the nurse's call
    assert days(50, 90).isin(["escalate", "escalate_update"]).any()  # real decompensation escalated
    mary = m.decision
    assert "outreach" in mary.values                                 # missed readings followed up
    loops = small_run_loops[0]
    first = loops[loops.patient_id == "DEMO-MARY"].iloc[0]           # Mary misses the message (remote):
    assert first.contacts >= 1 and first.confirmed_by in ("caregiver", "community_health_worker")
    # her plan is followed and her readings are checked 24 h after the action (cardiologist review);
    # when symptoms come back after that check, a new case opens straight away
    ml = loops[loops.patient_id == "DEMO-MARY"]
    assert (ml.outcome == "readings improved after the action").all() and "loop_reopened" not in mary.values
    assert 24 <= first.closed_h - first.action_done_h < 48
    assert days(53, 55).isin(["escalate", "escalate_update"]).any()


def test_reopenings_follow_real_episodes(small_run):
    _, feed, dec, *_ = small_run
    gt = feed.set_index(["patient_id", "day"]).gt_episode
    ro = dec[(dec.decision == "loop_reopened") & dec.notes.fillna("").str.contains("after the action")]
    outside = [not gt.get((r.patient_id, r.day), False) for r in ro.itertuples()]
    assert len(ro) and np.mean(outside) < 0.25


def test_scale_change_not_confirmed_is_escalated(small_run):
    patients, feed, *_ = small_run
    mary = feed[feed.patient_id == "DEMO-MARY"]
    tr = Tracker()
    agent = HeartSafeAgent(CFG, patients[patients.patient_id == "DEMO-MARY"], tr, np.random.default_rng(1),
                           60 * 24.0, scale_truth=set())          # the nurse finds the same scale
    dec = agent.run(PlatformAdapter(mary).observable()).set_index("day")
    assert dec.decision[21] == "nurse_verify" and "scale_verified" not in dec.decision.values
    esc = dec[dec["flags"].fillna("").str.contains("scale_check_rejected")]
    assert len(esc) == 1 and esc.decision.iloc[0] == "escalate" and esc.index[0] <= 30
    assert any(c["kind"] == "scale_check" and c["state"] == "resolved" for c in tr.cases())


def test_end_to_end_metrics(small_run):
    patients, feed, dec, cases, base, horizon, _ = small_run
    res, er = evaluate(feed, patients, dec, cases, base, horizon, CFG["alert_rules"])
    assert res["true_episodes"]["escalated_by_heartsafe"] >= 0.9
    assert res["fault_driven_alerts"]["stopped_by_heartsafe"] >= 0.6
    # every HeartSafe clinical case created 3+ days before the end is acknowledged; the platform alone loses some alerts
    clinical = cases[(cases.kind == "clinical") & (cases.created_h < horizon - 72)]
    assert clinical.acknowledged_h.notna().all()
    assert res["followup"]["platform_alone_alert_items_never_acknowledged"] > 0
    json.dumps(res, default=float)
    res, _ = evaluate(feed, patients, dec, cases, base, horizon, CFG["alert_rules"], loops=small_run_loops[0])
    pl = res["patient_loop"]
    assert pl["n"] > 0 and pl["heartsafe_action_done"] > pl["platform_alone_action_done"]
    assert set(pl["confirmed_by"]) >= {"patient"}


def test_simulated_heart_rate_spells_are_outside_the_patients_limits(small_run):
    patients, feed, *_ = small_run
    assert set(feed.gt_rate) <= {"none", "slow", "fast"} and not feed.gt_rate.eq("none").all()
    pats = patients.set_index("patient_id").to_dict("index")
    for r in feed[feed.gt_rate.ne("none") & feed.submitted].itertuples():
        lim = patient_limits(CFG["alert_rules"], pats[r.patient_id])
        assert r.hr > lim["hr_high"] if r.gt_rate == "fast" else r.hr < lim["hr_low"]
        assert r.alert_level == 2                                     # the platform's own rule fires too


def test_rhythm_and_selfcare_results(small_run):
    patients, feed, dec, cases, base, horizon, _ = small_run
    res, _ = evaluate(feed, patients, dec, cases, base, horizon, CFG["alert_rules"], cfg=CFG)
    rh, sc = res["rhythm"], res["selfcare"]
    # every reliable AF recording of a patient with AF at enrollment is logged as known, never as new
    known = set(patients[patients.atrial_fibrillation.astype(bool)].patient_id)
    named = dec[dec.ecg_af.notna()]
    assert rh["known_af_recordings_not_escalated_as_new"] == int(named.patient_id.isin(known).sum()) > 0
    assert (named[named.patient_id.isin(known)].ecg_af == "known").all()
    assert not dec[dec.patient_id.isin(known)]["flags"].fillna("").str.contains("ecg_new_af").any()
    assert rh["new_af_reports"] == int((dec.ecg_af == "new").sum())
    assert set(rh["new_af_report_severity"]) <= {"caution", "critical", "emergency"}
    # sleep: check-in calls are low priority, at most one per patient inside the cooldown
    checks = dec[dec.decision == "sleep_check"]
    assert sc["sleep_checkins"] == len(checks) > 0 and (checks.severity == "low").all()
    gap = checks.sort_values("day").groupby("patient_id").day.diff().dropna()
    assert (gap >= CFG["selfcare"]["sleep"]["repeat_after_days"]).all()
    assert (cases[cases.kind == "sleep_check"].severity == "low").all()
    assert sc["nights_logged"] == int(feed.sleep_hours.notna().sum())
    assert 0 < sc["breathing"]["mean_sessions_per_day"] <= sc["breathing"]["plan_sessions_per_day"]
    assert res["nurse_queue_per_100_patient_months"]["heartsafe_sleep_checkins"] == sc["checkins_per_100_patient_months"]
    json.dumps(res, default=float)


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeClient:
    """Stands in for anthropic.Anthropic(): scripted tool calls, then submit_review."""
    def __init__(self, script, fail=False):
        self.script, self.fail, self.seen = list(script), fail, []
        self.beta = self
        self.messages = self

    def create(self, **kw):
        if self.fail:
            raise ConnectionError("offline")
        self.seen.append(kw["messages"])
        return _Block(stop_reason="tool_use", content=self.script.pop(0))


def test_orchestrator_tool_loop_and_fallback(small_run):
    from heartsafe.orchestrator import orchestrate
    *_, tr = small_run
    case = next(c for c in tr.cases() if c["kind"] == "clinical")
    submit = {"summary": "Weight is rising.", "why_flagged": "weight gain", "reliability_caveats": [],
              "evidence_used": "readings trusted", "next_step_wording": "Call the patient today."}
    client = _FakeClient([[_Block(type="tool_use", id="t1", name="get_case_facts", input={}),
                           _Block(type="tool_use", id="t2", name="check_ecg", input={})],
                          [_Block(type="tool_use", id="t3", name="submit_review", input=submit)]])
    out = orchestrate(case, tr, "claude-sonnet-5-5", client=client, now_h=1.0)
    assert out["source"] == "claude-sonnet-5-5" and out["tool_calls"] == ["get_case_facts", "check_ecg", "submit_review"]
    assert "Weight is rising." in out["text"] and "clinical team" in out["text"]
    results = client.seen[1][2]["content"]                       # both tool results in ONE user message
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert any(a["action"] == "llm_review" for a in tr.audit(case["case_id"]))
    out = orchestrate(case, tr, "claude-sonnet-5-5", client=_FakeClient([], fail=True))
    assert out["source"] == "template" and "unavailable" in out["text"]


def test_equity_audit_runs(small_run):
    patients, feed, dec, cases, _, _, _ = small_run
    eq, props = audit(feed, patients, dec, cases, CFG)
    assert set(eq.metric) == {"missed_day_rate", "repeat_rate", "ecg_unreliable_rate", "hours_to_ack",
                              "ack_within_deadline", "patient_action_done"}
    assert all(not p["approved"] for p in props)
