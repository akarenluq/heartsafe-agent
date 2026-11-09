import json

import numpy as np
import pandas as pd
import pytest

from heartsafe import load_config
from heartsafe.agent import HeartSafeAgent, run_baseline
from heartsafe.equity import audit
from heartsafe.evaluate import evaluate
from heartsafe.monitoring import PlatformAdapter, platform_levels, rule_levels
from heartsafe.reliability import assess_ecg, assess_reading
from heartsafe.signals import ecg_quality_index
from heartsafe.simulate import simulate_cohort
from heartsafe.tracker import PermissionDenied, Tracker

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


@pytest.fixture(scope="module")
def small_run():
    patients, feed = simulate_cohort(CFG, n_patients=150, n_days=60, seed=3)
    feed = platform_levels(feed, CFG["alert_rules"])
    horizon = 60 * 24.0
    tr = Tracker()
    truth = {(r.patient_id, r.day): bool(r.gt_episode) for r in feed.itertuples()}
    agent = HeartSafeAgent(CFG, patients, tr, np.random.default_rng(1), horizon, truth=truth)
    obs = PlatformAdapter(feed).observable()
    assert not any(c.startswith("gt_") for c in obs.columns)
    dec = agent.run(obs)
    base = run_baseline(obs, patients, CFG, np.random.default_rng(2), horizon)
    return patients, feed, dec, pd.DataFrame(tr.cases()), base, horizon, tr


def test_demo_patient_story(small_run):
    _, _, dec, _, _, _, _ = small_run
    mary = dec[dec.patient_id == "DEMO-MARY"].set_index("day").decision
    assert mary[20] == "repeat" and mary[21] == "filtered"           # scale change caught, not escalated
    assert any(mary.loc[50:].isin(["escalate", "escalate_update"]))  # real decompensation escalated
    assert "outreach" in mary.values                                 # missed readings followed up


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
                              "ack_within_deadline"}
    assert all(not p["approved"] for p in props)
