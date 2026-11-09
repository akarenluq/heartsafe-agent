"""Run HeartSafe end to end: monitoring-platform feed -> agent -> tracker -> evaluation -> equity audit.

Usage (from the agent/ folder):
  python -m heartsafe.run                         # simulated cohort from config.yaml
  python -m heartsafe.run --patients 300 --days 60
  python -m heartsafe.run --feed platform_export.csv --patients-csv patients.csv
  python -m heartsafe.run --imaging ../imaging/results/clean/exams --ecg ../ecg/results/clean/exams
  python -m heartsafe.run --simulated-models     # ignore pipeline outputs even if present

Real ECG and imaging outputs are used automatically when ../ecg/results and
../imaging/results exist (clean + fault-injected sets). Writes everything the
dashboard needs to data/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA, load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOLS = {
    "ecg": [ROOT / "ecg/results/clean/exams", ROOT / "ecg/results/faulty/exams"],
    "imaging": [ROOT / "imaging/results/clean/exams", ROOT / "imaging/results/faulty/exams"],
}


def safeguard_effect(res, eq, out, safeguards):
    """Before/after an approved safeguard: workload added and the targeted group's response."""
    approved = [s for s in safeguards if s.get("approved")]
    ref_res, ref_eq = out / "evaluation_no_safeguards.json", out / "equity_no_safeguards.csv"
    if not approved:
        json.dump(res, open(ref_res, "w"), indent=2, default=float)
        eq.to_csv(ref_eq, index=False)
        return None
    if not ref_res.exists():
        return {"note": "run once with no approved safeguards first, to have a reference"}
    before, before_eq = json.load(open(ref_res)), pd.read_csv(ref_eq)
    q0, q1 = before["nurse_queue_per_100_patient_months"], res["nurse_queue_per_100_patient_months"]

    def grp(e, s, metric):
        r = e[(e.metric == metric) & (e.group == s["field"]) & (e.value.astype(str) == str(s["value"]))]
        return round(float(r["mean"].iloc[0]), 2) if len(r) else None
    return {
        "extra_clinical_cases_per_100_pm": round(q1["heartsafe_clinical_cases"] - q0["heartsafe_clinical_cases"], 1),
        "extra_false_alarms_per_100_pm": round(q1["false_alarms_heartsafe"] - q0["false_alarms_heartsafe"], 1),
        "extra_reminders": res["followup"]["reminders_sent"] - before["followup"]["reminders_sent"],
        "targeted_groups": [{"safeguard": s["id"],
                             "hours_to_ack_before": grp(before_eq, s, "hours_to_ack"),
                             "hours_to_ack_after": grp(eq, s, "hours_to_ack")} for s in approved],
    }
from .agent import HeartSafeAgent, run_baseline
from .equity import audit
from .evaluate import evaluate
from .monitoring import PlatformAdapter, platform_levels
from .signals import load_outputs
from .simulate import simulate_cohort
from .tracker import Tracker


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--patients", type=int)
    ap.add_argument("--days", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--feed", help="Monitoring-platform CSV export (e.g. Medly) instead of the simulation")
    ap.add_argument("--patients-csv", help="patient attributes for --feed")
    ap.add_argument("--imaging", help="imaging pipeline exam JSON folder(s), comma-separated")
    ap.add_argument("--ecg", help="ECG pipeline output JSON folder(s), comma-separated")
    ap.add_argument("--simulated-models", action="store_true", help="simulate ECG/MRI outputs instead")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.feed:
        feed = pd.read_csv(args.feed)
        patients = pd.read_csv(args.patients_csv)
    else:
        patients, feed = simulate_cohort(cfg, args.patients, args.days, args.seed)
    feed = platform_levels(PlatformAdapter(feed).feed, cfg["alert_rules"], patients)
    horizon = (feed.day.max() + 1) * 24.0

    sg_path = out / "safeguards.json"
    safeguards = json.load(open(sg_path)) if sg_path.exists() else []
    if args.simulated_models:
        ecg_pool, mri_pool = [], []
    else:
        ecg_pool = load_outputs(args.ecg or DEFAULT_POOLS["ecg"])
        mri_pool = load_outputs(args.imaging or DEFAULT_POOLS["imaging"])
    print(f"model outputs: ECG {len(ecg_pool) or 'simulated'}, MRI {len(mri_pool) or 'simulated'}")

    db = out / "heartsafe.db"
    if db.exists():
        db.unlink()
    tr = Tracker(db)
    truth = {(r.patient_id, r.day): bool(r.gt_episode) for r in feed.itertuples()} if "gt_episode" in feed else {}
    scale_truth = {(r.patient_id, r.day) for r in feed[feed.gt_fault == "scale_change"].itertuples()} \
        if "gt_fault" in feed else None
    seed = cfg["cohort"]["seed"] if args.seed is None else args.seed
    agent = HeartSafeAgent(cfg, patients, tr, np.random.default_rng(seed + 1), horizon,
                           ecg_pool=ecg_pool, mri_pool=mri_pool, safeguards=safeguards, truth=truth,
                           scale_truth=scale_truth)
    observable = PlatformAdapter(feed).observable()
    decisions = agent.run(observable)
    tr.commit()
    baseline = run_baseline(observable, patients, cfg, np.random.default_rng(seed + 2), horizon)
    cases = pd.DataFrame(tr.cases()).merge(pd.DataFrame(agent.gt_cases), on="case_id", how="left")

    if patients.attrs.get("screening"):
        json.dump(patients.attrs["screening"], open(out / "screening.json", "w"), indent=2)
    patients.to_csv(out / "patients.csv", index=False)
    feed.to_csv(out / "feed.csv", index=False)
    decisions.to_csv(out / "decisions.csv", index=False)
    baseline.to_csv(out / "baseline.csv", index=False)

    if "gt_episode" in feed:
        res, er = evaluate(feed, patients, decisions, cases, baseline, horizon, cfg["alert_rules"], cfg["followup"])
        res["safeguards_applied"] = [s["id"] for s in safeguards if s.get("approved")]
        res["sources"] = {"ecg": "pipeline outputs" if ecg_pool else "simulated",
                          "mri": "imaging pipeline outputs" if mri_pool else "simulated"}
        er.to_csv(out / "episodes.csv", index=False)
    eq, proposals = audit(feed, patients, decisions, cases, cfg)
    eq.to_csv(out / "equity.csv", index=False)
    if "gt_episode" in feed:
        effect = safeguard_effect(res, eq, out, safeguards)
        if effect:
            res["safeguard_effect"] = effect
        json.dump(res, open(out / "evaluation.json", "w"), indent=2, default=float)
    approved = {s["id"] for s in safeguards if s.get("approved")}
    for p in proposals:
        p["approved"] = p["id"] in approved
    json.dump(proposals, open(out / "proposals.json", "w"), indent=2)
    json.dump({"horizon_h": horizon}, open(out / "run_info.json", "w"))

    print(f"patients {len(patients)}, patient-days {len(feed)}, platform alert days {(feed.alert_level > 0).sum()}")
    print("agent actions:", decisions.decision.value_counts().to_dict())
    print(f"cases {len(cases)} (open at end {(cases.state != 'resolved').sum()}), reminders {cases.reminders.sum()}")
    if "gt_episode" in feed:
        print(json.dumps({k: res[k] for k in ("true_episodes", "fault_driven_alerts",
                                              "nurse_queue_per_100_patient_months")}, indent=2, default=float))
    print(f"equity gaps: {int(eq.gap.sum())}; proposals: {[p['id'] for p in proposals]}")
    print(f"outputs -> {out}")


if __name__ == "__main__":
    main()
