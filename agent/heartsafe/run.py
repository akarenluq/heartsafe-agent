"""Run HeartSafe end to end: Medly-style feed -> agent -> tracker -> evaluation -> equity audit.

Usage (from the agent/ folder):
  python -m heartsafe.run                         # simulated cohort from config.yaml
  python -m heartsafe.run --patients 300 --days 60
  python -m heartsafe.run --feed my_medly_export.csv --patients-csv patients.csv
  python -m heartsafe.run --imaging ../imaging/results/clean/exams --ecg ../ecg/results/exams

Writes everything the dashboard needs to data/.
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import DATA, load_config
from .agent import HeartSafeAgent, run_baseline
from .equity import audit
from .evaluate import evaluate
from .medly import MedlyAdapter, medly_levels
from .signals import load_outputs
from .simulate import simulate_cohort
from .tracker import Tracker


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--patients", type=int)
    ap.add_argument("--days", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--feed", help="Medly-style CSV export instead of the simulation")
    ap.add_argument("--patients-csv", help="patient attributes for --feed")
    ap.add_argument("--imaging", help="imaging pipeline exam JSONs (shared HeartSafe format)")
    ap.add_argument("--ecg", help="ECG pipeline output JSONs (shared HeartSafe format)")
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
    feed = medly_levels(MedlyAdapter(feed).feed, cfg["medly_rules"])
    horizon = (feed.day.max() + 1) * 24.0

    sg_path = out / "safeguards.json"
    safeguards = json.load(open(sg_path)) if sg_path.exists() else []
    ecg_pool, mri_pool = load_outputs(args.ecg), load_outputs(args.imaging)

    db = out / "heartsafe.db"
    if db.exists():
        db.unlink()
    tr = Tracker(db)
    truth = {(r.patient_id, r.day): bool(r.gt_episode) for r in feed.itertuples()} if "gt_episode" in feed else {}
    seed = cfg["cohort"]["seed"] if args.seed is None else args.seed
    agent = HeartSafeAgent(cfg, patients, tr, np.random.default_rng(seed + 1), horizon,
                           ecg_pool=ecg_pool, mri_pool=mri_pool, safeguards=safeguards, truth=truth)
    observable = MedlyAdapter(feed).observable()
    decisions = agent.run(observable)
    tr.commit()
    baseline = run_baseline(observable, patients, cfg, np.random.default_rng(seed + 2), horizon)
    cases = pd.DataFrame(tr.cases())

    patients.to_csv(out / "patients.csv", index=False)
    feed.to_csv(out / "feed.csv", index=False)
    decisions.to_csv(out / "decisions.csv", index=False)
    baseline.to_csv(out / "baseline.csv", index=False)

    if "gt_episode" in feed:
        res, er = evaluate(feed, patients, decisions, cases, baseline, horizon, cfg["medly_rules"])
        res["safeguards_applied"] = [s["id"] for s in safeguards if s.get("approved")]
        res["sources"] = {"ecg": "pipeline outputs" if ecg_pool else "simulated",
                          "mri": "imaging pipeline outputs" if mri_pool else "simulated"}
        json.dump(res, open(out / "evaluation.json", "w"), indent=2, default=float)
        er.to_csv(out / "episodes.csv", index=False)
    eq, proposals = audit(feed, patients, decisions, cases, cfg)
    eq.to_csv(out / "equity.csv", index=False)
    approved = {s["id"] for s in safeguards if s.get("approved")}
    for p in proposals:
        p["approved"] = p["id"] in approved
    json.dump(proposals, open(out / "proposals.json", "w"), indent=2)
    json.dump({"horizon_h": horizon}, open(out / "run_info.json", "w"))

    print(f"patients {len(patients)}, patient-days {len(feed)}, Medly alert days {(feed.medly_level > 0).sum()}")
    print("agent actions:", decisions.decision.value_counts().to_dict())
    print(f"cases {len(cases)} (open at end {(cases.state != 'resolved').sum()}), reminders {cases.reminders.sum()}")
    if "gt_episode" in feed:
        print(json.dumps({k: res[k] for k in ("true_episodes", "fault_driven_alerts",
                                              "nurse_queue_per_100_patient_months")}, indent=2, default=float))
    print(f"equity gaps: {int(eq.gap.sum())}; proposals: {[p['id'] for p in proposals]}")
    print(f"outputs -> {out}")


if __name__ == "__main__":
    main()
