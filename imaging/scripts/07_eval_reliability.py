"""Step 7: imaging reliability metrics against the no-agent baseline.

Baseline = every prediction accepted. The agent flags an exam when its
suggested action is repeat or escalate. An exam is "unreliable" when it has an
injected fault or its prediction is wrong (LVEF error above --ef-tol points
or mean LV Dice below --dice-tol).

Usage:
  python 07_eval_reliability.py --clean results/clean/exam_results.csv \
      --faulty results/faulty/exam_results.csv --out results/reliability.json
"""
import argparse
import json

import pandas as pd


def rates(flagged, positive):
    tp = int((flagged & positive).sum()); fn = int((~flagged & positive).sum())
    tn = int((~flagged & ~positive).sum()); fp = int((flagged & ~positive).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": round(sens, 3), "specificity": round(spec, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--faulty", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ef-tol", type=float, default=10.0)
    ap.add_argument("--dice-tol", type=float, default=0.80)
    args = ap.parse_args()

    clean = pd.read_csv(args.clean).assign(run="clean")
    faulty = pd.read_csv(args.faulty)
    faulty = faulty[faulty.fault != "none"].assign(run="faulty")
    df = pd.concat([clean, faulty], ignore_index=True)

    flagged = df.suggested_action != "accept"
    wrong = (df.lvef_abs_err > args.ef_tol) | (df.dice_lv < args.dice_tol)
    unreliable = (df.fault != "none") | wrong

    res = {
        "exams": len(df),
        "definition": f"unreliable = injected fault, or LVEF error > {args.ef_tol} points, or LV Dice < {args.dice_tol}",
        "agent_vs_unreliable": rates(flagged, unreliable),
        "agent_vs_injected_faults": rates(flagged, df.fault != "none"),
        "baseline": "accepts every exam: sensitivity 0, specificity 1",
        "appropriate_escalation_share": round(float(unreliable[flagged].mean()), 3) if flagged.any() else None,
        "missed_wrong_predictions_baseline": int(wrong.sum()),
        "missed_wrong_predictions_agent": int((wrong & ~flagged).sum()),
        "by_fault": {f: {"n": int(len(g)), "flagged_share": round(float((g.suggested_action != "accept").mean()), 3)}
                     for f, g in df.groupby("fault")},
        "actions": df.suggested_action.value_counts().to_dict(),
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
