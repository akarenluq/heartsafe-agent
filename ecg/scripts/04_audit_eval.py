"""Step 4: ECG evaluation and equity audit.

Writes <out>/evaluation.json with
  performance   AUROC (bootstrap 95% CI), sensitivity, specificity on PTB-XL test,
                faulted ECGs and CinC 2017 (device shift: the performance drop)
  selective     the same on ECGs the reliability layer accepted vs all ECGs
  reliability   sensitivity / specificity of repeat-or-escalate on injected faults
                vs a no-agent baseline that accepts everything; detection per fault type
  escalation    share of repeat/escalate decisions where the model was wrong or the input faulty
and <out>/audit/ with subgroup accuracy and unreliable-data rate (sex, age band,
device, site) with bootstrap 95% CIs, forest plots and safeguard proposals (approved: false).

Usage:
  python 04_audit_eval.py --results results/ --out results/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GROUPS = ["sex", "age_band", "device", "site"]


def boot_mean(x, n_boot, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_auc(y, p, n_boot, rng):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return None
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return [round(float(roc_auc_score(y, p)), 4), round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def perf(df, n_boot, rng):
    d = df[df.label >= 0]
    if d.empty:
        return None
    y, pred = d.label.values, d.pred.values
    return {"n": int(len(d)), "abnormal_share": round(float(y.mean()), 3),
            "auroc_ci": boot_auc(y, d.prob.values, n_boot, rng),
            "sensitivity": round(float(pred[y == 1].mean()), 4) if (y == 1).any() else None,
            "specificity": round(float(1 - pred[y == 0].mean()), 4) if (y == 0).any() else None,
            "accuracy": round(float((pred == y).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    clean = pd.read_csv(res / "clean" / "exam_results.csv")
    faulty = pd.read_csv(res / "faulty" / "exam_results.csv")
    cinc = pd.read_csv(res / "cinc2017" / "exam_results.csv")
    for d in (clean, faulty, cinc):
        d["flagged"] = (d.suggested_action != "accept").astype(int)
        d["wrong"] = ((d.label >= 0) & (d.pred != d.label)).astype(int)

    ev = {"performance": {
        "ptbxl_test_clean": perf(clean, args.n_boot, rng),
        "ptbxl_test_faulted_only": perf(faulty[faulty.fault != "none"], args.n_boot, rng),
        "cinc2017_device_shift": perf(cinc, args.n_boot, rng),
    }}
    a, b = ev["performance"]["ptbxl_test_clean"], ev["performance"]["cinc2017_device_shift"]
    if a and b and a["auroc_ci"] and b["auroc_ci"]:
        ev["performance"]["device_shift_auroc_drop"] = round(a["auroc_ci"][0] - b["auroc_ci"][0], 4)

    ev["selective"] = {}
    for name, d in (("ptbxl_clean", clean), ("ptbxl_faulty_mix", faulty), ("cinc2017", cinc)):
        ev["selective"][name] = {"all": perf(d, args.n_boot, rng),
                                 "accepted_only": perf(d[d.flagged == 0], args.n_boot, rng),
                                 "accepted_share": round(float(1 - d.flagged.mean()), 4)}

    pos, neg = faulty[faulty.fault != "none"], faulty[faulty.fault == "none"]
    ev["reliability"] = {
        "injected_faults": {
            "agent": {"sensitivity": round(float(pos.flagged.mean()), 4),
                      "specificity": round(float(1 - neg.flagged.mean()), 4),
                      "n_faulty": int(len(pos)), "n_clean": int(len(neg))},
            "no_agent_baseline": {"sensitivity": 0.0, "specificity": 1.0},
            "detection_by_fault": {k: round(float(v), 4) for k, v in pos.groupby("fault").flagged.mean().items()},
            "action_by_fault": {f: g.suggested_action.value_counts().to_dict() for f, g in faulty.groupby("fault")},
        },
        "device_shift": {
            "ptbxl_clean_flag_rate": round(float(clean.flagged.mean()), 4),
            "cinc2017_flag_rate": round(float(cinc.flagged.mean()), 4),
            "cinc2017_device_shift_flag_rate": round(float(cinc.flags.fillna("").str.contains("device_shift").mean()), 4),
            "cinc2017_noisy_recordings_flag_rate": round(float(cinc[cinc.cinc_class == "~"].flagged.mean()), 4)
            if (cinc.cinc_class == "~").any() else None,
        },
    }
    esc = pd.concat([faulty, cinc])
    esc = esc[esc.flagged == 1]
    appropriate = (esc.wrong == 1) | (esc.fault != "none") | (esc.label < 0)
    ev["escalation"] = {"n_repeat_or_escalate": int(len(esc)),
                        "appropriate_share": round(float(appropriate.mean()), 4) if len(esc) else None,
                        "note": "appropriate = model wrong, input faulty, cross-device or unlabeled-noisy"}

    # Equity audit on clean PTB-XL test.
    rows = []
    clean["correct"] = (clean.pred == clean.label).astype(float)
    for metric, col, higher in (("accuracy", "correct", True), ("unreliable_rate", "flagged", False)):
        o, olo, ohi = boot_mean(clean[col], args.n_boot, rng)
        rows.append({"metric": metric, "group": "overall", "value": "all", "n": len(clean),
                     "mean": o, "ci_low": olo, "ci_high": ohi, "gap": False})
        for g in GROUPS:
            for v, sub in clean.groupby(clean[g].astype(str)):
                if len(sub) < args.min_n or v in ("unknown", "nan"):
                    continue
                m, lo, hi = boot_mean(sub[col], args.n_boot, rng)
                gap = (hi < o) if higher else (lo > o)
                rows.append({"metric": metric, "group": g, "value": v, "n": len(sub),
                             "mean": m, "ci_low": lo, "ci_high": hi, "gap": bool(gap)})
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "audit" / "subgroup_audit.csv", index=False)

    proposals = []
    for _, r in audit[audit.gap].iterrows():
        sub = clean[clean[r.group].astype(str) == r.value]
        rest = clean[clean[r.group].astype(str) != r.value]
        causes = []
        if sub.sqi.median() < rest.sqi.median():
            causes.append(f"lower signal quality (median SQI {sub.sqi.median():.2f} vs {rest.sqi.median():.2f})")
        if sub.shift_score.median() > rest.shift_score.median():
            causes.append(f"device/recording shift (median {sub.shift_score.median():.1f} vs {rest.shift_score.median():.1f})")
        if sub.uncertainty.median() > rest.uncertainty.median():
            causes.append("model is less certain on this subgroup")
        if sub.label.mean() > rest.label.mean() + 0.1:
            causes.append(f"more abnormal ECGs ({sub.label.mean():.0%} vs {rest.label.mean():.0%})")
        proposals.append({"subgroup": {"field": r.group, "value": r.value}, "metric": r.metric,
                          "subgroup_mean": round(r["mean"], 4), "ci": [round(r.ci_low, 4), round(r.ci_high, 4)],
                          "n": int(r.n), "likely_causes": causes or ["no clear cause in the available data"],
                          "proposed_safeguard": {"field": r.group, "value": r.value, "uncertainty_scale": 0.7,
                                                 "approved": False},
                          "requires": "administrator approval"})
    json.dump(proposals, open(out / "audit" / "safeguard_proposals.json", "w"), indent=2)
    extra = int(((clean.suggested_action != "accept") & (clean.action_without_safeguards == "accept")).sum())
    json.dump({"extra_escalations_from_safeguards": extra, "ecgs": len(clean)},
              open(out / "audit" / "safeguard_cost.json", "w"), indent=2)
    ev["equity"] = {"gaps_flagged": int(audit.gap.sum()), "proposals": len(proposals),
                    "extra_escalations_from_approved_safeguards": extra}

    for metric, label in (("accuracy", "Accuracy"), ("unreliable_rate", "Unreliable-data rate (repeat/escalate)")):
        a = audit[audit.metric == metric].iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(a) + 1.2))
        ys = np.arange(len(a))
        colors = ["#c0392b" if g else "#34495e" for g in a.gap]
        ax.errorbar(a["mean"], ys, xerr=[a["mean"] - a.ci_low, a.ci_high - a["mean"]], fmt="none", ecolor=colors, lw=1.4)
        ax.scatter(a["mean"], ys, c=colors, s=18, zorder=3)
        ax.axvline(a[a.group == "overall"]["mean"].iloc[0], color="#888", ls="--", lw=1)
        ax.set_yticks(ys, [f"{g}={v} (n={n})" if g != "overall" else f"overall (n={n})"
                           for g, v, n in zip(a.group, a.value, a.n)], fontsize=8)
        ax.set_xlabel(f"{label}, mean with 95% bootstrap CI")
        ax.set_title("ECG, PTB-XL test. Red = subgroup CI entirely worse than overall", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "audit" / f"forest_{metric}.png", dpi=180)
        plt.close(fig)

    json.dump(ev, open(out / "evaluation.json", "w"), indent=2)
    print(json.dumps(ev, indent=2))
    print(audit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
