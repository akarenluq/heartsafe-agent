"""Step 6: equity audit of the imaging model on held-out exams.

For each subgroup (vendor, centre, sex, age band) it reports LV Dice and LVEF
absolute error with bootstrap 95% CIs, flags a gap when the subgroup's whole CI
is worse than the overall estimate, and writes safeguard proposals for an
administrator to approve (they are copied into config.yaml only by a human).

Usage:
  python 06_subgroup_audit.py --results results/clean/exam_results.csv --out results/audit
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import age_band

METRICS = {  # name: (column, higher_is_better, label)
    "lv_dice": ("dice_lv", True, "LV Dice"),
    "lvef_err": ("lvef_abs_err", False, "LVEF absolute error (points)"),
}
GROUPS = ["vendor_name", "vendor", "centre", "sex", "age_band"]


def boot_ci(x, n_boot, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metadata", help="<raw>/metadata.csv, to check training representation")
    ap.add_argument("--splits", default="val,test", help="original splits to audit")
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train_meta = None
    if args.metadata:
        tm = pd.read_csv(args.metadata)
        tm = tm[tm.nnunet_set == "train"].drop_duplicates("code").copy()
        tm["age_band"] = tm.age.map(age_band)
        train_meta = tm
    df = pd.read_csv(args.results)
    df = df[df.split.astype(str).isin(args.splits.split(",")) & (df.fault == "none")]
    groups = [g for g in GROUPS if g in df and df[g].notna().any()]
    if "vendor_name" in groups and "vendor" in groups:
        groups.remove("vendor")  # same grouping, keep the readable one

    rows = []
    for mkey, (col, higher, label) in METRICS.items():
        if col not in df:
            continue
        overall, olo, ohi = boot_ci(df[col], args.n_boot, rng)
        rows.append({"metric": mkey, "group": "overall", "value": "all", "n": len(df),
                     "mean": overall, "ci_low": olo, "ci_high": ohi, "gap": False})
        for g in groups:
            for v, sub in df.groupby(df[g].astype(str)):
                mean, lo, hi = boot_ci(sub[col], args.n_boot, rng)
                gap = len(sub) >= args.min_n and ((hi < overall) if higher else (lo > overall))
                rows.append({"metric": mkey, "group": g, "value": v, "n": len(sub),
                             "mean": mean, "ci_low": lo, "ci_high": hi, "gap": bool(gap)})
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "subgroup_audit.csv", index=False)

    # Likely causes, checked against the data rather than guessed.
    proposals = []
    for _, r in audit[audit.gap].iterrows():
        sub = df[df[r.group].astype(str) == r.value]
        rest = df[df[r.group].astype(str) != r.value]
        causes = []
        if train_meta is not None and r.group in train_meta:
            n_train = int((train_meta[r.group].astype(str) == r.value).sum())
            share = n_train / max(len(train_meta), 1)
            if share < 0.10:
                causes.append(f"under-represented in training ({n_train} of {len(train_meta)} training exams)")
        if sub.snr.median() < rest.snr.median():
            causes.append(f"lower image quality (median SNR {sub.snr.median():.1f} vs {rest.snr.median():.1f})")
        if sub.shift_score.median() > rest.shift_score.median():
            causes.append(f"scanner/intensity shift (median {sub.shift_score.median():.2f} vs {rest.shift_score.median():.2f})")
        if sub.uncertainty.median() > rest.uncertainty.median():
            causes.append("model is less certain on this subgroup")
        proposals.append({
            "subgroup": {"field": r.group, "value": r.value},
            "metric": r.metric, "subgroup_mean": round(r["mean"], 3),
            "ci": [round(r.ci_low, 3), round(r.ci_high, 3)], "n": int(r.n),
            "likely_causes": causes or ["no clear cause in the available data"],
            "proposed_safeguard": {"field": r.group, "value": r.value, "uncertainty_scale": 0.7, "approved": False},
            "requires": "administrator approval",
        })
    json.dump(proposals, open(out / "safeguard_proposals.json", "w"), indent=2)

    # Extra escalations caused by approved safeguards.
    extra = None
    if {"suggested_action", "action_without_safeguards"} <= set(df.columns):
        extra = int(((df.suggested_action != "accept") & (df.action_without_safeguards == "accept")).sum())
        json.dump({"extra_escalations_from_safeguards": extra, "exams": len(df)},
                  open(out / "safeguard_cost.json", "w"), indent=2)

    # Forest plot per metric.
    for mkey, (col, higher, label) in METRICS.items():
        a = audit[(audit.metric == mkey)]
        if a.empty:
            continue
        a = a.iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(a) + 1.2))
        ys = np.arange(len(a))
        colors = ["#c0392b" if g else "#34495e" for g in a.gap]
        ax.errorbar(a["mean"], ys, xerr=[a["mean"] - a.ci_low, a.ci_high - a["mean"]], fmt="none", ecolor=colors, lw=1.4)
        ax.scatter(a["mean"], ys, c=colors, s=18, zorder=3)
        ax.axvline(a[a.group == "overall"]["mean"].iloc[0], color="#888", ls="--", lw=1)
        ax.set_yticks(ys, [f"{g}={v} (n={n})" if g != "overall" else f"overall (n={n})" for g, v, n in zip(a.group, a.value, a.n)], fontsize=8)
        ax.set_xlabel(f"{label}, mean with 95% bootstrap CI")
        ax.set_title("Red = subgroup CI entirely worse than overall", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"forest_{mkey}.png", dpi=180)
        plt.close(fig)

    print(audit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(proposals)} safeguard proposal(s) -> {out / 'safeguard_proposals.json'}")
    if extra is not None:
        print(f"extra escalations from approved safeguards: {extra}")


if __name__ == "__main__":
    main()
