"""Step 4: turn nnU-Net predictions into exam-level results for the agent.

Per exam (ED + ES of one subject) it computes:
  prediction   LV/RV volumes and EF, LV mass
  probability  mean max-softmax over predicted heart voxels
  uncertainty  mean normalised entropy over predicted heart voxels (+ saved map)
  quality      SNR proxy, LV slice coverage and gaps
  shift_score  Mahalanobis distance of scanner/intensity features to training scans
  plausibility physiological range and shape checks
and a suggested action (accept / repeat / escalate). The agent's policy engine
makes the final call; this is the imaging side of the reliability check.

Ground truth (Dice, true EF) is written under "_eval" for the evaluation only;
the agent must not read it.

Usage:
  python 04_postprocess_reliability.py --raw $nnUNet_raw/Dataset201_HeartSafeMNMs \
      --pred preds/clean --out results/clean [--images DIR --labels DIR --faults faults.csv]
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from common import (LABELS, MYO_DENSITY_G_PER_ML, age_band, dice, ef, largest_cc_fraction,
                    load_config, shift_features, slice_coverage, snr_proxy, voxel_ml)

LV, MYO, RV = LABELS["LV"], LABELS["MYO"], LABELS["RV"]


def build_reference(raw, cache):
    """Shift model and SNR distribution from the training scans."""
    if cache.exists():
        d = np.load(cache)
        return d["feats"], d["snr"]
    feats, snrs = [], []
    for p in sorted((raw / "imagesTr").glob("*_0000.nii.gz")):
        img = nib.load(str(p))
        feats.append(shift_features(img))
        lab = np.asanyarray(nib.load(str(raw / "labelsTr" / p.name.replace("_0000", ""))).dataobj)
        snrs.append(snr_proxy(img.get_fdata(dtype=np.float32), lab == LV))
    feats, snrs = np.array(feats), np.array(snrs)
    np.savez(cache, feats=feats, snr=snrs)
    return feats, snrs


class ShiftModel:
    def __init__(self, feats):
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-6
        self.cov = LedoitWolf().fit((feats - self.mu) / self.sd)
        self.train_d = self(feats)

    def __call__(self, x):
        z = (np.atleast_2d(x) - self.mu) / self.sd
        return np.sqrt(self.cov.mahalanobis(z))


def load_probs(npz_path, seg_shape):
    """nnU-Net saves probabilities as (C, z, y, x); nibabel volumes are (x, y, z)."""
    if not npz_path.exists():
        return None
    p = np.load(npz_path)["probabilities"].astype(np.float32)
    if p.shape[1:] == tuple(seg_shape[::-1]):
        p = np.transpose(p, (0, 3, 2, 1))
    elif p.shape[1:] != tuple(seg_shape):
        raise ValueError(f"{npz_path}: probabilities {p.shape} do not match segmentation {seg_shape}")
    return p


def case_metrics(case, pred_dir, img_dir, lab_dir, unc_dir):
    seg_img = nib.load(str(pred_dir / f"{case}.nii.gz"))
    seg = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.uint8)
    img = nib.load(str(img_dir / f"{case}_0000.nii.gz"))
    vol = img.get_fdata(dtype=np.float32)
    vml = voxel_ml(img)
    r = {
        "lv_ml": (seg == LV).sum() * vml,
        "myo_ml": (seg == MYO).sum() * vml,
        "rv_ml": (seg == RV).sum() * vml,
        "lv_cc_frac": largest_cc_fraction(seg == LV),
        "snr": snr_proxy(vol, seg == LV),
        "shift_feat": shift_features(img),
    }
    r["lv_slices"], r["lv_gap"] = slice_coverage(seg == LV)

    probs = load_probs(pred_dir / f"{case}.npz", seg.shape)
    if probs is not None:
        c = probs.shape[0]
        ent = -(probs * np.log(np.clip(probs, 1e-8, 1))).sum(0) / np.log(c)
        heart = probs.argmax(0) > 0
        r["uncertainty"] = float(ent[heart].mean()) if heart.any() else 1.0
        r["probability"] = float(probs.max(0)[heart].mean()) if heart.any() else 0.0
        out = nib.Nifti1Image(ent.astype(np.float32), seg_img.affine)
        nib.save(out, str(unc_dir / f"{case}_entropy.nii.gz"))
    else:
        r["uncertainty"], r["probability"] = float("nan"), float("nan")

    gt_path = lab_dir / f"{case}.nii.gz" if lab_dir else None
    if gt_path is not None and gt_path.exists():
        gt = np.rint(np.asanyarray(nib.load(str(gt_path)).dataobj)).astype(np.uint8)
        r["gt_lv_ml"] = (gt == LV).sum() * vml
        r["gt_rv_ml"] = (gt == RV).sum() * vml
        for name, k in (("lv", LV), ("myo", MYO), ("rv", RV)):
            r[f"dice_{name}"] = dice(seg == k, gt == k)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="nnUNet_raw/Dataset201_HeartSafeMNMs")
    ap.add_argument("--pred", required=True, help="nnUNetv2_predict output (with --save_probabilities)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", help="default: <raw>/imagesTs")
    ap.add_argument("--labels", help="default: <raw>/labelsTs")
    ap.add_argument("--faults", help="faults.csv from step 5, for faulty runs")
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw, pred, out = Path(args.raw), Path(args.pred), Path(args.out)
    img_dir = Path(args.images) if args.images else raw / "imagesTs"
    lab_dir = Path(args.labels) if args.labels else raw / "labelsTs"
    (out / "exams").mkdir(parents=True, exist_ok=True)
    (out / "uncertainty_maps").mkdir(exist_ok=True)

    meta = pd.read_csv(raw / "metadata.csv")
    meta_by_code = meta.drop_duplicates("code").set_index("code")
    faults = pd.read_csv(args.faults).set_index("code")["fault"] if args.faults else None

    feats, train_snr = build_reference(raw, raw / "reliability_reference.npz")
    shift = ShiftModel(feats)

    # Per-case metrics, then pair ED with ES.
    cases = sorted(p.name[:-7] for p in pred.glob("*.nii.gz"))
    per = {c: case_metrics(c, pred, img_dir, lab_dir, out / "uncertainty_maps") for c in cases}
    codes = sorted({c.rsplit("_", 1)[0] for c in cases})

    pl, q = cfg["plausibility"], cfg["quality"]
    rows = []
    for code in codes:
        ed, es = per.get(f"{code}_ED"), per.get(f"{code}_ES")
        if ed is None or es is None:
            print(f"skip {code}: needs both ED and ES predictions")
            continue
        m = meta_by_code.loc[code] if code in meta_by_code.index else pd.Series(dtype=object)
        row = {
            "code": code,
            "split": m.get("split"), "vendor": m.get("vendor"), "vendor_name": m.get("vendor_name"),
            "centre": m.get("centre"), "sex": m.get("sex"), "age": m.get("age"),
            "age_band": age_band(m.get("age")),
            "fault": "none" if faults is None else faults.get(code, "none"),
            "lv_edv_ml": ed["lv_ml"], "lv_esv_ml": es["lv_ml"], "lvef_pct": ef(ed["lv_ml"], es["lv_ml"]),
            "rv_edv_ml": ed["rv_ml"], "rv_esv_ml": es["rv_ml"], "rvef_pct": ef(ed["rv_ml"], es["rv_ml"]),
            "lv_mass_g": ed["myo_ml"] * MYO_DENSITY_G_PER_ML,
            "probability": np.nanmean([ed["probability"], es["probability"]]),
            "uncertainty": np.nanmax([ed["uncertainty"], es["uncertainty"]]),
            "snr": np.nanmin([ed["snr"], es["snr"]]),
            "lv_slices": min(ed["lv_slices"], es["lv_slices"]),
            "lv_gap": bool(ed["lv_gap"] or es["lv_gap"]),
            "lv_cc_frac": min(ed["lv_cc_frac"], es["lv_cc_frac"]),
            "shift_score": float(max(shift(ed["shift_feat"])[0], shift(es["shift_feat"])[0])),
        }
        if "dice_lv" in ed and "dice_lv" in es:
            for k in ("lv", "myo", "rv"):
                row[f"dice_{k}"] = (ed[f"dice_{k}"] + es[f"dice_{k}"]) / 2
            row["gt_lvef_pct"] = ef(ed["gt_lv_ml"], es["gt_lv_ml"])
            row["gt_rvef_pct"] = ef(ed["gt_rv_ml"], es["gt_rv_ml"])
            row["lvef_abs_err"] = abs(row["lvef_pct"] - row["gt_lvef_pct"])
            row["rvef_abs_err"] = abs(row["rvef_pct"] - row["gt_rvef_pct"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no complete ED/ES exams found in --pred")

    # Thresholds: fixed numbers from config, or calibrated.
    thr_path = raw / "reliability_thresholds.json"
    if thr_path.exists():
        thr = json.load(open(thr_path))  # keep thresholds identical across clean and faulty runs
    else:
        uc = cfg["uncertainty"]
        val = df[(df.split == "val") & (df.fault == "none")]
        calib = val if len(val) >= 10 else df[df.fault == "none"]
        thr = {
            "uncertainty": float(calib.uncertainty.quantile(uc["quantile"])) if uc["threshold"] == "auto" else float(uc["threshold"]),
            "uncertainty_calibrated_on": "val" if len(val) >= 10 else "all clean test exams",
            "shift": float(np.quantile(shift.train_d, cfg["shift"]["quantile"])) if cfg["shift"]["threshold"] == "auto" else float(cfg["shift"]["threshold"]),
            "min_snr": float(np.nanquantile(train_snr, q["min_snr_quantile"])) if q["min_snr"] == "auto" else float(q["min_snr"]),
        }
        json.dump(thr, open(thr_path, "w"), indent=2)
        print(f"calibrated thresholds saved to {thr_path}")
    if "shift_severe" not in thr:
        # Beyond every scanner seen in clean held-out validation exams = unknown input; always escalated.
        val = df[(df.split == "val") & (df.fault == "none")]
        if len(val) < 10:
            raise SystemExit("shift_severe needs clean validation exams: run the clean set first")
        thr["shift_severe"] = float(val.shift_score.quantile(cfg["shift"].get("severe_quantile", 0.99)))
        json.dump(thr, open(thr_path, "w"), indent=2)
        print(f"severe-shift threshold {thr['shift_severe']:.2f} added to {thr_path}")

    def scale_for(r):
        s = 1.0
        for g in cfg.get("safeguards") or []:
            if g.get("approved") and str(r.get(g["field"])) == str(g["value"]):
                s = min(s, float(g.get("uncertainty_scale", 1.0)))
        return s

    actions, actions_base, flags_all = [], [], []
    for _, r in df.iterrows():
        f = []
        if not (r.snr >= thr["min_snr"]):
            f.append("low_snr")
        if r.lv_gap:
            f.append("lv_slice_gap")
        if r.lv_slices < pl["min_lv_slices"]:
            f.append("lv_coverage")
        quality_fail = bool(f)
        lo, hi = pl["lvef_range_pct"]
        if not (lo <= r.lvef_pct <= hi):
            f.append("lvef_out_of_range")
        if r.lv_esv_ml >= r.lv_edv_ml:
            f.append("esv_not_below_edv")
        if not (pl["lv_edv_ml_range"][0] <= r.lv_edv_ml <= pl["lv_edv_ml_range"][1]):
            f.append("lv_edv_out_of_range")
        if not (pl["rv_edv_ml_range"][0] <= r.rv_edv_ml <= pl["rv_edv_ml_range"][1]):
            f.append("rv_edv_out_of_range")
        if r.lv_cc_frac < pl["min_lv_component_fraction"]:
            f.append("lv_fragmented")
        if r.shift_score > thr["shift_severe"]:
            f.append("severe_scanner_shift")
        elif r.shift_score > thr["shift"]:
            f.append("scanner_shift")
        base_unc = r.uncertainty > thr["uncertainty"]
        unc = r.uncertainty > thr["uncertainty"] * scale_for(r)
        # Scanner shift alone: a caveat on the review unless config says escalate
        # (on M&Ms, shift-only exams were barely more often wrong than accepted ones).
        shift_escalates = cfg.get("shift", {}).get("action", "caveat") == "escalate"
        hard = [x for x in f if x != "scanner_shift" or shift_escalates]

        def decide(high_unc):
            if quality_fail:
                return "repeat"
            if hard or high_unc:
                return "escalate"
            return "accept"

        actions.append(decide(unc))
        actions_base.append(decide(base_unc))
        flags_all.append(";".join(f + (["high_uncertainty"] if unc else [])))
    df["flags"], df["suggested_action"], df["action_without_safeguards"] = flags_all, actions, actions_base
    df.to_csv(out / "exam_results.csv", index=False)

    # One JSON per exam in the shared HeartSafe interface.
    eval_cols = [c for c in df.columns if c.startswith(("dice_", "gt_")) or c.endswith("_abs_err")]
    for _, r in df.iterrows():
        rec = {
            "exam_id": r.code, "modality": "cardiac_mri",
            "prediction": {k: round(float(r[k]), 2) for k in
                           ("lv_edv_ml", "lv_esv_ml", "lvef_pct", "rv_edv_ml", "rv_esv_ml", "rvef_pct", "lv_mass_g")},
            "probability": round(float(r.probability), 4),
            "uncertainty": round(float(r.uncertainty), 4),
            "quality": {"snr": round(float(r.snr), 2), "lv_slices": int(r.lv_slices), "lv_slice_gap": bool(r.lv_gap)},
            "shift_score": round(float(r.shift_score), 3),
            "plausibility": {"lv_component_fraction": round(float(r.lv_cc_frac), 3)},
            "flags": [x for x in r["flags"].split(";") if x],
            "suggested_action": r.suggested_action,
            "caveats": (["scanner differs from training scans; result accepted, review with care"]
                        if "scanner_shift" in r["flags"] and r.suggested_action == "accept" else []),
            "explanation_files": {p: f"uncertainty_maps/{r.code}_{p}_entropy.nii.gz" for p in ("ED", "ES")},
            "metadata": {k: (None if pd.isna(r[k]) else r[k]) for k in ("vendor", "vendor_name", "centre", "sex", "age", "split")},
            "thresholds": thr,
            "_eval": {k: (None if pd.isna(r[k]) else float(r[k])) for k in eval_cols} | {"fault": r.fault},
        }
        json.dump(rec, open(out / "exams" / f"{r.code}.json", "w"), indent=2, default=str)

    print(f"{len(df)} exams -> {out}/exam_results.csv")
    print(df.suggested_action.value_counts().to_string())
    if "dice_lv" in df:
        print(f"mean Dice LV/MYO/RV: {df.dice_lv.mean():.3f}/{df.dice_myo.mean():.3f}/{df.dice_rv.mean():.3f}, "
              f"LVEF MAE {df.lvef_abs_err.mean():.1f} points")


if __name__ == "__main__":
    main()
