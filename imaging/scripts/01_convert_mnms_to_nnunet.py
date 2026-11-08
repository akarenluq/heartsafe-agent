"""Step 1: convert M&Ms 4D cines into an nnU-Net v2 raw dataset.

Each labelled subject gives two 3D cases, ED and ES (the only annotated frames).
Original Training split -> imagesTr/labelsTr; Validation + Testing -> imagesTs/labelsTs.
Also writes metadata.csv (vendor, centre, sex, age, split) for the audit.

Usage:
  python 01_convert_mnms_to_nnunet.py --mnms /path/to/unzipped --out $nnUNet_raw --config ../config.yaml
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from common import LABELS, find_metadata_csv, load_config, load_mnms_metadata


def split_of(path):
    parts = "/".join(path.parts).lower()
    if "training" in parts:
        return "train"
    if "validation" in parts:
        return "val"
    if "testing" in parts:
        return "test"
    return "unknown"


def save3d(vol, ref, path, dtype):
    out = nib.Nifti1Image(vol.astype(dtype), ref.affine)
    out.header.set_zooms(ref.header.get_zooms()[:3])
    nib.save(out, str(path))


def labelled_frames(gt4):
    sums = gt4.reshape(-1, gt4.shape[3]).astype(bool).sum(axis=0)
    return [int(t) for t in np.where(sums > 0)[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnms", required=True)
    ap.add_argument("--out", required=True, help="nnUNet_raw folder")
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.mnms)
    ds = Path(args.out) / cfg["dataset_name"]
    for d in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds / d).mkdir(parents=True, exist_ok=True)

    csv = find_metadata_csv(root)
    meta = load_mnms_metadata(csv) if csv else pd.DataFrame()
    if csv is None:
        print("WARNING: no metadata CSV found; ED/ES taken from labelled frames and the audit will lack subgroups.")

    rows, n_train, skipped = [], 0, []
    # Releases ship either .nii.gz or plain .nii
    for gt_path in sorted(p for p in root.rglob("*_sa_gt.nii*") if p.name.endswith((".nii", ".nii.gz"))):
        code, ext = gt_path.name.split("_sa_gt")
        img_path = gt_path.with_name(f"{code}_sa{ext}")
        if not img_path.exists():
            skipped.append((code, "no image"))
            continue
        img = nib.load(str(img_path))
        gt = nib.load(str(gt_path))
        im4 = np.asanyarray(img.dataobj)
        gt4 = np.rint(np.asanyarray(gt.dataobj)).astype(np.uint8)
        if im4.ndim != 4 or im4.shape != gt4.shape:
            skipped.append((code, f"shape {im4.shape} vs {gt4.shape}"))
            continue

        m = meta.loc[code] if code in meta.index else None
        frames = labelled_frames(gt4)
        if m is not None and "ed" in m and "es" in m and not pd.isna(m["ed"]):
            ed, es = int(m["ed"]), int(m["es"])
        elif len(frames) >= 2:
            # Fallback: the labelled frame with the larger LV is ED.
            lv = [(gt4[..., t] == LABELS["LV"]).sum() for t in frames[:2]]
            ed, es = (frames[0], frames[1]) if lv[0] >= lv[1] else (frames[1], frames[0])
        else:
            skipped.append((code, "fewer than two labelled frames"))
            continue
        if ed not in frames or es not in frames:
            skipped.append((code, f"ED/ES {ed}/{es} not labelled (labelled: {frames})"))
            continue

        split = split_of(gt_path)
        in_train = split in cfg["train_splits"]
        idir, ldir = ("imagesTr", "labelsTr") if in_train else ("imagesTs", "labelsTs")
        for phase, t in (("ED", ed), ("ES", es)):
            case = f"{code}_{phase}"
            save3d(im4[..., t].astype(np.float32), img, ds / idir / f"{case}_0000.nii.gz", np.float32)
            save3d(gt4[..., t], img, ds / ldir / f"{case}.nii.gz", np.uint8)
            n_train += int(in_train)
            rows.append({
                "case_id": case, "code": code, "phase": phase, "frame": t, "split": split,
                "nnunet_set": "train" if in_train else "test",
                "vendor": None if m is None else m.get("vendor"),
                "vendor_name": None if m is None else m.get("vendor_name"),
                "centre": None if m is None else m.get("centre"),
                "sex": None if m is None else m.get("sex"),
                "age": None if m is None else m.get("age"),
            })

    pd.DataFrame(rows).to_csv(ds / "metadata.csv", index=False)
    json.dump({
        "channel_names": {"0": "cineMRI"},
        "labels": LABELS,
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": cfg["dataset_name"],
        "description": "M&Ms ED/ES short-axis frames for HeartSafe Agent",
    }, open(ds / "dataset.json", "w"), indent=2)

    df = pd.DataFrame(rows)
    print(f"wrote {ds}")
    print(f"training cases: {n_train}, test cases: {len(df) - n_train}")
    if len(df):
        print(df.groupby(["nnunet_set", "split"]).code.nunique().rename("subjects"))
    if skipped:
        print(f"skipped {len(skipped)} subjects, first few: {skipped[:5]}")


if __name__ == "__main__":
    main()
