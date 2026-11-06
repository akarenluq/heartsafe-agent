"""Step 5: make a faulty copy of the held-out scans with known faults.

These give ground truth for "unreliable input" in the evaluation. The same fault
is applied to ED and ES of an exam. Faults: noise, blur, slice_drop, contrast.

Usage:
  python 05_inject_faults.py --raw $nnUNet_raw/Dataset201_HeartSafeMNMs --out faulty --fraction 0.5
Writes faulty/imagesTs, faulty/labelsTs (labels unchanged) and faulty/faults.csv.
"""
import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

FAULTS = ("noise", "blur", "slice_drop", "contrast")


def apply(vol, fault, rng):
    p99 = np.percentile(vol, 99)
    if fault == "noise":  # Rician-like noise, as from a weak coil or fast scan
        s = 0.12 * p99
        return np.sqrt((vol + rng.normal(0, s, vol.shape)) ** 2 + rng.normal(0, s, vol.shape) ** 2)
    if fault == "blur":  # in-plane blur, a stand-in for motion or breathing
        return ndimage.gaussian_filter(vol, sigma=(2.5, 2.5, 0))
    if fault == "slice_drop":  # two mid-ventricular slices lost
        v = vol.copy()
        k = v.shape[2] // 2
        v[..., k - 1:k + 1] = 0
        return v
    if fault == "contrast":  # poor contrast / wrong windowing
        n = np.clip(vol / (p99 + 1e-6), 0, None)
        return (n ** 0.35) * p99 * 0.3
    raise ValueError(fault)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, default=0.5, help="share of exams that get a fault")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    (out / "imagesTs").mkdir(parents=True, exist_ok=True)
    (out / "labelsTs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = pd.read_csv(raw / "metadata.csv")
    codes = sorted(meta[meta.nnunet_set == "test"].code.unique())
    chosen = rng.permutation(codes)[: int(round(args.fraction * len(codes)))]
    plan = {c: FAULTS[i % len(FAULTS)] for i, c in enumerate(sorted(chosen))}

    rows = []
    for code in codes:
        fault = plan.get(code, "none")
        rows.append({"code": code, "fault": fault})
        for phase in ("ED", "ES"):
            src = raw / "imagesTs" / f"{code}_{phase}_0000.nii.gz"
            lab = raw / "labelsTs" / f"{code}_{phase}.nii.gz"
            if not src.exists():
                continue
            if fault == "none":
                shutil.copy(src, out / "imagesTs" / src.name)
            else:
                img = nib.load(str(src))
                v = apply(img.get_fdata(dtype=np.float32), fault, rng).astype(np.float32)
                o = nib.Nifti1Image(v, img.affine)
                o.header.set_zooms(img.header.get_zooms()[:3])
                nib.save(o, str(out / "imagesTs" / src.name))
            if lab.exists():
                shutil.copy(lab, out / "labelsTs" / lab.name)
    pd.DataFrame(rows).to_csv(out / "faults.csv", index=False)
    print(pd.DataFrame(rows).fault.value_counts().to_string())


if __name__ == "__main__":
    main()
