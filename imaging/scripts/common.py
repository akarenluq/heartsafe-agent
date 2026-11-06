"""Shared helpers for the HeartSafe imaging path."""
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

LABELS = {"background": 0, "LV": 1, "MYO": 2, "RV": 3}
MYO_DENSITY_G_PER_ML = 1.05


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _norm(c):
    return re.sub(r"[^a-z]", "", str(c).lower())


# Column names differ slightly between M&Ms releases; match loosely.
_COLMAP = {
    "code": ["externalcode", "code", "subjectcode"],
    "vendor": ["vendor"],
    "vendor_name": ["vendorname"],
    "centre": ["centre", "center"],
    "ed": ["ed"],
    "es": ["es"],
    "sex": ["sex", "gender"],
    "age": ["age"],
}


def find_metadata_csv(root):
    for c in sorted(Path(root).rglob("*.csv")):
        try:
            cols = {_norm(x) for x in pd.read_csv(c, nrows=1).columns}
        except Exception:
            continue
        if "externalcode" in cols:
            return c
    return None


def load_mnms_metadata(csv_path):
    df = pd.read_csv(csv_path)
    m = {_norm(c): c for c in df.columns}
    out = {}
    for key, cands in _COLMAP.items():
        for cand in cands:
            if cand in m:
                out[key] = df[m[cand]]
                break
    meta = pd.DataFrame(out)
    meta["code"] = meta["code"].astype(str).str.strip()
    return meta.drop_duplicates("code").set_index("code")


def voxel_ml(img):
    return float(np.prod(img.header.get_zooms()[:3])) / 1000.0


def noise_sigma(vol):
    """Robust per-slice noise estimate (Laplacian MAD) inside the body."""
    sig = []
    p99 = np.percentile(vol, 99) + 1e-6
    for k in range(vol.shape[2]):
        s = vol[..., k].astype(np.float64)
        body = s > 0.05 * p99
        if body.sum() < 50:
            continue
        lap = ndimage.laplace(s)[body]
        mad = np.median(np.abs(lap - np.median(lap)))
        sig.append(1.4826 * mad / np.sqrt(20.0))
    return float(np.median(sig)) if sig else float("nan")


def snr_proxy(vol, lv_mask):
    """Mean LV blood-pool intensity over the noise estimate."""
    if lv_mask.sum() == 0:
        return float("nan")
    sigma = noise_sigma(vol)
    return float(vol[lv_mask].mean() / sigma) if sigma > 0 else float("nan")


def shift_features(img):
    """Acquisition and intensity features that differ between scanners."""
    vol = img.get_fdata(dtype=np.float32)
    z = img.header.get_zooms()[:3]
    p50, p99 = np.percentile(vol, [50, 99])
    s = vol / (p99 + 1e-6)
    body = s > 0.05
    sb = s[body] if body.any() else s.ravel()
    return np.array([
        np.log(z[0] * z[1]),          # in-plane resolution
        np.log(z[2]),                 # slice thickness
        np.log(vol.shape[0] * vol.shape[1]),  # matrix size
        np.log1p(max(p99, 0)),        # raw intensity scale
        p50 / (p99 + 1e-6),
        float(body.mean()),
        float(np.mean(sb > 0.5)),
        float(np.std(sb)),
    ], dtype=np.float64)


def largest_cc_fraction(mask):
    if mask.sum() == 0:
        return 0.0
    lab, _ = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max() / sizes.sum())


def slice_coverage(mask):
    """Number of slices with the structure and whether any slice in between is empty."""
    present = mask.reshape(-1, mask.shape[2]).any(axis=0)
    idx = np.where(present)[0]
    if idx.size == 0:
        return 0, True
    gap = bool((~present[idx[0]:idx[-1] + 1]).any())
    return int(idx.size), gap


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s > 0 else 1.0


def ef(edv, esv):
    return float(100.0 * (edv - esv) / edv) if edv and edv > 0 else float("nan")


def age_band(age):
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if np.isnan(a):
        return "unknown"
    return "<40" if a < 40 else ("40-59" if a < 60 else "60+")
