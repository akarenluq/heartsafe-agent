"""Step 0: check the unzipped download really is M&Ms and summarise it.

Usage: python 00_inspect_dataset.py /path/to/unzipped_archive
"""
import sys
from collections import Counter
from pathlib import Path

from common import find_metadata_csv, load_mnms_metadata

root = Path(sys.argv[1])
NII = (".nii", ".nii.gz")  # releases ship either
imgs = sorted(p for p in root.rglob("*_sa.nii*") if p.name.endswith(NII))
gts = sorted(p for p in root.rglob("*_sa_gt.nii*") if p.name.endswith(NII))
print(f"root: {root}")
print(f"short-axis cines (*_sa.nii[.gz]): {len(imgs)}")
print(f"ground truths (*_sa_gt.nii[.gz]): {len(gts)}")

splits = Counter()
for g in gts:
    parts = "/".join(g.parts).lower()
    s = "train" if "training" in parts else "val" if "validation" in parts else "test" if "testing" in parts else "unknown"
    splits[s] += 1
print("labelled cases per split:", dict(splits))

csv = find_metadata_csv(root)
if csv is None:
    print("WARNING: no metadata CSV with an 'External code' column found.")
else:
    meta = load_mnms_metadata(csv)
    print(f"metadata: {csv} ({len(meta)} subjects, columns {list(meta.columns)})")
    for col in ("vendor", "vendor_name", "centre", "sex"):
        if col in meta:
            print(f"  {col}: {dict(meta[col].value_counts())}")

other = Counter(p.suffix for p in root.rglob("*") if p.is_file() and not p.name.endswith(NII))
print("other file types:", dict(other.most_common(8)))
if not gts:
    print("\nThis does not look like M&Ms (no *_sa_gt.nii[.gz]). Send me the folder listing.")
