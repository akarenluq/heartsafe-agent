"""Step 1: PTB-XL and PhysioNet/CinC 2017 -> lead I, 250 Hz, 10 s windows.

PTB-XL label (lead I, binary):
  normal   = only the NORM diagnostic superclass and no rhythm abnormality
  abnormal = any other diagnostic superclass (MI, STTC, CD, HYP) or an abnormal rhythm
  records with no diagnostic statement are dropped
CinC 2017 (AliveCor single-lead, a different device): N -> normal, A and O -> abnormal,
'~' (too noisy to label) -> unknown, kept as naturally poor-quality recordings.

Writes <out>/ptbxl.npz + ptbxl_meta.csv and <out>/cinc2017.npz + cinc2017_meta.csv.

Usage:
  python 01_prepare.py --ptbxl /path/ptbxl_unzipped --cinc /path/cinc_unzipped --out work/ --config ../config.yaml
"""
import argparse
import ast
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from common import fix_length, load_config, resample

RHYTHM_ABNORMAL = {"AFIB", "AFLT", "SVTAC", "PSVT", "BIGU", "TRIGU", "PACE", "SVARR"}
NOISE_COLS = ["static_noise", "burst_noise", "baseline_drift", "electrodes_problems"]


def mentions_lead_i(note):
    """PTB-XL noise notes list leads ('I-AVR', 'alles', ', V1,'): does lead I take part?"""
    if pd.isna(note):
        return False
    s = str(note).upper().replace(" ", "")
    if "ALLES" in s or "ALL" in s:
        return True
    return any(t == "I" or t.startswith("I-") for t in re.split(r"[,;]", s))


def read_lead(path, lead, fs_out, n_out):
    sig, fields = wfdb.rdsamp(str(path))
    names = [s.upper() for s in fields["sig_name"]]
    ch = names.index(lead.upper()) if lead.upper() in names else 0
    x = np.nan_to_num(sig[:, ch])
    return fix_length(resample(x, fields["fs"], fs_out), n_out, fs_out)


def prepare_ptbxl(root, cfg, workers, limit):
    db_csv = next(Path(root).rglob("ptbxl_database.csv"))
    base = db_csv.parent
    df = pd.read_csv(db_csv, index_col="ecg_id")
    stm = pd.read_csv(base / "scp_statements.csv", index_col=0)
    diag = stm[stm.diagnostic == 1]

    rows = []
    for ecg_id, r in df.iterrows():
        codes = set(ast.literal_eval(r.scp_codes).keys())
        supers = {diag.loc[c, "diagnostic_class"] for c in codes if c in diag.index}
        rhythm = codes & RHYTHM_ABNORMAL
        if not supers and not rhythm:
            continue
        label = 0 if supers == {"NORM"} and not rhythm else 1
        age = r.age if pd.notna(r.age) and r.age < 120 else (90 if pd.notna(r.age) else np.nan)  # 300 = anonymised >89
        rows.append({"exam_id": f"ptbxl_{ecg_id}", "dataset": "ptbxl", "path": str(base / r.filename_hr),
                     "label": label, "fold": int(r.strat_fold), "patient_id": r.patient_id,
                     "sex": {0: "M", 1: "F"}.get(r.sex, "unknown"), "age": age,
                     "device": " ".join(str(r.device).split()), "site": "unknown" if pd.isna(r.site) else str(int(r.site)),
                     "superclasses": ";".join(sorted(supers)), "rhythm": ";".join(sorted(rhythm)),
                     # Real recording problems annotated by PTB-XL experts, on lead I
                     "lead1_noise": ";".join(c for c in NOISE_COLS if c in r and mentions_lead_i(r[c]))})
    meta = pd.DataFrame(rows)
    if limit:
        meta = meta.groupby("fold", group_keys=False).head(limit)
    n = cfg["fs"] * cfg["seconds"]
    with Pool(workers) as pool:
        X = pool.map(partial(read_lead, lead=cfg["lead"], fs_out=cfg["fs"], n_out=n), meta.path, chunksize=64)
    return np.stack(X).astype(np.float32), meta.drop(columns="path")


def prepare_cinc(root, cfg, workers, limit):
    ref_csv = next(Path(root).rglob("REFERENCE.csv"))
    base = ref_csv.parent
    ref = pd.read_csv(ref_csv, header=None, names=["record", "cinc_class"])
    if limit:
        ref = ref.groupby("cinc_class", group_keys=False).head(limit)
    label = ref.cinc_class.map({"N": 0, "A": 1, "O": 1, "~": -1})
    meta = pd.DataFrame({"exam_id": "cinc2017_" + ref.record, "dataset": "cinc2017", "label": label,
                         "fold": -1, "cinc_class": ref.cinc_class, "sex": "unknown", "age": np.nan,
                         "device": "AliveCor", "site": "cinc2017"})
    n = cfg["fs"] * cfg["seconds"]
    with Pool(workers) as pool:
        X = pool.map(partial(read_lead, lead=cfg["lead"], fs_out=cfg["fs"], n_out=n),
                     [base / r for r in ref.record], chunksize=64)
    return np.stack(X).astype(np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--cinc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="quick test: records per fold/class")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, fn, root in (("ptbxl", prepare_ptbxl, args.ptbxl), ("cinc2017", prepare_cinc, args.cinc)):
        X, meta = fn(root, cfg, args.workers, args.limit)
        np.savez(out / f"{name}.npz", X=X)
        meta.to_csv(out / f"{name}_meta.csv", index=False)
        print(f"{name}: {X.shape}, labels {meta.label.value_counts().to_dict()}")
        if name == "ptbxl":
            print(meta.groupby("fold").label.agg(["count", "mean"]).rename(columns={"mean": "abnormal_share"}))
            print("devices:", meta.device.value_counts().head(8).to_dict())
            print("lead I annotated noise:", int((meta.lead1_noise.fillna("") != "").sum()))


if __name__ == "__main__":
    main()
