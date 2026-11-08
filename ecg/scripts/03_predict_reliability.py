"""Step 3: ensemble predictions + reliability checks -> shared HeartSafe JSON per ECG.

For every ECG: abnormal probability (ensemble mean), uncertainty (ensemble spread,
scaled to 0-1), signal quality index and heart rate, device-shift score (Mahalanobis
distance of the embedding from PTB-XL training ECGs), flags and a suggested action:
  repeat    poor signal quality or implausible heart rate -> ask for a new recording
  escalate  high model uncertainty or device shift -> clinician review
  accept    otherwise

Three sets are written to <out>/{clean,faulty,cinc2017}/:
  clean     PTB-XL test fold
  faulty    the same ECGs, with a known fault injected into a share of them
  cinc2017  PhysioNet/CinC 2017 AliveCor ECGs (a different device: the shift test)
Thresholds are calibrated once on PTB-XL validation and saved in <out>/thresholds.json.

Usage:
  python 03_predict_reliability.py --data work/ --models work/models --out results/ --config ../config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from common import LABEL_NAMES, age_band, inject_fault, load_config, model_input, signal_quality
from model import ECGNet, run


class Ensemble:
    def __init__(self, model_dir, cfg, dev):
        self.cfg, self.dev, self.nets = cfg, dev, []
        for p in sorted(Path(model_dir).glob("member_*.pt")):
            net = ECGNet(cfg["model"]["dropout"]).to(dev)
            net.load_state_dict(torch.load(p, map_location=dev))
            self.nets.append(net)
        if not self.nets:
            raise SystemExit(f"no member_*.pt in {model_dir}")

    def __call__(self, Xraw):
        Xin = model_input(Xraw, self.cfg["fs"])
        outs = [run(n, Xin, self.dev) for n in self.nets]
        P = np.stack([o[0] for o in outs])
        # Device-shift score uses member 0's embedding (one consistent feature space).
        return P.mean(0), P.std(0) / 0.5, outs[0][1]


def safeguard_scale(row, safeguards):
    s = 1.0
    for g in safeguards or []:
        if g.get("approved") and str(row.get(g["field"])) == str(g["value"]):
            s = min(s, float(g.get("uncertainty_scale", 1.0)))
    return s


def decide(q_flags, unc, shift, unc_thr, shift_thr):
    flags = list(q_flags)
    if unc > unc_thr:
        flags.append("high_uncertainty")
    if shift > shift_thr:
        flags.append("device_shift")
    if q_flags:
        return flags, "repeat"
    return flags, ("escalate" if flags else "accept")


def assess(name, X, meta, faults, ens, maha, thr, cfg, out):
    prob, unc, emb = ens(X)
    shift = maha(emb)
    hr_range = cfg["quality"]["hr_range_bpm"]
    (out / name / "exams").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, m in enumerate(meta.to_dict("records")):
        q = signal_quality(X[i], cfg["fs"], hr_range)
        q_flags = []
        if q["sqi"] < cfg["quality"]["min_sqi"]:
            q_flags.append("low_quality")
        if q["heart_rate_bpm"] is None or not hr_range[0] <= q["heart_rate_bpm"] <= hr_range[1]:
            q_flags.append("implausible_heart_rate")
        m["age_band"] = age_band(m.get("age"))
        scale = safeguard_scale(m, cfg.get("safeguards"))
        flags, action = decide(q_flags, unc[i], shift[i], thr["uncertainty"] * scale, thr["shift"])
        _, action_base = decide(q_flags, unc[i], shift[i], thr["uncertainty"], thr["shift"])
        if scale < 1 and action != action_base:
            flags.append("safeguard")
        label = int(m["label"]) if pd.notna(m["label"]) else -1
        p = float(prob[i])
        rec = {
            "exam_id": m["exam_id"], "modality": "ecg_single_lead",
            "prediction": {"label": "abnormal" if p >= cfg["abnormal_prob"] else "normal", "abnormal_prob": round(p, 4)},
            "probability": round(p, 4),
            "uncertainty": round(float(unc[i]), 4),
            "quality": q,
            "shift_score": round(float(shift[i]), 3),
            "metadata": {"dataset": m["dataset"], "device": m.get("device"), "site": m.get("site"),
                         "sex": m.get("sex"), "age": None if pd.isna(m.get("age")) else float(m["age"]),
                         "age_band": m["age_band"], "lead": cfg["lead"], "fs": cfg["fs"],
                         "seconds": cfg["seconds"], "model": f"{len(ens.nets)}x 1D-ResNet ensemble"},
            "thresholds": {"uncertainty": round(thr["uncertainty"] * scale, 4), "shift": round(thr["shift"], 3),
                           "min_sqi": cfg["quality"]["min_sqi"]},
            "flags": flags, "suggested_action": action,
            # For evaluation only; the agent must not use this to decide.
            "_eval": {"label": LABEL_NAMES[label], "fault": faults[i], "cinc_class": m.get("cinc_class")},
        }
        json.dump(rec, open(out / name / "exams" / f"{m['exam_id']}.json", "w"))
        rows.append({**{k: m.get(k) for k in ("exam_id", "dataset", "fold", "device", "site", "sex", "age",
                                               "age_band", "cinc_class")},
                     "label": label, "fault": faults[i], "prob": p, "pred": int(p >= cfg["abnormal_prob"]),
                     "uncertainty": float(unc[i]), "sqi": q["sqi"], "heart_rate_bpm": q["heart_rate_bpm"],
                     "shift_score": float(shift[i]), "flags": ";".join(flags),
                     "suggested_action": action, "action_without_safeguards": action_base})
    df = pd.DataFrame(rows)
    df.to_csv(out / name / "exam_results.csv", index=False)
    print(f"\n{name}: {len(df)} ECGs -> {out / name}")
    print(df.suggested_action.value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ens = Ensemble(args.models, cfg, dev)

    data = Path(args.data)
    X = np.load(data / "ptbxl.npz")["X"]
    meta = pd.read_csv(data / "ptbxl_meta.csv")
    tr = meta.fold.isin(cfg["folds"]["train"]).values
    va = meta.fold.isin(cfg["folds"]["val"]).values
    te = meta.fold.isin(cfg["folds"]["test"]).values

    # Device-shift reference: PTB-XL training embeddings.
    _, _, emb_tr = ens(X[tr])
    lw = LedoitWolf().fit(emb_tr)
    mu, prec = lw.location_, lw.precision_

    def maha(E):
        d = E - mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0))

    thr_path = out / "thresholds.json"
    if thr_path.exists():
        thr = json.load(open(thr_path))
        print("using saved thresholds", thr)
    else:
        _, unc_va, emb_va = ens(X[va])
        u, s = cfg["uncertainty"], cfg["shift"]
        thr = {"uncertainty": float(np.quantile(unc_va, u["quantile"])) if u["threshold"] == "auto" else float(u["threshold"]),
               "shift": float(np.quantile(maha(emb_va), s["quantile"])) if s["threshold"] == "auto" else float(s["threshold"]),
               "calibrated_on": "PTB-XL validation fold", "n_val": int(va.sum())}
        json.dump(thr, open(thr_path, "w"), indent=2)
        print("calibrated thresholds", thr)

    Xte, mte = X[te], meta[te].reset_index(drop=True)
    assess("clean", Xte, mte, ["none"] * len(mte), ens, maha, thr, cfg, out)

    rng = np.random.default_rng(args.seed)
    Xf, faults = Xte.copy(), ["none"] * len(mte)
    for i in rng.choice(len(mte), int(cfg["faults"]["fraction"] * len(mte)), replace=False):
        faults[i] = str(rng.choice(cfg["faults"]["types"]))
        Xf[i] = inject_fault(Xf[i], cfg["fs"], faults[i], rng)
    mf = mte.copy()
    mf["exam_id"] = mf.exam_id + "_f"
    assess("faulty", Xf, mf, faults, ens, maha, thr, cfg, out)

    Xc = np.load(data / "cinc2017.npz")["X"]
    mc = pd.read_csv(data / "cinc2017_meta.csv")
    fc = ["noisy_recording" if c == "~" else "cross_device" for c in mc.cinc_class]
    assess("cinc2017", Xc, mc, fc, ens, maha, thr, cfg, out)


if __name__ == "__main__":
    main()
