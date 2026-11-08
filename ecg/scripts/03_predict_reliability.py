"""Step 3: ensemble predictions + reliability checks -> shared HeartSafe JSON per ECG.

For every ECG:
  probability   ensemble mean, temperature-calibrated on PTB-XL validation
  uncertainty   entropy of that probability (0-1 bits); ensemble spread kept as a field
  quality       agent index (flatline, clipping) + template match, baseline power,
                QRS-band power, high-frequency power, robust heart rate
  shift_score   max of two distances from PTB-XL training ECGs, each divided by its threshold
                (>1 = shifted): model embedding, and recording characteristics
  flags + suggested_action:
    repeat    poor signal quality or implausible heart rate -> ask for a new recording
    escalate  high uncertainty or device shift -> clinician review
    accept    otherwise
Every threshold is calibrated once on CLEAN PTB-XL validation ECGs (fold 9), never on
faults or on the shifted device, and saved in <out>/thresholds.json.

Sets written to <out>/{clean,faulty,cinc2017}/: PTB-XL test; the same ECGs with a known
fault injected into a share of them; PhysioNet/CinC 2017 AliveCor ECGs (device shift).

Usage:
  python 03_predict_reliability.py --data work/ --models work/models --out results/ --config ../config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf

from common import (LABEL_NAMES, age_band, inject_fault, load_config, model_input, quality_components,
                    recording_features)
from model import ECGNet, run

QUALITY_FLAG = {"template_corr": "noisy_signal", "qrs_power": "noisy_signal", "hf_power": "noisy_signal",
                "baseline_power": "baseline_wander"}


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
        # Embedding shift uses member 0 (one consistent feature space).
        return np.stack([o[0] for o in outs]), outs[0][1]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def entropy_bits(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def fit_temperature(p, y):
    z = logit(p)

    def nll(T):
        q = np.clip(1 / (1 + np.exp(-z / T)), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(minimize_scalar(nll, bounds=(0.25, 5), method="bounded").x)


class Mahalanobis:
    def __init__(self, E):
        self.mu_sd = (E.mean(0), E.std(0) + 1e-6)
        lw = LedoitWolf().fit((E - self.mu_sd[0]) / self.mu_sd[1])
        self.mu, self.prec = lw.location_, lw.precision_

    def __call__(self, E):
        d = (E - self.mu_sd[0]) / self.mu_sd[1] - self.mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec, d), 0))


def safeguard_scale(row, safeguards):
    s = 1.0
    for g in safeguards or []:
        if g.get("approved") and str(row.get(g["field"])) == str(g["value"]):
            s = min(s, float(g.get("uncertainty_scale", 1.0)))
    return s


class Reliability:
    def __init__(self, ens, cfg, maha_emb, maha_rec, thr):
        self.ens, self.cfg, self.maha_emb, self.maha_rec, self.thr = ens, cfg, maha_emb, maha_rec, thr

    def measure(self, X):
        P, emb = self.ens(X)
        p_raw = P.mean(0)
        p = 1 / (1 + np.exp(-logit(p_raw) / self.thr.get("temperature", 1.0)))
        unc = entropy_bits(p) if self.cfg["uncertainty"]["method"] == "entropy" else P.std(0) / 0.5
        fs = self.cfg["fs"]
        comps = [quality_components(x, fs) for x in X]
        rec = np.stack([recording_features(x, fs) for x in X])
        return {"p": p, "p_raw": p_raw, "unc": unc, "spread": P.std(0) / 0.5,
                "d_emb": self.maha_emb(emb), "d_rec": self.maha_rec(rec), "comps": comps}

    def quality_flags(self, c):
        q, flags = self.cfg["quality"], []
        if c["agent_sqi"] < q["min_sqi"]:
            flags.append("low_quality")
        for k, direction in q["checks"].items():
            t = self.thr["quality"][k]
            if (c[k] < t) if direction == "low" else (c[k] > t):
                flags.append(QUALITY_FLAG.get(k, "low_quality"))
        lo, hi = q["hr_range_bpm"]
        if not lo <= c["heart_rate_bpm"] <= hi:
            flags.append("implausible_heart_rate")
        return list(dict.fromkeys(flags))

    def decide(self, q_flags, unc, shift, scale):
        flags = list(q_flags)
        if unc > self.thr["uncertainty"] * scale:
            flags.append("high_uncertainty")
        if shift > 1:
            flags.append("device_shift")
        if q_flags:
            return flags, "repeat"
        return flags, ("escalate" if flags else "accept")


def assess(name, X, meta, faults, rel, out):
    cfg, thr = rel.cfg, rel.thr
    m_ = rel.measure(X)
    shift = np.maximum(m_["d_emb"] / thr["shift_embedding"], m_["d_rec"] / thr["shift_recording"])
    (out / name / "exams").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, m in enumerate(meta.to_dict("records")):
        c = m_["comps"][i]
        q_flags = rel.quality_flags(c)
        m["age_band"] = age_band(m.get("age"))
        scale = safeguard_scale(m, cfg.get("safeguards"))
        flags, action = rel.decide(q_flags, m_["unc"][i], shift[i], scale)
        _, action_base = rel.decide(q_flags, m_["unc"][i], shift[i], 1.0)
        if scale < 1 and action != action_base:
            flags.append("safeguard")
        label = int(m["label"]) if pd.notna(m["label"]) else -1
        p = float(m_["p"][i])
        sqi = c["agent_sqi"] if not q_flags else min(c["agent_sqi"], 0.5)  # agent reads quality.sqi < 0.6 as poor
        noise = m.get("lead1_noise")
        noise = "" if noise is None or (isinstance(noise, float) and np.isnan(noise)) else str(noise)
        rec = {
            "exam_id": m["exam_id"], "modality": "ecg_single_lead",
            "prediction": {"label": "abnormal" if p >= cfg["abnormal_prob"] else "normal", "abnormal_prob": round(p, 4)},
            "probability": round(p, 4),
            "uncertainty": round(float(m_["unc"][i]), 4),
            "quality": {"sqi": round(sqi, 3), **{k: c[k] for k in ("agent_sqi", "template_corr", "baseline_power",
                                                                    "qrs_power", "hf_power", "heart_rate_bpm")},
                        "issues": c["agent_issues"] + q_flags},
            "shift_score": round(float(shift[i]), 3),
            "metadata": {"dataset": m["dataset"], "device": m.get("device"), "site": m.get("site"),
                         "sex": m.get("sex"), "age": None if pd.isna(m.get("age")) else float(m["age"]),
                         "age_band": m["age_band"], "lead": cfg["lead"], "fs": cfg["fs"], "seconds": cfg["seconds"],
                         "model": f"{len(rel.ens.nets)}x 1D-ResNet ensemble",
                         "ensemble_spread": round(float(m_["spread"][i]), 4),
                         "shift_embedding": round(float(m_["d_emb"][i]), 2),
                         "shift_recording": round(float(m_["d_rec"][i]), 2)},
            "thresholds": {"uncertainty": round(thr["uncertainty"] * scale, 4), "shift_score": 1.0,
                           "min_sqi": cfg["quality"]["min_sqi"]},
            "flags": flags, "suggested_action": action,
            # For evaluation only; the agent must not use this to decide.
            "_eval": {"label": LABEL_NAMES[label], "fault": faults[i], "cinc_class": m.get("cinc_class"),
                      "annotated_noise": noise},
        }
        json.dump(rec, open(out / name / "exams" / f"{m['exam_id']}.json", "w"))
        rows.append({**{k: m.get(k) for k in ("exam_id", "dataset", "fold", "device", "site", "sex", "age",
                                               "age_band", "cinc_class")},
                     "lead1_noise": noise, "label": label, "fault": faults[i], "prob": p,
                     "prob_raw": float(m_["p_raw"][i]), "pred": int(p >= cfg["abnormal_prob"]),
                     "uncertainty": float(m_["unc"][i]), "ensemble_spread": float(m_["spread"][i]),
                     "sqi": sqi, **{k: c[k] for k in ("agent_sqi", "template_corr", "baseline_power", "qrs_power",
                                                      "hf_power", "heart_rate_bpm")},
                     "shift_embedding": float(m_["d_emb"][i]), "shift_recording": float(m_["d_rec"][i]),
                     "shift_score": float(shift[i]), "flags": ";".join(flags),
                     "suggested_action": action, "action_without_safeguards": action_base})
    df = pd.DataFrame(rows)
    df.to_csv(out / name / "exam_results.csv", index=False)
    print(f"\n{name}: {len(df)} ECGs -> {out / name}")
    print(df.suggested_action.value_counts().to_string())
    return df


def calibrate(ens, cfg, X_tr, X_va, y_va):
    """Every threshold from clean ECGs only: training (shift reference) and validation."""
    fs = cfg["fs"]
    _, emb_tr = ens(X_tr)
    maha_emb = Mahalanobis(emb_tr)
    maha_rec = Mahalanobis(np.stack([recording_features(x, fs) for x in X_tr]))
    P_va, emb_va = ens(X_va)
    T = fit_temperature(P_va.mean(0), y_va)
    p_va = 1 / (1 + np.exp(-logit(P_va.mean(0)) / T))
    unc_va = entropy_bits(p_va) if cfg["uncertainty"]["method"] == "entropy" else P_va.std(0) / 0.5
    comps = pd.DataFrame([quality_components(x, fs) for x in X_va])
    tq = cfg["quality"]["tail_quantile"]
    u, s = cfg["uncertainty"], cfg["shift"]
    thr = {
        "temperature": T,
        "uncertainty": float(np.quantile(unc_va, u["quantile"])) if u["threshold"] == "auto" else float(u["threshold"]),
        "shift_embedding": float(np.quantile(maha_emb(emb_va), s["quantile"])),
        "shift_recording": float(np.quantile(maha_rec(np.stack([recording_features(x, fs) for x in X_va])), s["quantile"])),
        "quality": {k: float(np.quantile(comps[k], tq if d == "low" else 1 - tq))
                    for k, d in cfg["quality"]["checks"].items()},
        "calibrated_on": "clean PTB-XL validation fold (shift reference: training folds)", "n_val": int(len(y_va)),
    }
    return maha_emb, maha_rec, thr


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

    maha_emb, maha_rec, thr = calibrate(ens, cfg, X[tr], X[va], meta.label.values[va])
    thr_path = out / "thresholds.json"
    if thr_path.exists():  # reuse saved values so clean/faulty/safeguard runs match
        saved = json.load(open(thr_path))
        if set(saved) >= set(thr):
            thr = saved
            print("using saved thresholds")
    json.dump(thr, open(thr_path, "w"), indent=2)
    print("thresholds", json.dumps(thr, indent=2))
    rel = Reliability(ens, cfg, maha_emb, maha_rec, thr)

    Xte, mte = X[te], meta[te].reset_index(drop=True)
    assess("clean", Xte, mte, ["none"] * len(mte), rel, out)

    rng = np.random.default_rng(args.seed)
    Xf, faults = Xte.copy(), ["none"] * len(mte)
    for i in rng.choice(len(mte), int(cfg["faults"]["fraction"] * len(mte)), replace=False):
        faults[i] = str(rng.choice(cfg["faults"]["types"]))
        Xf[i] = inject_fault(Xf[i], cfg["fs"], faults[i], rng)
    mf = mte.copy()
    mf["exam_id"] = mf.exam_id + "_f"
    assess("faulty", Xf, mf, faults, rel, out)

    Xc = np.load(data / "cinc2017.npz")["X"]
    mc = pd.read_csv(data / "cinc2017_meta.csv")
    fc = ["noisy_recording" if c == "~" else "cross_device" for c in mc.cinc_class]
    assess("cinc2017", Xc, mc, fc, rel, out)


if __name__ == "__main__":
    main()
