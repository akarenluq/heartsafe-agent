"""Step 5: name the rhythm - normal / atrial fibrillation (AF) / other abnormal.

The step 2 model only says normal or abnormal. This trains a separate 3-class ensemble
(same 1D ResNet, three outputs) and adds its answer to every ECG output written by step 3,
so the agent can tell new AF from the patient's known AF and treat new AF with symptoms
as needing immediate attention.

Labels
  PTB-XL      AFIB or AFLT rhythm statement -> af; binary label normal -> normal; else other
  CinC 2017   A -> af, N -> normal, O -> other, '~' (too noisy) left out
Training data: PTB-XL folds 1-8 plus 60% of CinC 2017 (a second device, AliveCor), stratified
by class with a fixed seed. Validation (best epoch, temperature): PTB-XL fold 9 plus 10% of
CinC. Held out: PTB-XL fold 10 (clean and faulty sets) and the remaining 30% of CinC.
The step 2 model never sees CinC, so its device-shift test is unchanged. Rhythm results on
CinC are reported on the held-out 30% only, and every output records its split.

Heart rate from the R-peaks (already in quality.heart_rate_bpm) is checked against the PTB-XL
sinus bradycardia / tachycardia statements when --ptbxl is given.

Writes: prediction.rhythm in each <results>/{clean,faulty,cinc2017}/exams/*.json, rhythm columns
in each exam_results.csv, <results>/rhythm_evaluation.json, models in <models>/rhythm_member_*.pt.

Usage:
  python 05_rhythm.py --data work/ --models work/models --results results/ --config ../config.yaml \
      [--ptbxl /path/ptbxl_unzipped]
"""
import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from common import inject_fault, load_config, model_input
from model import ECGNet

CLASSES = ["normal", "af", "other"]
AF_CODES = ("AFIB", "AFLT")


def ptbxl_rhythm(meta):
    r = meta.rhythm.fillna("").astype(str)
    af = r.apply(lambda s: any(c in s.split(";") for c in AF_CODES))
    return np.where(af, 1, np.where(meta.label == 0, 0, 2))


def cinc_rhythm(meta):
    return meta.cinc_class.map({"N": 0, "A": 1, "O": 2}).fillna(-1).astype(int).values


def cinc_split(y, seed, shares=(0.6, 0.1)):
    """train / val / test per class, fixed seed (CinC 2017 has no patient IDs)."""
    rng = np.random.default_rng(seed)
    split = np.array(["none"] * len(y), dtype=object)
    for c in (0, 1, 2):
        idx = rng.permutation(np.where(y == c)[0])
        a, b = int(shares[0] * len(idx)), int((shares[0] + shares[1]) * len(idx))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"
    return split


@torch.no_grad()
def logits(net, X, dev, bs=1024):
    net.eval()
    out = [net(torch.from_numpy(X[i:i + bs][:, None, :]).to(dev))[0].float().cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_auc(y, P):
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in range(3) if 0 < (y == c).sum() < len(y)]))


def fit_temperature(Z, y):
    def nll(T):
        P = softmax(Z / T)
        return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-9))
    return float(minimize_scalar(nll, bounds=(0.25, 5), method="bounded").x)


def train(X, y, Xva, yva, mc, members, epochs, out, dev):
    counts = np.bincount(y, minlength=3)
    w = torch.tensor((counts.sum() / (3 * counts)) ** 0.5, dtype=torch.float32, device=dev)  # softened class weights
    ce = nn.CrossEntropyLoss(weight=w)
    Xt = torch.from_numpy(X[:, None, :]).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    log_path = out / "rhythm_train_log.json"
    log = json.load(open(log_path)) if log_path.exists() else {}
    bs = mc["batch_size"]
    for m in range(members):
        path = out / f"rhythm_member_{m}.pt"
        if path.exists():
            print(f"rhythm member {m}: already trained, skipping")
            continue
        torch.manual_seed(100 + m)
        net = ECGNet(mc["dropout"], n_out=3).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=mc["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best, best_ep, t0 = -1.0, -1, time.time()
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(yt), device=dev)
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                xb = Xt[idx] * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, device=dev))
                xb = torch.roll(xb, int(torch.randint(0, xb.shape[-1], (1,))), -1)
                loss = ce(net(xb)[0], yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            auc = macro_auc(yva, softmax(logits(net, Xva, dev)))
            if auc > best:
                best, best_ep = auc, ep
                torch.save(net.state_dict(), path)
            print(f"rhythm member {m} epoch {ep:2d} val macro AUROC {auc:.4f}")
        log[f"member_{m}"] = {"best_val_macro_auroc": round(best, 4), "best_epoch": best_ep,
                              "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(log, open(log_path, "w"), indent=2)


class RhythmEnsemble:
    def __init__(self, model_dir, cfg, dev):
        self.cfg, self.dev, self.nets, self.T = cfg, dev, [], 1.0
        for p in sorted(Path(model_dir).glob("rhythm_member_*.pt")):
            net = ECGNet(cfg["model"]["dropout"], n_out=3).to(dev)
            net.load_state_dict(torch.load(p, map_location=dev))
            self.nets.append(net)

    def logits(self, Xin):
        return np.mean([logits(n, Xin, self.dev) for n in self.nets], 0)

    def __call__(self, Xin):
        return softmax(self.logits(Xin) / self.T)


def scores(y, P, mask=None):
    """Per-class sensitivity / specificity, AF PPV and F1, and the confusion matrix."""
    if mask is not None:
        y, P = y[mask], P[mask]
    keep = y >= 0
    y, P = y[keep], P[keep]
    if not len(y):
        return {"n": 0}
    pred = P.argmax(1)
    out = {"n": int(len(y)), "counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
           "accuracy": round(float((pred == y).mean()), 4),
           "confusion_true_by_pred": [[int(((y == i) & (pred == j)).sum()) for j in range(3)] for i in range(3)]}
    for i, c in enumerate(CLASSES):
        t, p = y == i, pred == i
        out[c] = {"sensitivity": round(float((t & p).sum() / max(t.sum(), 1)), 4),
                  "specificity": round(float((~t & ~p).sum() / max((~t).sum(), 1)), 4)}
    t, p = y == 1, pred == 1
    ppv = (t & p).sum() / max(p.sum(), 1)
    sens = out["af"]["sensitivity"]
    out["af"].update({"ppv": round(float(ppv), 4), "f1": round(float(2 * ppv * sens / max(ppv + sens, 1e-9)), 4)})
    if 0 < t.sum() < len(t):
        out["af"]["auroc"] = round(float(roc_auc_score(t, P[:, 1])), 4)
    return out


def write_outputs(name, ids, y, P, split, results, tag):
    """Add prediction.rhythm (and _eval.rhythm) to step 3's JSON outputs and CSV."""
    d = results / name
    if not d.exists():
        print(f"{name}: no step 3 outputs, skipped")
        return None
    pred = P.argmax(1)
    for i, eid in enumerate(ids):
        f = d / "exams" / f"{eid}.json"
        if not f.exists():
            continue
        o = json.load(open(f))
        o.setdefault("prediction", {})["rhythm"] = {
            "label": CLASSES[pred[i]], "probs": {c: round(float(P[i, j]), 4) for j, c in enumerate(CLASSES)},
            "model": tag}
        o.setdefault("_eval", {}).update({"rhythm": CLASSES[y[i]] if y[i] >= 0 else "unknown",
                                          "rhythm_split": split[i]})
        json.dump(o, open(f, "w"))
    csv = d / "exam_results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        add = pd.DataFrame({"exam_id": ids, "rhythm_true": [CLASSES[v] if v >= 0 else "unknown" for v in y],
                            "rhythm_pred": [CLASSES[v] for v in pred], "p_af": P[:, 1], "rhythm_split": split})
        df = df.drop(columns=[c for c in add.columns if c != "exam_id" and c in df], errors="ignore")
        df.merge(add, on="exam_id", how="left").to_csv(csv, index=False)
        return df.merge(add, on="exam_id", how="left")
    return None


def heart_rate_check(ptbxl_root, results):
    """Heart rate from the R-peaks vs PTB-XL sinus bradycardia (SBRAD) / tachycardia (STACH) statements."""
    db = next(Path(ptbxl_root).rglob("ptbxl_database.csv"), None)
    csv = results / "clean" / "exam_results.csv"
    if db is None or not csv.exists():
        return None
    codes = pd.read_csv(db, index_col="ecg_id").scp_codes.apply(lambda s: set(ast.literal_eval(s)))
    df = pd.read_csv(csv)
    df["ecg_id"] = df.exam_id.str.replace("ptbxl_", "").astype(int)
    df["codes"] = df.ecg_id.map(codes)
    acc = df[df.suggested_action == "accept"]
    out = {}
    for code, test in (("SBRAD", lambda h: h < 60), ("STACH", lambda h: h > 100)):
        has = acc.codes.apply(lambda c: code in c)
        hr = acc.heart_rate_bpm
        out[code] = {"n": int(has.sum()),
                     "sensitivity": round(float(test(hr[has]).mean()), 4) if has.any() else None,
                     "specificity": round(float((~test(hr[~has])).mean()), 4),
                     "median_hr_bpm": round(float(hr[has].median()), 1) if has.any() else None}
    out["note"] = "accepted clean PTB-XL test ECGs; heart rate from the R-peaks in step 3"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    ap.add_argument("--ptbxl", help="PTB-XL folder, for the heart-rate check (optional)")
    ap.add_argument("--seed", type=int, default=0, help="must match step 3 (faulty set)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rc = {"members": 3, "epochs": 30, **(cfg.get("rhythm") or {})}
    mc = {**cfg["model"], **{k: rc[k] for k in ("batch_size", "lr", "dropout") if k in rc}}
    data, models, results = Path(args.data), Path(args.models), Path(args.results)
    models.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = cfg["fs"]

    X = np.load(data / "ptbxl.npz")["X"]
    meta = pd.read_csv(data / "ptbxl_meta.csv")
    yp = ptbxl_rhythm(meta)
    Xc = np.load(data / "cinc2017.npz")["X"]
    mcm = pd.read_csv(data / "cinc2017_meta.csv")
    yc = cinc_rhythm(mcm)
    sc = cinc_split(yc, rc.get("cinc_split_seed", 0))
    Xp_in, Xc_in = model_input(X, fs), model_input(Xc, fs)

    tr = meta.fold.isin(cfg["folds"]["train"]).values
    va = meta.fold.isin(cfg["folds"]["val"]).values
    te = meta.fold.isin(cfg["folds"]["test"]).values
    Xtr = np.concatenate([Xp_in[tr], Xc_in[sc == "train"]])
    ytr = np.concatenate([yp[tr], yc[sc == "train"]])
    Xva = np.concatenate([Xp_in[va], Xc_in[sc == "val"]])
    yva = np.concatenate([yp[va], yc[sc == "val"]])
    print(f"train {len(ytr)} {np.bincount(ytr).tolist()}  val {len(yva)} {np.bincount(yva).tolist()}  "
          f"(normal, af, other)")

    train(Xtr, ytr, Xva, yva, mc, rc["members"], rc["epochs"], models, dev)
    ens = RhythmEnsemble(models, cfg, dev)
    ens.T = fit_temperature(ens.logits(Xva), yva)
    tag = f"{len(ens.nets)}x 1D-ResNet rhythm ensemble"

    # Held-out sets, in the same order as step 3's outputs
    mte, yte = meta[te].reset_index(drop=True), yp[te]
    Pte = ens(Xp_in[te])
    rng = np.random.default_rng(args.seed)  # the same faults as step 3
    Xf, faults = X[te].copy(), ["none"] * len(mte)
    for i in rng.choice(len(mte), int(cfg["faults"]["fraction"] * len(mte)), replace=False):
        faults[i] = str(rng.choice(cfg["faults"]["types"]))
        Xf[i] = inject_fault(Xf[i], fs, faults[i], rng)
    Pf = ens(model_input(Xf, fs))
    Pc = ens(Xc_in)

    ev = {"model": tag, "temperature": round(ens.T, 3), "classes": CLASSES,
          "training": "PTB-XL folds 1-8 + 60% of CinC 2017; validation PTB-XL fold 9 + 10% of CinC"}
    split_p = np.array(["test"] * len(mte), dtype=object)
    df = write_outputs("clean", mte.exam_id.values, yte, Pte, split_p, results, tag)
    ev["ptbxl_test"] = scores(yte, Pte)
    if df is not None:
        ev["ptbxl_test_accepted"] = scores(yte, Pte, (df.suggested_action == "accept").values)
    df = write_outputs("faulty", (mte.exam_id + "_f").values, yte, Pf, split_p, results, tag)
    if df is not None:
        acc = (df.suggested_action == "accept").values
        ev["ptbxl_faulty_accepted"] = scores(yte, Pf, acc)
        ev["ptbxl_faulty_not_accepted"] = scores(yte, Pf, ~acc)
    df = write_outputs("cinc2017", mcm.exam_id.values, yc, Pc, sc, results, tag)
    ev["cinc2017_test"] = scores(yc, Pc, sc == "test")
    if df is not None:
        ev["cinc2017_test_accepted"] = scores(yc, Pc, (sc == "test") & (df.suggested_action == "accept").values)
    if args.ptbxl:
        ev["heart_rate_from_r_peaks"] = heart_rate_check(args.ptbxl, results)
    json.dump(ev, open(results / "rhythm_evaluation.json", "w"), indent=2)
    print(json.dumps(ev, indent=2))


if __name__ == "__main__":
    main()
