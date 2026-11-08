"""Step 2: train a deep ensemble of small 1D ResNets on PTB-XL lead I (folds 1-8).

Each member uses a different seed; the best epoch by validation AUROC (fold 9) is kept.
Members already saved are skipped, so a timed-out job can simply be resubmitted.

Usage:
  python 02_train.py --data work/ --out work/models --config ../config.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from common import load_config, model_input
from model import ECGNet, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    ap.add_argument("--members", type=int)
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mc = cfg["model"]
    members = args.members or mc["members"]
    epochs = args.epochs or mc["epochs"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")

    X = np.load(Path(args.data) / "ptbxl.npz")["X"]
    meta = pd.read_csv(Path(args.data) / "ptbxl_meta.csv")
    Xin = model_input(X, cfg["fs"])
    tr = meta.fold.isin(cfg["folds"]["train"]).values
    va = meta.fold.isin(cfg["folds"]["val"]).values
    Xtr = torch.from_numpy(Xin[tr][:, None, :]).to(dev)
    ytr = torch.tensor(meta.label.values[tr], dtype=torch.float32, device=dev)
    Xva, yva = Xin[va], meta.label.values[va]
    print(f"train {len(ytr)} ({ytr.mean():.2f} abnormal), val {len(yva)} ({yva.mean():.2f} abnormal)")

    log_path = out / "train_log.json"
    log = json.load(open(log_path)) if log_path.exists() else {}
    bce = nn.BCEWithLogitsLoss()
    bs = mc["batch_size"]
    for m in range(members):
        path = out / f"member_{m}.pt"
        if path.exists():
            print(f"member {m}: already trained, skipping")
            continue
        torch.manual_seed(m)
        np.random.seed(m)
        net = ECGNet(mc["dropout"]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=mc["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best, best_ep, t0 = -1.0, -1, time.time()
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(ytr), device=dev)
            losses = []
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                xb = Xtr[idx] * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, device=dev))  # amplitude jitter
                xb = torch.roll(xb, int(torch.randint(0, xb.shape[-1], (1,))), -1)    # time shift
                loss = bce(net(xb)[0], ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()
            auc = roc_auc_score(yva, run(net, Xva, dev)[0])
            if auc > best:
                best, best_ep = auc, ep
                torch.save(net.state_dict(), path)
            print(f"member {m} epoch {ep:2d} loss {np.mean(losses):.4f} val AUROC {auc:.4f}")
        log[f"member_{m}"] = {"best_val_auroc": round(best, 4), "best_epoch": best_ep,
                              "epochs": epochs, "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(log, open(log_path, "w"), indent=2)
        print(f"member {m}: best val AUROC {best:.4f} at epoch {best_ep}")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
