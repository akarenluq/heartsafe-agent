"""Small 1D ResNet for single-lead ECG. Returns (logit, embedding)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class ECGNet(nn.Module):
    def __init__(self, dropout=0.2, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(1, w, 15, 2, 7, bias=False), nn.BatchNorm1d(w), nn.ReLU(),
                                  nn.MaxPool1d(3, 2, 1))
        self.blocks = nn.Sequential(Block(w, w, 1), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w, 2))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * w, 1)

    def forward(self, x):
        emb = self.blocks(self.stem(x)).mean(-1)
        return self.fc(self.drop(emb)).squeeze(1), emb


@torch.no_grad()
def run(net, X, device, bs=1024):
    """X: (N, T) float32 model input -> (probabilities (N,), embeddings (N, D))."""
    net.eval()
    probs, embs = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs][:, None, :]).to(device)
        logit, emb = net(xb)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        embs.append(emb.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(embs)
