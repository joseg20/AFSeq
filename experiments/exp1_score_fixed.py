#!/usr/bin/env python3
"""Test AP por fold de los anclas UTAE-AR en los tres conjuntos de features.

Los JSON de los checkpoints publicados solo guardan best_val_ap; la tabla
principal reporta AP de TEST. Esto lo mide con el protocolo de siempre
(pooled a t+1, crop 128, mascara de validez) para poder desglosar por fold.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT/"models", ROOT/"training", ROOT/"benchmarks", ROOT/"experiments"):
    sys.path.insert(0, str(p))
from evaluation.metrics import pooled_ap_gpu
from train import build_model
from wstsplus_data import FEATURES, make_dataset

ANCHOR = Path("/workspace1/users/josegui/data/wstsplus/runs_folds4")
HDF5 = "/workspace1/users/josegui/data/wstsplus/wsts8/hdf5"
dev = "cuda"


def collate(batch, pad=0.0):
    seqs = [b[0] for b in batch]
    tmax = max(s.shape[0] for s in seqs)
    B = len(seqs); _, C, h, w = seqs[0].shape
    X = torch.full((B, tmax, C, h, w), pad, dtype=torch.float32)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s
    F_ = torch.stack([b[1] for b in batch])
    Y = torch.stack([b[2] for b in batch])
    V = torch.stack([b[3] for b in batch])
    return X, F_, Y, V, [b[4] for b in batch]


@torch.no_grad()
def score(feat, fold, workers=6, bs=8):
    ck = ANCHOR / f"best_AR_{feat}_p4f{fold}_bce_dice_lr0p001.pth"
    if not ck.exists():
        return None
    n_ch = 40 if FEATURES[feat] is None else len(FEATURES[feat])
    m = build_model("utae_ar", 1, in_ch=n_ch)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    m = m.to(dev).eval()
    ds = make_dataset(fold, "test", HDF5, features=feat, history="fixed",
                      horizon=1, crop=128)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=True, collate_fn=collate)
    ys, ps = [], []
    for X, FC, Y, V, _ in dl:
        X, FC, Y, V = X.to(dev), FC.to(dev), Y.to(dev), V.to(dev)
        keep = V[:, 0] > 0.5
        if not keep.any():
            continue
        out = m(X, FC, y_true=None, teacher_forcing_ratio=0.0)
        if isinstance(out, tuple):
            out = out[0]
        ys.append((Y[keep, 0, 0] > 0.5).flatten())
        ps.append(torch.sigmoid(out[keep, 0, 0]).flatten())
    del m; torch.cuda.empty_cache()
    return float(pooled_ap_gpu(torch.cat(ys), torch.cat(ps), device=dev))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["Veg", "Multi", "All"])
    ap.add_argument("--out", type=Path,
                    default=Path("/workspace1/users/josegui/runs_v7/exp4/exp1_perfold_fixed.json"))
    a = ap.parse_args()
    res = {}
    for feat in a.features:
        vals = []
        for f in range(4):
            v = score(feat, f)
            vals.append(v)
            print(f"PCT {feat} f{f}: {v}", file=sys.stderr, flush=True)
        res[feat] = vals
        got = np.array([x for x in vals if x is not None])
        if len(got):
            print(f"PCT {feat} media {got.mean():.4f} +- {got.std(ddof=1):.4f}",
                  file=sys.stderr, flush=True)
    a.out.write_text(json.dumps(res, indent=1))
