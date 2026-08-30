#!/usr/bin/env python3
"""S3.1 -- scoring de exp7: UTAE-AR en los 12 folds del WSTS original.

Test AP a t+1, pooled, crop 128 -- el mismo protocolo que exp2/exp4. Los
checkpoints de 5 pasos cargan en un modelo de 1 paso sin cambios (el primer
paso nunca toca fut_covars).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT/"models", ROOT/"training", ROOT/"benchmarks",
           ROOT/"experiments", ROOT/"experiments"/"exp4"):
    sys.path.insert(0, str(_p))
from paths import RUNS
from evaluation.metrics import pooled_ap_gpu
from train import build_model
from fusion import collate, single
from wstsplus_data import FOLDS12, split12_years

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=RUNS / "exp7")
p.add_argument("--workers", type=int, default=6)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--json", type=Path, default=None)
a = p.parse_args()
dev = "cuda"

rows = []
for i in range(len(FOLDS12)):
    d = a.out / f"f{i}"
    cks = sorted(d.glob("best_*.pth"))
    tr, va, te = split12_years(i)
    if not cks:
        rows.append((i, tr, va[0], te[0], None, None)); continue
    m = build_model("utae_ar", 1, in_ch=7)   # Veg
    m.load_state_dict(torch.load(cks[0], map_location="cpu"))
    m = m.to(dev).eval()
    ds = single("wstsplus", 100 + i, "test", horizon=1)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, pin_memory=True, collate_fn=collate)
    ys, ps, qs = [], [], []
    with torch.no_grad():
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
            qs.append((X[keep, -1, -1] > 0.5).float().flatten())   # persistencia
    y, pr, q = torch.cat(ys), torch.cat(ps), torch.cat(qs)
    ap = float(pooled_ap_gpu(y, pr, device=dev))
    per = float(pooled_ap_gpu(y, q, device=dev))
    rows.append((i, tr, va[0], te[0], ap, per))
    print(f"% f{i}: train {tr} val {va[0]} test {te[0]} -> AP {ap:.4f} "
          f"(persistence {per:.4f}, {y.numel():,} px)", file=sys.stderr)

got = [(r[4], r[5]) for r in rows if r[4] is not None]
print(r"\begin{tabular}{crrrrr}")
print(r"\toprule")
print(r"Fold & Train years & Val & Test & Persistence & UTAE-AR \\")
print(r"\midrule")
for i, tr, va, te, ap, per in rows:
    tr_s = ", ".join(str(y) for y in tr)
    ap_s = f"{ap:.3f}" if ap is not None else "--"
    pe_s = f"{per:.3f}" if per is not None else "--"
    print(f"{i} & {tr_s} & {va} & {te} & {pe_s} & {ap_s} \\\\")
if got:
    aps = np.array([g[0] for g in got]); pers = np.array([g[1] for g in got])
    print(r"\midrule")
    print(f"\\multicolumn{{4}}{{l}}{{Mean $\\pm$ std over {len(got)} folds}} & "
          f"{pers.mean():.3f} $\\pm$ {pers.std():.3f} & "
          f"{aps.mean():.3f} $\\pm$ {aps.std():.3f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
if a.json:
    a.json.write_text(json.dumps([{"fold": r[0], "train": r[1], "val": r[2],
                                   "test": r[3], "ap": r[4], "persistence": r[5]}
                                  for r in rows], indent=1))
