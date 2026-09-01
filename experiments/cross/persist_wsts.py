"""Persistencia en WSTS+ por fold (t+1, AP pooled, protocolo de tab:transfer)."""
import json, statistics, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
for _p in (ROOT, ROOT/"benchmarks", ROOT/"experiments", HERE):
    sys.path.insert(0, str(_p))
from paths import RUNS
from evaluation.metrics import pooled_ap_gpu
from fusion import collate, single
dev = "cuda" if torch.cuda.is_available() else "cpu"
res = {}
for fold in range(4):
    dl = DataLoader(single("wstsplus", fold, "test", horizon=1), batch_size=8,
                    shuffle=False, num_workers=8, pin_memory=True,
                    collate_fn=collate)
    ys, ps = [], []
    with torch.no_grad():
        for X, FC, Y, V, _ in dl:
            X, Y, V = X.to(dev), Y.to(dev), V.to(dev)
            keep = V[:, 0] > 0.5
            if not keep.any(): continue
            ys.append((Y[keep, 0, 0] > 0.5).flatten())
            ps.append((X[keep, -1, 6] > 0.5).float().flatten())
    ap = pooled_ap_gpu(torch.cat(ys), torch.cat(ps), device=dev)
    res[f"f{fold}"] = dict(ap=ap, pixels=int(sum(y.numel() for y in ys)))
    print(f"f{fold}: AP={ap:.4f} ({res[f'f{fold}']['pixels']:,} px)")
aps = [r["ap"] for r in res.values()]
print(f"media {statistics.mean(aps):.4f} +- {statistics.stdev(aps):.4f}")
(RUNS/"cross"/"persist_wsts_t1.json").write_text(json.dumps(res, indent=1))
