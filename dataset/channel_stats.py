#!/usr/bin/env python3
"""Per-channel statistics of the TRAIN split, two files, one pass.

    python dataset/channel_stats.py [--days 3000] [--splits <file>]

  dataset/channel_means.json            NaN-aware mean of every channel BEFORE
                                        imputation: what impute_missing writes
                                        into a hole (paths.CHANNEL_MEANS).
  experiments/exp4/afseq_channel_stats.json
                                        mean and std of every channel AFTER
                                        normalisation and imputation: what
                                        exp4/channels.py z-scores with.

Both were once produced by one-off scripts in the research tree (steps12.py,
with_wstsplus/...); this is the same computation on the release loader, over
`--days` random training days (fire days of the Train split, whole rasters).
Regenerate whenever the channel layout or the dataset changes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import AFSEQ, AFSEQ_SPLITS  # noqa: E402
from dataset import afseq_loader as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=3000)
    ap.add_argument("--splits", default=str(AFSEQ_SPLITS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ids = L.dl_ids(L.load_ids("Train", a.splits), AFSEQ)
    days = [(f, p.parent.name) for f in ids
            for p in sorted((Path(AFSEQ) / f).glob("*/active-fire.tif"))]
    random.seed(a.seed)
    days = random.sample(days, min(a.days, len(days)))
    print(f"{len(ids)} fuegos de Train, {len(days)} dias muestreados")

    C = L.N_CHANNELS
    raw_sum = np.zeros(C); raw_n = np.zeros(C)          # before imputation
    s1 = np.zeros(C + 1); s2 = np.zeros(C + 1); n = 0   # after (+ detection hour)
    impute = L.impute_missing
    for i, (fid, day) in enumerate(days):
        # replicate get_features up to (not including) impute_missing
        L.impute_missing = lambda X: X
        X, _ = L.get_features(Path(AFSEQ) / fid, day)
        L.impute_missing = impute
        if X is None:
            continue
        X = X.astype(np.float64)
        ok = ~np.isnan(X)
        raw_sum += np.where(ok, X, 0).sum(axis=(1, 2)); raw_n += ok.sum(axis=(1, 2))
        Y = impute(X.astype(np.float32)).astype(np.float64)
        m, _ = L.load_raster(Path(AFSEQ) / fid / day / "active-fire.tif")
        hour = (m[1] if m.shape[0] > 1 else np.zeros_like(m[0])).astype(np.float64)
        Y = np.concatenate([Y, hour[None]])              # channel N_CHANNELS = hour
        s1 += Y.sum(axis=(1, 2)); s2 += (Y ** 2).sum(axis=(1, 2)); n += Y.shape[1] * Y.shape[2]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(days)}", flush=True)
    mean_raw = raw_sum / np.maximum(raw_n, 1)
    mean = s1 / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))

    out1 = ROOT / "dataset" / (f"channel_means_{a.tag}.json" if a.tag
                              else "channel_means.json")
    out1.write_text(json.dumps(dict(
        source=str(AFSEQ), split="train", n_days=len(days), seed=a.seed,
        note="per-channel mean over TRAIN, NaN ignored, BEFORE imputation; "
             "impute_missing writes these into holes",
        channels=L.CHANNELS, mean=mean_raw.tolist(),
        nan_fraction=(1 - raw_n / raw_n.max()).tolist()), indent=1))
    out2 = ROOT / "experiments" / "exp4" / "afseq_channel_stats.json"
    out2.write_text(json.dumps(dict(
        source=str(AFSEQ), n_samples=len(days), n_pixels=int(n),
        channels=L.CHANNELS + ["fire_hour"], mean=mean.tolist(), std=std.tolist()), indent=1))
    for k, name in enumerate(L.CHANNELS + ["fire_hour"]):
        extra = f"  mean_raw {mean_raw[k]:.4f}  nan {1 - raw_n[k] / raw_n.max():.3f}" if k < C else "  (detection hour, fusion only)"
        print(f"  {k:2d} {name:14s} mean {mean[k]:.4f}  std {std[k]:.4f}{extra}")
    print(f"-> {out1}\n-> {out2}")


if __name__ == "__main__":
    main()
