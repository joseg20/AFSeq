#!/usr/bin/env python3
"""Fusing AFSeq into each published WSTS+ fold, in the shared Veg space.

    python experiments/exp4/fusion.py        # sizes of every fused split

WHAT "FUSED" MEANS HERE. A WSTS+ fold is three year groups: 4 train, 2 val, 2
test. AFSeq has its own train/val/test split of fires. The fused fold joins them
part by part:

    train(fold k) = WSTS+ train years of fold k  +  AFSeq Train fires
    val(fold k)   = WSTS+ val years of fold k    +  AFSeq Val fires
    test(fold k)  = WSTS+ test years of fold k   +  AFSeq Test fires

Joining part by part, rather than pooling and re-splitting, is what keeps the
fused rows comparable with the single-benchmark rows: a fire that was in AFSeq's
test set is still only ever in test, and a WSTS+ year held out in fold k is
still held out in fold k. Nothing a model sees in training reappears in test on
either side.

SCORED SEPARATELY, NEVER POOLED. Training is on the union; scoring is on each
benchmark's own test set, reported in its own column. Pooling the two would let
whichever side contributes more pixels dictate the number -- measured on the
with_mix12 checkpoints, micro pooling said one thing and macro the opposite,
because WSTS contributes the large majority of pixels.

THE DUPLICATE FIRES. Three fires exist in both benchmarks (fire_AZ3370210944920231026,
fire_CA3343811682320231030, fire_CA3396811772620231030). They are dropped from
the WSTS+ side so each physical fire enters the union exactly once and the AFSeq
split stays byte-identical to the one every other table uses.

BOTH SIDES ARE CAPPED AT 5 DAYS of history and give the model the same tensor
shape. If one benchmark handed longer sequences than the other, sequence length
would itself identify the source -- on top of the two constant Veg slots that
already do (see channels.py). Capping is the cheaper of the two evils.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (ROOT, ROOT / "benchmarks", str(HERE)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from channels import N_VEG, afseq_to_veg, wstsplus_to_veg     # noqa: E402
from paths import AFSEQ, AFSEQ_SPLITS, WSTS_HDF5  # noqa: E402

T_HIST = 5
H_DEFAULT = 5
PART = ("train", "val", "test")
# fires held by both benchmarks; dropped from the WSTS+ side (fire_overlap.json)
WSTS_DROP = ("fire_AZ3370210944920231026", "fire_CA3343811682320231030",
             "fire_CA3396811772620231030")
AFSEQ_STATS = HERE / "afseq_channel_stats.json"     # shipped with exp4


def _afseq_stats():
    """Per-channel mean/std of the AFSeq TRAIN split, for the z-scored slots."""
    import json
    d = json.loads(AFSEQ_STATS.read_text())
    return (np.asarray(d["mean"], dtype=np.float32),
            np.asarray(d["std"], dtype=np.float32))


class _WSTSVeg(Dataset):
    """One part of a WSTS+ fold, sliced to Veg. Their loader does the reading."""

    def __init__(self, fold: int, part: str, horizon=H_DEFAULT, crop=128,
                 drop_dupes=True):
        from wstsplus_data import make_dataset
        self.base = make_dataset(fold, part, str(WSTS_HDF5), features="Veg",
                                 history="fixed", n_history=T_HIST,
                                 horizon=horizon, crop=crop,
                                 require_fire_on_t=False)
        if drop_dupes:
            before = len(self.base.samples)
            self.base.samples = [s for s in self.base.samples
                                 if s[1] not in WSTS_DROP]
            self.dropped = before - len(self.base.samples)
        else:
            self.dropped = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        X, FC, Y, valid, meta = self.base[i]
        # already Veg-sliced by make_dataset(features="Veg"); FC lost its last
        # channel there, which is the one UTAE-AR re-supplies as its own mask
        return X, FC, Y, valid, ("wstsplus",) + tuple(meta[:2])


class _AFSeqVeg(Dataset):
    """One part of the AFSeq split, mapped into the Veg space."""

    def __init__(self, part: str, horizon=H_DEFAULT, crop=128, max_seq=16,
                 drop_refl=False):
        from dataset import afseq_loader
        from dataset.afseq_loader import Seq2SeqActivePadded
        from dataset.afseq_cache import install_if_configured
        install_if_configured(afseq_loader)    # only if AFSEQ_CACHE is set
        from afseq_datamodule import dl_ids, load_ids

        section = {"train": "Train", "val": "Val", "test": "Test"}[part]
        ids = dl_ids(load_ids(section, str(AFSEQ_SPLITS)), AFSEQ)
        self.base = Seq2SeqActivePadded(ids, str(AFSEQ), crop, max_seq, horizon,
                                        with_fire_hour=True)   # Veg slot 5
        self.mean, self.std = _afseq_stats()
        self.drop_refl = drop_refl

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        X, FC, Y, valid, meta = self.base[i]
        X = torch.from_numpy(np.asarray(X)).float()
        FC = torch.from_numpy(np.asarray(FC)).float()
        # Same 5 real days ending on day t as the WSTS+ side, repeating the
        # oldest frame when the fire is younger. Zero padding would seed the
        # rollout with a blank frame AND leave a length pattern that identifies
        # the benchmark -- exactly the confound this file is trying not to add.
        n = X.shape[0]
        X = X[[max(0, n - T_HIST + i) for i in range(T_HIST)]]
        Xv = afseq_to_veg(X, self.mean, self.std, self.drop_refl)
        if FC.numel():
            # fut_covars carry 16 channels (no fire); pad the fire slot so the
            # mapper sees the 17 it expects, then drop the fire slot again --
            # UTAE-AR supplies its own predicted mask there
            # fut_covars carry N_CHANNELS - 1 channels (no fire); pad the fire
            # and the detection-hour slots so the same mapping applies, then
            # zero the hour slot: the WSTS+ side sets its hour to 0 on the
            # forecast days too (no future fire leaks in)
            pad = torch.zeros(FC.shape[:-3] + (2,) + FC.shape[-2:], dtype=FC.dtype)
            FCv = afseq_to_veg(torch.cat([FC, pad], dim=-3), self.mean,
                               self.std, self.drop_refl)[..., :N_VEG - 1, :, :]
            FCv[..., N_VEG - 2, :, :] = 0.0
        else:
            FCv = FC
        return (Xv, FCv, torch.from_numpy(np.asarray(Y)).float(),
                torch.from_numpy(np.asarray(valid)).float(),
                ("afseq",) + tuple(meta[:2]))


def fused(fold: int, part: str, **kw):
    """WSTS+ fold `fold` joined with AFSeq, part by part, in Veg space."""
    return ConcatDataset([_WSTSVeg(fold, part, **kw), _AFSeqVeg(part, **kw)])


def single(benchmark: str, fold: int, part: str, **kw):
    """One benchmark alone, same space and same shapes as the fused set."""
    if benchmark == "wstsplus":
        return _WSTSVeg(fold, part, **kw)
    if benchmark == "afseq":
        return _AFSeqVeg(part, **kw)
    raise SystemExit(f"benchmark desconocido: {benchmark}")


def collate(batch, pad_value: float = 0.0):
    """Both sides are 5 days by construction, so this never actually pads."""
    seqs = [b[0] for b in batch]
    tmax = max(s.shape[0] for s in seqs)
    B = len(seqs)
    _, C, h, w = seqs[0].shape
    X = torch.full((B, tmax, C, h, w), pad_value, dtype=torch.float32)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s
    futs = [b[1] for b in batch]
    FC = (torch.stack(futs) if futs[0].numel()
          else torch.zeros(B, 0, C - 1, h, w))
    return (X, FC, torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]), [b[4] for b in batch])


if __name__ == "__main__":
    print(f"{'fold':>5}{'part':>7}{'WSTS+':>9}{'AFSeq':>9}{'fusion':>9}")
    for fold in range(4):
        for part in PART:
            w = _WSTSVeg(fold, part)
            a = _AFSeqVeg(part)
            print(f"{fold:>5}{part:>7}{len(w):>9,}{len(a):>9,}{len(w)+len(a):>9,}"
                  + (f"   (-{w.dropped} duplicados)" if w.dropped else ""))
