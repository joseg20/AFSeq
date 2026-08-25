#!/usr/bin/env python3
"""WSTS+ served to our models, from the release, using THEIR loader.

The mirror image of afseq_datamodule.py: there, their model on our data; here,
our model on their data. Both sides of the cross-benchmark therefore go through
one declared adapter each, and their repository stays byte-identical.

Everything that reads pixels is theirs: FireSpreadDataset from benchmarks/wsts+
does the standardisation, the one-hot of land cover, the degree-channel sin,
the binary active-fire mask and the augmentation. This file only decides WHICH
windows become samples, and hands them over as (X_hist, FC, Y, valid, meta) --
the same 5-tuple the AFSeq loader emits, so training/train.py does not care
which benchmark it is looking at.

--------------------------------------------------------------------------
THREE AXES, all declared, because each one changes the number

HISTORY  "fixed" (default 5) or "variable"

    Their protocol is fixed: every sample carries exactly n_leading_observations
    days, and `datapoints_in_fire = n_imgs - n_leading_observations` silently
    drops the opening days of every fire. Measured on fold 0's training years,
    lead=5 keeps 10,408 of the 12,644 windows a lead of 1 would allow: 82.3%.
    No fire is lost -- WSTS+ fires are long (min 9 days, median 21) -- but the
    first five days of each one never appear.

    "variable" instead takes the window [max(0, i-max_history+1), i], so a fire
    contributes from its second day on, with however much history exists. Short
    histories are zero-padded at the END of the batch's longest sequence, and
    the models mask those frames (utae.masked_group_norm, ConvLSTM's frozen
    state, SMPTempModelPadded's pad_mask). Before those fixes this option would
    have quietly fed padded frames to the temporal attention, which is why it
    did not exist until now.

    "fixed" reproduces the published anchor; "variable" is a separate
    experiment and its numbers do not belong in the same column.

REQUIRE_FIRE_ON_T  False by default here

    AFSeq's sampler only emits windows with active fire on day t. Applied to
    WSTS that discards ignition (nothing at t, fire at t+1), extinction and
    empty windows -- about 31% of their test set. Their own benchmark keeps
    them, so the default here is False: train and score on everything. Set it
    True only to match an AFSeq-style protocol, and say so.

FEATURES  Veg / Multi / All, as index lists over the 40 processed channels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
THEIRS = HERE / "wsts+"
for _p in (str(THEIRS), str(THEIRS / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STATS_JSON = HERE / "stats_wsts8.json"

FIRE_BIN = 39      # binary active fire, last of the 40 processed channels
FIRE_RAW = 38      # active fire, detection hour
RAW_AF = 22        # active fire in the 23-channel HDF5

# Feature sets as index lists over the 40 processed channels
# (their cfgs/UTAE/wandb_cross_val.yaml). "All" is None = keep everything.
FEATURES = {
    "Veg": [0, 1, 2, 3, 4, 38, 39],
    "Multi": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
              + list(range(16, 33)) + [38, 39]),
    "All": None,
}

# The PUBLISHED WSTS+ protocol: four folds of two consecutive years; per fold,
# 4 years train, 2 val, 2 test, every 2-year block serving as test exactly once.
# Table 5 of their paper lives on this one, NOT on the 12-fold table that a
# deleted commit of their repo carried.
FOLDS4 = [
    (2016, 2017, 2020, 2021, 2018, 2019, 2022, 2023),
    (2018, 2019, 2022, 2023, 2020, 2021, 2016, 2017),
    (2016, 2017, 2020, 2021, 2022, 2023, 2018, 2019),
    (2018, 2019, 2022, 2023, 2016, 2017, 2020, 2021),
]


def split4_years(fold_id: int):
    """(train, val, test) year lists for a published WSTS+ 4-fold."""
    f = FOLDS4[fold_id]
    return list(f[:4]), list(f[4:6]), list(f[6:8])


def patch_stats() -> int:
    """Teach their stats lookup about the 4-year combos. Idempotent.

    `get_means_stds_missing_values` only knows the six 2-year combos of the
    original WSTS. The published 4-fold protocol trains on four years, so those
    keys do not exist; the values here were computed once by compute_stats.py.
    Anything it already knows falls through to the original, so the plain WSTS
    path is untouched.
    """
    from dataloader import utils as off_utils

    if getattr(off_utils, "_wsts8_patched", False):
        return 0
    raw = json.loads(STATS_JSON.read_text())
    combos = {}
    for key, v in raw["combos"].items():
        years = tuple(int(y) for y in key.split(","))
        combos[years] = (np.asarray(v["mean"], dtype=np.float32),
                         np.asarray(v["std"], dtype=np.float32))
    original = off_utils.get_means_stds_missing_values

    def patched(training_years):
        key = tuple(training_years)
        if key in combos:
            mean, std = combos[key]
            return mean.copy(), std.copy(), np.zeros(len(mean), dtype=np.float32)
        return original(training_years)

    off_utils.get_means_stds_missing_values = patched
    off_utils._wsts8_patched = True
    from dataloader import FireSpreadDataset as fsd_mod
    fsd_mod.get_means_stds_missing_values = patched
    return len(combos)


def _base_class():
    patch_stats()
    from dataloader.FireSpreadDataset import FireSpreadDataset
    return FireSpreadDataset


def make_dataset(fold: int, part: str, data_dir: str, features: str = "All",
                 history: str = "fixed", n_history: int = 5, max_history: int = 16,
                 horizon: int = 5, crop: int = 64, is_train: bool | None = None,
                 require_fire_on_t: bool = False):
    """One part ("train"/"val"/"test") of a published 4-fold."""
    Base = _base_class()

    class WSTSPlusSeq(Base):
        """Their dataset, re-windowed into our seq2seq 5-tuple."""

        def __init__(self):
            tr_y, va_y, te_y = split4_years(fold)
            years = {"train": tr_y, "val": va_y, "test": te_y}[part]
            train_flag = (part == "train") if is_train is None else is_train
            super().__init__(
                data_dir=data_dir, included_fire_years=years,
                n_leading_observations=1,      # windowing is done here, not there
                n_leading_observations_test_adjustment=None,
                crop_side_length=crop, load_from_hdf5=True, is_train=train_flag,
                remove_duplicate_features=False, stats_years=tuple(tr_y),
                features_to_keep=None, return_doy=False,
            )
            self.keep = FEATURES[features]
            self.history = history
            self.T = n_history
            self.max_history = max_history
            self.H = horizon
            self.require_fire_on_t = require_fire_on_t
            self._build()

        def _build(self):
            """Which windows become samples. `end` is day t, inclusive."""
            self.samples = []
            for year in self.imgs_per_fire:
                for fire, paths in self.imgs_per_fire[year].items():
                    if not paths:
                        continue
                    with h5py.File(paths[0], "r") as f:
                        n = len(f["data"])
                        has_fire = ((f["data"][:, RAW_AF] > 0)
                                    .reshape(n, -1).any(axis=1)
                                    if self.require_fire_on_t else None)
                    if self.history == "fixed":
                        # their protocol: T full days of history, so day t runs
                        # from T-1 to n-2 and the opening days are dropped
                        starts = [(s, s + self.T - 1) for s in range(0, n - self.T)]
                    else:
                        # every day with at least one future day, with whatever
                        # history exists behind it
                        starts = [(max(0, i - self.max_history + 1), i)
                                  for i in range(0, n - 1)]
                    for start, end in starts:
                        if self.require_fire_on_t and not has_fire[end]:
                            continue
                        self.samples.append((year, fire, start, end))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            year, fire, start, end = self.samples[idx]
            H = self.H
            path = self.imgs_per_fire[year][fire][0]
            with h5py.File(path, "r") as f:
                n = len(f["data"])
                stop = min(end + 1 + H, n)
                raw = f["data"][start:stop]                  # (M, 23, h, w)
            T = end - start + 1
            navail = raw.shape[0] - T                        # future days, >= 1

            # theirs: standardise, one-hot, sin of degrees, binary fire mask
            proc, _ = self.preprocess_and_augment(raw, raw[-1, RAW_AF])
            proc = proc.float()
            _, C, h, w = proc.shape

            X_hist = proc[:T]
            Y = torch.zeros(H, 1, h, w)
            valid = torch.zeros(H)
            FC = torch.zeros(max(H - 1, 0), C, h, w)
            for k in range(H):
                if k < navail:
                    Y[k, 0] = proc[T + k, FIRE_BIN]
                    valid[k] = 1.0
            for k in range(H - 1):
                if k < navail:
                    fr = proc[T + k].clone()
                    fr[FIRE_RAW] = 0.0        # no future fire leaks in
                    fr[FIRE_BIN] = 0.0
                    FC[k] = fr
            if self.keep is not None:
                X_hist = X_hist[:, self.keep]
                FC = FC[:, self.keep] if FC.numel() else FC
            # UTAE-AR appends its OWN predicted fire mask as the last channel
            # of each rollout frame (`cat([fut_covars[:, k], mask])`), so the
            # covariate frames must carry input_dim - 1 channels. Dropping the
            # last one is right for all three feature sets: Veg, Multi and All
            # every one of them ends in [..., 38, 39], so FIRE_BIN is last.
            if FC.numel():
                FC = FC[:, :-1]
            return X_hist, FC, Y, valid, (fire, year, (h, w))

    return WSTSPlusSeq()


def collate_wstsplus(batch, pad_value: float = 0.0):
    """Pad histories with zero frames at the END, as the AFSeq collate does.

    Same convention on both benchmarks, so the pad masks the models compute
    mean the same thing whichever loader produced the batch.
    """
    seqs = [b[0] for b in batch]
    Tmax = max(s.shape[0] for s in seqs)
    B = len(seqs)
    _, C, h, w = seqs[0].shape
    X = torch.full((B, Tmax, C, h, w), pad_value, dtype=torch.float32)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s
    futs = [b[1] for b in batch]
    FC = (torch.stack(futs) if futs[0].shape[0] > 0
          else torch.zeros(B, 0, C, h, w))
    return (X, FC, torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]), [b[4] for b in batch])
