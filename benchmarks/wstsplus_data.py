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
    No fire is lost, since WSTS+ fires are long (min 9 days, median 21), but
    the
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
    empty windows, about 31% of their test set. Their own benchmark keeps
    them, so the default here is False: train and score on everything. Set it
    True only to match an AFSeq-style protocol, and say so.

FEATURES  Veg / Multi / All, as index lists over the 40 processed channels.
"""
from __future__ import annotations

import json
import os
# HDF5 file locking over NFS hangs or serialises the workers (each one opens
# the same files): 8 workers at 5% CPU, GPU at 0%, no bytes read, measured on
# an NFS-backed cluster. Local disks never showed it. Off here, in
# the loader, so every entry point (train.py, the experiments, the tests) gets it.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF

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


# The ORIGINAL WSTS protocol: 12 folds over the four years 2018-2021. Each
# fold holds one year out for test and one of the remaining three for val,
# training on the other two, 4 x 3 = 12. This is the table of the original
# WildfireSpreadTS paper, and the one supp. S3.1 reports UTAE-AR on. Folds are
# addressed as 100 + i so the 4-fold path stays byte-identical.
WSTS_YEARS = (2018, 2019, 2020, 2021)
FOLDS12 = [(te, va) for te in WSTS_YEARS for va in WSTS_YEARS if va != te]


def split12_years(fold_id: int):
    """(train, val, test) year lists for an original-WSTS 12-fold."""
    te, va = FOLDS12[fold_id]
    tr = [y for y in WSTS_YEARS if y not in (te, va)]
    return tr, [va], [te]


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
                 horizon: int = 5, crop: int = 128, is_train: bool | None = None,
                 require_fire_on_t: bool = False):
    """One part ("train"/"val"/"test") of a published 4-fold."""
    Base = _base_class()

    class WSTSPlusSeq(Base):
        """Their dataset, re-windowed into our seq2seq 5-tuple."""

        def __init__(self):
            tr_y, va_y, te_y = (split12_years(fold - 100) if fold >= 100
                                else split4_years(fold))
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

        # WINDOWED READ. The HDF5 datasets are contiguous, so the original
        # path reads the FULL rasters of every day of the window (~80 MB a
        # sample) to keep a 128-px crop (~15 MB). Choosing the crop BEFORE
        # reading needs only the fire channel (1/23 of the bytes): their
        # augment scores 10 candidate crops by fire content, and their eval
        # crop is a deterministic center crop. Every preprocessing step
        # downstream (flips/rotations with angle corrections, sin of degrees,
        # standardization by per-channel constants, binary mask, one-hot) is
        # per-pixel, so window-then-preprocess equals preprocess-then-window.
        # The np.random draws replicate FireSpreadDataset.augment exactly, in
        # the same order, so given the same seed state the produced samples
        # are BIT-IDENTICAL to the full-read path (verified by
        # test_windowed_equals_fullread.py). WSTS_FULLREAD=1 forces the
        # original path; rasters not larger than the crop also fall back.
        def _pick_window(self, raw, hh, ww):
            """(top, left, hflip, vflip, rotate), or None to fall back."""
            cs = self.crop_side_length
            if hh <= cs or ww <= cs:
                return None
            if not self.is_train:
                # TF.center_crop's rounding
                return (int(round((hh - cs) / 2.0)),
                        int(round((ww - cs) / 2.0)), False, False, 0)
            af = torch.Tensor(raw[:, RAW_AF])                  # (M, h, w)
            yb = (af[-1] > 0).long().float()
            best, bt, bl = None, 0, 0
            for _ in range(10):
                top = np.random.randint(0, hh - cs)
                left = np.random.randint(0, ww - cs)
                score = (af[:, top:top + cs, left:left + cs].mean()
                         + 1000 * yb[top:top + cs, left:left + cs].mean())
                if best is None or score > best:
                    best, bt, bl = score, top, left
            hf = bool(np.random.random() > 0.5)
            vf = bool(np.random.random() > 0.5)
            rot = int(np.floor(np.random.random() * 4))
            return bt, bl, hf, vf, rot

        def _preprocess_window(self, x, hf, vf, rot):
            """Their preprocess_and_augment, minus the crop (already applied),
            on an (M, 23, cs, cs) window. Same ops, same order."""
            x = torch.Tensor(x)
            deg = self.indices_of_degree_features
            if hf:
                x = TF.hflip(x)
                x[:, deg, ...] = 360 - x[:, deg, ...]
            if vf:
                x = TF.vflip(x)
                x[:, deg, ...] = (180 - x[:, deg, ...]) % 360
            if rot != 0:
                x = TF.rotate(x, rot * 90)
                x[:, deg, ...] = (x[:, deg, ...] - 90 * rot) % 360
            x[:, deg, ...] = torch.sin(torch.deg2rad(x[:, deg, ...]))
            binary_af_mask = (x[:, -1:, ...] > 0).float()
            x = self.standardize_features(x)
            x = torch.cat([x, binary_af_mask], axis=1)
            x = torch.nan_to_num(x, nan=0.0)
            new_shape = (x.shape[0], x.shape[2], x.shape[3],
                         self.one_hot_matrix.shape[0])
            landcover = self.one_hot_matrix[
                x[:, 16, ...].long().flatten() - 1
            ].reshape(new_shape).permute(0, 3, 1, 2)
            return torch.concatenate(
                [x[:, :16, ...], landcover, x[:, 17:, ...]], dim=1)

        def __getitem__(self, idx):
            year, fire, start, end = self.samples[idx]
            H = self.H
            path = self.imgs_per_fire[year][fire][0]
            windowed = not os.environ.get("WSTS_FULLREAD")
            with h5py.File(path, "r") as f:
                n = len(f["data"])
                stop = min(end + 1 + H, n)
                hh, ww = f["data"].shape[2], f["data"].shape[3]
                # One sequential read of the window's days. An h5py hyperslab
                # over the contiguous layout degenerates into ~30k scattered
                # reads (measured at 7.8 s/sample), so the crop is applied in
                # RAM instead: byte-identical, ~25x faster.
                raw = f["data"][start:stop]                  # (M, 23, h, w)
            win = self._pick_window(raw, hh, ww) if windowed else None
            if win is not None:
                top, left, hf, vf, rot = win
                cs = self.crop_side_length
                raw = raw[:, :, top:top + cs, left:left + cs]
            T = end - start + 1
            navail = raw.shape[0] - T                        # future days, >= 1

            if win is not None:
                proc = self._preprocess_window(raw, hf, vf, rot)
            else:
                # theirs: standardise, one-hot, sin of degrees, binary mask
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
