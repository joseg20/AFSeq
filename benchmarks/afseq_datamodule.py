#!/usr/bin/env python3
"""Serve AFSeq to the WSTS+ models, without touching the WSTS+ repository.

WHY THIS FILE EXISTS
--------------------
The cross-benchmark table needs their models (UTAE(Res18) = SMPTempModel, and
Res18-Unet = SMPModel) trained on AFSeq, with THEIR configuration and THEIR
training code. Their repo is a benchmark artefact: it stays byte-identical, so
that "whose code is this" always has a one-word answer. Everything AFSeq needs
that their `FireSpreadDataModule` cannot provide lives here instead.

Their models expect, from `BaseModel.get_pred_and_gt`:

    batch = (x, y)
    x : (B, T, C, H, W) float, T == n_leading_observations, day t LAST
    y : (B, H, W)       float/long, the active-fire mask at day t+1

The AFSeq loader (`Seq2SeqActivePadded`) does not deliver that shape. The four
mismatches, and what this file does about each, are listed below. Every one of
them is a deliberate choice for the cross-benchmark experiment, not a bug fix.

--------------------------------------------------------------------------
CHANGE 1 -- variable-length histories, padded and masked   [temporal models]

    AFSeq emits a VARIABLE-length history: `seq_dates = dates[max(0, idx-15) :
    idx+1]`, so a fire observed on its second day yields T=2. Measured on
    splits_v4_s23: 80.3% of training samples and 75.5% of test samples have
    T < 16, and 36.1% of the test set has T < 5.

    WSTS never has this -- it serves exactly `n_leading_observations` days -- so
    their code has no notion of a padded frame.

    Here, sequences keep their real length and are padded with zero frames at
    the END of the batch's longest sequence, and the temporal attention is told
    which frames are padding. See afseq_models.SMPTempModelPadded: their LTAE2d
    and Temporal_Aggregator both accept a `pad_mask` already, their
    SMPTempModel just never passes one. This is the same treatment our earlier
    Res18 replica used, so the AFSeq rows stay comparable with what came before.

    NOT done: dropping short samples. WSTS does exactly that, but it would
    shrink the AFSeq test set from 1198 to 765 samples, and the Res18 rows would
    then be measured on a different population from every other row of our
    tables.

    NOT done either: a fixed T with the oldest frame replicated at the front.
    That invents observations. `n_leading_observations` is still available for
    SMPModel (`flatten_temporal_dimension=True` folds T into the channel axis,
    so it structurally cannot take a variable T), and only for it.

--------------------------------------------------------------------------
CHANGE 2 -- the feature sets are not the same 40 channels  [affects every model]

    Their `features_to_keep` indexes the 40-channel WSTS layout. AFSeq has 17
    channels with a different layout, so their "Veg" = [0,1,2,3,4,38,39] cannot
    be passed through as indices. The correspondence, channel by channel:

      WSTS Veg channel        AFSeq equivalent
      ---------------------   ---------------------------------------------
      0  VIIRS M11            MISSING -- AFSeq has 2 reflectance bands, not 3
      1  VIIRS I2             14 (reflectance band 0)
      2  VIIRS I1             15 (reflectance band 1)
      3  NDVI                 4
      4  EVI2                 5
      38 active fire, hour    MISSING -- AFSeq keeps no detection hour
      39 active fire, binary  16

    So the closest AFSeq analogue of Veg is 5 channels, not 7: two of theirs
    have no counterpart in AFSeq. FEATURE_SETS below names this "veg"; "all"
    passes the full 17 channels, which is the honest analogue of their "All".

--------------------------------------------------------------------------
CHANGE 3 -- normalisation is already applied               [affects every model]

    `FireSpreadDataModule` standardises with per-channel means/stds computed
    over the WSTS training years. AFSeq's `get_features` already returns
    normalised values, and imputes its holes with the channel mean (holes are
    NaN in the dataset, never a plausible constant). No second normalisation is
    applied here -- doing it would rescale already-scaled data.

--------------------------------------------------------------------------
CHANGE 4 -- pos_class_weight is computed, and then ignored  [Focal loss only]

    Their `train.py:before_instantiate_classes` derives pos_class_weight from
    the WSTS fold's missing-value rates. That path is WSTS-specific, so this
    file computes nothing equivalent: with `loss_function="Focal"`, the value
    never reaches the loss anyway. `BaseModel.__init__` normalises it into
    (0,1), the subclass' `save_hyperparameters()` immediately restores the raw
    value, and `compute_loss` passes `alpha = 1 - pos_class_weight`, which is
    negative -- and torchvision 0.15.1 (pinned in their requirements.txt)
    applies the class re-weighting only `if alpha >= 0`. Their published numbers
    are therefore focal loss with gamma=2 and NO class weighting. See the
    COMPAT-ALPHA note in wsts+_compat.py.
"""
from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

RELEASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RELEASE))
from paths import AFSEQ, AFSEQ_SPLITS  # noqa: E402


# AFSeq's 17 channels, from utils.get_features:
#   0,1,2    DEM        elevation, slope, aspect
#   3,4,5,6  vegetation landcover, NDVI, EVI, ET
#   7..13    climate    temp, dew, rh, wind speed, wind dir, pressure, precip
#   14,15    reflectance
#   16       active-fire mask of day t
FEATURE_SETS = {
    # closest analogue of their Veg -- see CHANGE 2: M11 and the detection hour
    # have no AFSeq counterpart, so this is 5 channels where theirs is 7
    "veg": [4, 5, 14, 15, 16],
    # the analogue of their "All"
    "all": list(range(17)),
}
AFSEQ_N_CHANNELS = 17
FIRE_CHANNEL = 16


def _load_afseq_dataset():
    """The AFSeq loader (dataset/afseq_loader.py), with the cache if set."""
    from dataset import afseq_loader
    from dataset.afseq_cache import install_if_configured
    install_if_configured(afseq_loader)    # only if AFSEQ_CACHE is set
    return afseq_loader.Seq2SeqActivePadded


# Verbatim from with_wstsplus/train_afseq_proto.py (load_ids, dl_ids), inlined
# so that importing this file does not pull in that module's sklearn/model
# imports -- and because packaging the split reading is a release TODO anyway.
def load_ids(section: str, splits_file) -> list:
    lines = pathlib.Path(splits_file).read_text().splitlines()
    s = [i for i, l in enumerate(lines) if l.startswith(section)][0] + 1
    out = []
    for l in lines[s:]:
        if l.strip() == "" or l.startswith(("Train", "Val", "Test")):
            break
        out.append(l.strip())
    return out


def dl_ids(ids: list, dataset) -> list:
    dataset = pathlib.Path(dataset)
    return [i for i in ids
            if (dataset / i).is_dir() and any((dataset / i).glob("*/active-fire.tif"))]


class _TheirShape(Dataset):
    """Wrap an AFSeq dataset so each item is the (x, y) their BaseModel wants."""

    def __init__(self, inner, n_leading, features: list[int]):
        self.inner = inner
        self.n_leading = n_leading      # None: keep the real length (CHANGE 1)
        self.features = features

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        X_hist, _fut, Y, valid, _meta = self.inner[idx]   # (T,17,h,w), (H,1,h,w)

        # CHANGE 1: keep the real length; only SMPModel needs a fixed T, and
        # then day t must end up last, since their code reads x[:, -1] as day t
        if self.n_leading is not None:
            T = X_hist.shape[0]
            if T >= self.n_leading:
                x = X_hist[T - self.n_leading:]
            else:
                pad = np.zeros((self.n_leading - T, *X_hist.shape[1:]),
                               dtype=X_hist.dtype)
                x = np.concatenate([pad, X_hist], axis=0)   # pad in FRONT
        else:
            x = X_hist

        # CHANGE 2: channel subset, by AFSeq index
        x = x[:, self.features]

        # long, not float: their FireSpreadDataModule yields `(y > 0).long()`,
        # and torchmetrics AveragePrecision rejects a float target outright.
        # compute_loss casts back with y.float(), so the loss is unaffected.
        y = (Y[0, 0] > 0.5).astype(np.int64)              # day t+1, binary
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


class AFSeqDataModule(pl.LightningDataModule):
    """AFSeq, shaped for the WSTS+ models. Drop-in for FireSpreadDataModule."""

    def __init__(self, n_leading_observations: int | None = None,
                 feature_set: str = "all",
                 batch_size: int = 64, num_workers: int = 4, patch: int = 64,
                 max_history: int = 16, splits_file: str | None = None):
        super().__init__()
        self.save_hyperparameters()
        if feature_set not in FEATURE_SETS:
            raise SystemExit(f"feature_set desconocido: {feature_set}")
        self.features = FEATURE_SETS[feature_set]
        self._sets = {}

    # number of input channels their model must be built with
    @property
    def n_channels_flat(self):
        """For SMPModel, which folds T into the channel axis, so it needs a T."""
        if self.hparams.n_leading_observations is None:
            raise SystemExit("SMPModel necesita --n-leading: no admite T variable")
        return len(self.features) * self.hparams.n_leading_observations

    @property
    def n_channels_per_step(self):
        """For SMPTempModel, which keeps the temporal axis."""
        return len(self.features)

    def setup(self, stage=None):
        Seq2SeqActivePadded = _load_afseq_dataset()
        splits = self.hparams.splits_file or str(AFSEQ_SPLITS)
        for name in ("Train", "Val", "Test"):
            if name in self._sets:
                continue
            ids = dl_ids(load_ids(name, splits), AFSEQ)
            inner = Seq2SeqActivePadded(
                ids, str(AFSEQ), self.hparams.patch, self.hparams.max_history,
                horizon=1,               # these models predict t+1 only
            )
            self._sets[name] = _TheirShape(
                inner, self.hparams.n_leading_observations, self.features)

    @staticmethod
    def _collate(batch):
        """Pad to the batch's longest sequence, zero frames at the END.

        Same convention as AFSeq's own collate_padded, so the pad mask computed
        inside SMPTempModelPadded matches what the loader produced.
        """
        xs = [b[0] for b in batch]
        ys = torch.stack([b[1] for b in batch])
        tmax = max(x.shape[0] for x in xs)
        if all(x.shape[0] == tmax for x in xs):
            return torch.stack(xs), ys
        out = torch.zeros((len(xs), tmax, *xs[0].shape[1:]), dtype=xs[0].dtype)
        for i, x in enumerate(xs):
            out[i, :x.shape[0]] = x
        return out, ys

    def _dl(self, name, shuffle):
        return DataLoader(
            self._sets[name], batch_size=self.hparams.batch_size,
            shuffle=shuffle, num_workers=self.hparams.num_workers,
            pin_memory=True, persistent_workers=self.hparams.num_workers > 0,
            drop_last=False, collate_fn=self._collate,
        )

    def train_dataloader(self):
        return self._dl("Train", True)

    def val_dataloader(self):
        return self._dl("Val", False)

    def test_dataloader(self):
        return self._dl("Test", False)
