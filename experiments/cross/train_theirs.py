#!/usr/bin/env python3
"""Their models on the fused benchmark, with their configuration.

    python experiments/cross/train_theirs.py --model utae_res18 --fold 0
    python experiments/cross/train_theirs.py --model res50_unet --source wstsplus

Same split of responsibilities as benchmarks/train_afseq_theirs.py: the model,
the loss, the optimiser, the three Lightning steps and the AP metric all come
from benchmarks/wsts+ unmodified. This file only hands them data -- here, the
fused Veg-space dataset defined by cross/fusion.py -- and runs their Trainer.

THE MODELS, and where each configuration comes from:

  utae_res18   SMPTempModel: smp-Unet(resnet18, imagenet) + LTAE on the last
               encoder stage. Their best model overall (0.478 AP on WSTS).
  res50_unet   SMPModel with a resnet50 encoder, cfgs/unet/res50_monotemporal
               .yaml. Their second-best single-frame model (0.457 vs 0.455).

  Both take their configuration from cfgs/unet/res18_monotemporal.yaml, which
  is what EVERY published sweep of theirs passes -- focal loss, AdamW at 1e-3,
  10k steps, batch 64, imagenet weights. The res50 cfg differs only in the
  encoder name.

ON THEIR FOCAL LOSS: the class weighting never applies. compute_loss passes
alpha = 1 - pos_class_weight, which is negative because the subclass'
save_hyperparameters() undoes the normalisation in BaseModel.__init__, and
torchvision 0.15.1 (their pin) skips the re-weighting for alpha < 0. Their
published numbers are focal, gamma=2, no class weighting. wsts+_compat.py
reproduces exactly that on a current torchvision, which raises instead.

VEG SPACE, 7 channels. Both sides of the fusion are already in it, so the
models are built with n_channels=7 (or 35 flattened, for the single-frame one).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
THEIRS = ROOT / "benchmarks" / "wsts+"
for _p in (str(THEIRS), str(ROOT / "benchmarks"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
torch.set_float32_matmul_precision("high")

from fusion import N_VEG, T_HIST, fused, single                # noqa: E402

THEIR_CFG = dict(encoder_weights="imagenet", loss_function="Focal",
                 pos_class_weight=236.0, lr=1e-3, max_steps=10000,
                 batch_size=64)


class FusedDataModule(pl.LightningDataModule):
    """The fused Veg dataset, shaped as (x, y) for their BaseModel."""

    def __init__(self, fold: int, source: str, batch_size: int, workers: int,
                 patch: int = 128):
        super().__init__()
        self.save_hyperparameters()
        self._sets = {}

    def setup(self, stage=None):
        h = self.hparams
        for part in ("train", "val", "test"):
            if part in self._sets:
                continue
            ds = (fused(h.fold, part, horizon=1, crop=h.patch)
                  if h.source == "both"
                  else single(h.source, h.fold, part, horizon=1, crop=h.patch))
            self._sets[part] = _TheirShape(ds)

    def _dl(self, part, shuffle):
        return DataLoader(self._sets[part], batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=self.hparams.workers,
                          pin_memory=True, drop_last=False,
                          persistent_workers=self.hparams.workers > 0)

    def train_dataloader(self):
        return self._dl("train", True)

    def val_dataloader(self):
        return self._dl("val", False)

    def test_dataloader(self):
        return self._dl("test", False)


class _TheirShape(torch.utils.data.Dataset):
    """(X, FC, Y, valid, meta) -> (x, y), which is all their BaseModel reads."""

    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        X, _FC, Y, _valid, _meta = self.inner[i]
        # long, not float: their FireSpreadDataModule yields (y > 0).long() and
        # torchmetrics' AveragePrecision rejects a float target outright
        return X.float(), (Y[0, 0] > 0.5).long()


def build(model_name: str):
    from src.models.SMPModel import SMPModel
    common = dict(encoder_weights=THEIR_CFG["encoder_weights"],
                  pos_class_weight=THEIR_CFG["pos_class_weight"],
                  loss_function=THEIR_CFG["loss_function"])
    if model_name == "utae_res18":
        from afseq_models import SMPTempModelPadded
        return SMPTempModelPadded(encoder_name="resnet18", n_channels=N_VEG,
                                  flatten_temporal_dimension=False, **common)
    if model_name == "res50_unet":
        # single-frame, data-level fusion: T folded into the channel axis
        return SMPModel(encoder_name="resnet50", n_channels=N_VEG * T_HIST,
                        flatten_temporal_dimension=True, **common)
    raise SystemExit(f"modelo desconocido: {model_name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=["utae_res18", "res50_unet"])
    p.add_argument("--source", default="both",
                   choices=["both", "wstsplus", "afseq"])
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=THEIR_CFG["batch_size"])
    p.add_argument("--lr", type=float, default=THEIR_CFG["lr"])
    p.add_argument("--max-steps", type=int, default=THEIR_CFG["max_steps"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)      # their seed_everything
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    pl.seed_everything(a.seed, workers=True)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_fused_{a.source}_f{a.fold}"

    dm = FusedDataModule(a.fold, a.source, a.batch_size, a.workers, a.patch)
    dm.setup()
    model = build(a.model)
    model.configure_optimizers = lambda: torch.optim.AdamW(model.parameters(),
                                                           lr=a.lr)
    ckpt = ModelCheckpoint(dirpath=a.out / tag, monitor="val_avg_precision",
                           mode="max", save_top_k=1,
                           filename="best-{epoch}-{val_avg_precision:.4f}")
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="32-true",
        max_steps=3 if a.smoke else a.max_steps, max_epochs=-1,
        callbacks=[ckpt], logger=False, deterministic="warn",
        default_root_dir=str(a.out / tag),
        limit_train_batches=3 if a.smoke else 1.0,
        limit_val_batches=2 if a.smoke else 1.0,
        limit_test_batches=2 if a.smoke else 1.0,
        val_check_interval=1.0, num_sanity_val_steps=0)

    n_par = sum(q.numel() for q in model.parameters())
    print(f"[{tag}] {n_par/1e6:.2f}M params | train={len(dm._sets['train'])} "
          f"val={len(dm._sets['val'])} test={len(dm._sets['test'])}")

    trainer.fit(model, dm)
    res = trainer.test(model, dm, ckpt_path="best")[0]
    out = dict(tag=tag, model=a.model, source=a.source, fold=a.fold,
               params=n_par, best_ckpt=ckpt.best_model_path,
               best_val_ap=float(ckpt.best_model_score or 0.0), **res)
    (a.out / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
