"""Train the WSTS+ models on AFSeq, with their configuration and their code.

    python train_afseq_theirs.py --model utae_res18 --features veg
    python train_afseq_theirs.py --model res18_unet --features veg --seed 7

The model, the loss, the optimiser, the training/validation/test steps and the
AP metric all come from the WSTS+ repository, unmodified: this file only builds
their LightningModule, hands it the AFSeq data module, and runs their Trainer.
It replaces their `src/train.py`, which cannot be reused as-is because it is
WSTS-specific in two places. `before_instantiate_classes` derives n_channels
and pos_class_weight from `FireSpreadDataset`/`FireSpreadDataModule`, and
`wandb_setup` assumes a wandb sweep agent created the run.

THEIR CONFIGURATION, and where each value comes from
----------------------------------------------------
Every sweep that produces a published number, the five cfgs/unet/saad/
table_5_*.yaml behind their Table 5, and wandb_all_sweep_jake_kube.yaml for
WSTS+, passes `-c cfgs/unet/res18_monotemporal.yaml`. So their configuration
is that file, and nothing else:

    encoder            resnet18, encoder_weights=imagenet
    loss               Focal, gamma=2   (see the alpha note below)
    optimiser          AdamW, lr 1e-3
    schedule           max_steps 10000
    batch size         64                 (cfgs/data_*_full_features.yaml)
    precision          32-true            (cfgs/trainer_single_gpu.yaml)
    selection          val_avg_precision, mode=max, save_top_k=1

    cfgs/unet/saad/res18_multitemporal.yaml says Dice and omits imagenet, and
    PROTOCOLO.md:311 followed it. It is an orphan: no sweep references it. The
    "Dice" default in load_trained_model.py is an argparse default for a script
    that only rebuilds a model to run a dummy forward pass. The loss is not
    part of a state_dict and that default says nothing about training. Their
    paper settles it: "we included both pre-training and focal loss in all our
    models", with an ablation (their Tab. 1) where removing focal loss costs
    a fraction of the AP it should.

    On alpha: their paper says alpha is the inverse frequency of positive
    pixels, but the code passes `1 - pos_class_weight`, which is negative, and
    torchvision 0.15.1 skips the class re-weighting entirely for alpha < 0.
    Their numbers are focal loss with NO class weighting. wsts+_compat.py
    reproduces that on a current torchvision, which raises instead of skipping.

WHAT IS NOT THEIRS: the data. See afseq_datamodule.py, which lists the four
changes AFSeq required and why each one was made.
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

HERE = Path(__file__).resolve().parent
THEIRS = HERE / "wsts+"
sys.path.insert(0, str(THEIRS))
sys.path.insert(0, str(HERE))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
torch.set_float32_matmul_precision("high")   # as their src/train.py does

from afseq_datamodule import AFSeqDataModule            # noqa: E402

# their configuration, read off cfgs/unet/res18_monotemporal.yaml
THEIR_CFG = dict(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    loss_function="Focal",
    pos_class_weight=236.0,   # never reaches the loss under Focal; see above
    lr=1e-3,
    max_steps=10000,
    batch_size=64,
)


def build(model_name: str, dm: AFSeqDataModule):
    """Their model classes, their hyper-parameters, AFSeq's channel count."""
    from src.models.SMPModel import SMPModel

    common = dict(
        encoder_name=THEIR_CFG["encoder_name"],
        encoder_weights=THEIR_CFG["encoder_weights"],
        pos_class_weight=THEIR_CFG["pos_class_weight"],
        loss_function=THEIR_CFG["loss_function"],
    )
    if model_name == "res18_unet":
        # data-level fusion: T folded into the channel axis
        return SMPModel(n_channels=dm.n_channels_flat,
                        flatten_temporal_dimension=True, **common)
    if model_name == "utae_res18":
        # feature-level fusion: LTAE over the temporal axis, T kept.
        # SMPTempModelPadded is their SMPTempModel plus the pad mask AFSeq's
        # variable-length fires require. See afseq_models.py.
        from afseq_models import SMPTempModelPadded
        return SMPTempModelPadded(n_channels=dm.n_channels_per_step,
                                  flatten_temporal_dimension=False, **common)
    raise SystemExit(f"modelo desconocido: {model_name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True,
                   choices=["utae_res18", "res18_unet"])
    p.add_argument("--features", default="all", choices=["veg", "all"])
    p.add_argument("--n-leading", type=int, default=None,
                   help="fixed T. Default None = AFSeq's real variable length, "
                        "masked in the temporal attention. SMPModel needs a "
                        "value here; utae_res18 does not.")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--max-history", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=THEIR_CFG["batch_size"])
    p.add_argument("--lr", type=float, default=THEIR_CFG["lr"])
    p.add_argument("--max-steps", type=int, default=THEIR_CFG["max_steps"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)   # their seed_everything: 0
    p.add_argument("--splits", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="3 batches per stage, to check it runs")
    a = p.parse_args()

    pl.seed_everything(a.seed, workers=True)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_{a.features}_T{a.n_leading or 'var'}_s{a.seed}"

    dm = AFSeqDataModule(n_leading_observations=a.n_leading,
                         feature_set=a.features, batch_size=a.batch_size,
                         num_workers=a.workers, patch=a.patch,
                         max_history=a.max_history, splits_file=a.splits)
    dm.setup()
    model = build(a.model, dm)

    # their optimiser, configured the way their cfg does it
    model.configure_optimizers = lambda: torch.optim.AdamW(
        model.parameters(), lr=a.lr)

    ckpt = ModelCheckpoint(dirpath=a.out / tag, monitor="val_avg_precision",
                           mode="max", save_top_k=1,
                           filename="best-{epoch}-{val_avg_precision:.4f}")
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="32-true",
        max_steps=3 if a.smoke else a.max_steps, max_epochs=-1,
        callbacks=[ckpt], logger=False, deterministic="warn",
        default_root_dir=str(a.out / tag),
        limit_train_batches=3 if a.smoke else 1.0,
        limit_val_batches=3 if a.smoke else 1.0,
        limit_test_batches=3 if a.smoke else 1.0,
        # once per epoch, which is their cfgs/trainer_single_gpu.yaml
        # (check_val_every_n_epoch: 1). AFSeq has 82 steps per epoch at
        # batch 64, so max_steps=10000 is ~122 epochs; validating four
        # times per epoch would run ~8.8k validation batches against
        # 10k training ones, nearly doubling the run for nothing.
        val_check_interval=1.0,
        num_sanity_val_steps=0,
    )

    n_par = sum(q.numel() for q in model.parameters())
    in_ch = (dm.n_channels_flat if a.model == "res18_unet"
             else dm.n_channels_per_step)
    print(f"[{tag}] {n_par/1e6:.2f}M params, in_channels={in_ch}, "
          f"train={len(dm._sets['Train'])} val={len(dm._sets['Val'])} "
          f"test={len(dm._sets['Test'])}")

    trainer.fit(model, dm)
    res = trainer.test(model, dm, ckpt_path="best")[0]

    out = dict(tag=tag, model=a.model, features=a.features,
               n_leading=a.n_leading, seed=a.seed, params=n_par,
               best_ckpt=ckpt.best_model_path,
               best_val_ap=float(ckpt.best_model_score or 0.0), **res)
    (a.out / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
