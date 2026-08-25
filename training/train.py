#!/usr/bin/env python3
"""Train any model of the architecture table on AFSeq. One entry point.

    python train.py --model utae_ar --loss bce_dice --lr 1e-3
    python train.py --model unet    --loss focal    --epochs 20

Every hyper-parameter is a flag, and the tag of the run records the ones that
were not the default, so two runs cannot silently overwrite each other.

TWO FAMILIES, ONE LOOP. Single-step models (unet, convnext, convlstm, utae)
predict day t+1 only, so they are trained with H=1 and scored on the first
horizon. UTAE-AR rolls five days forward and is trained with teacher forcing
that decays over the run. The difference is confined to `forward()` and to the
horizon; everything else -- data, loss, optimizer, selection -- is shared, which
is what makes the table a comparison of architectures rather than of recipes.

CHECKPOINT SELECTION is by validation AP under free running (teacher forcing
off), never by validation loss. Selecting on loss picks a model that is good at
the training regime, not at the one it will be evaluated in.

--------------------------------------------------------------------------
SPEED: what was done, and what was deliberately NOT done

Everything here is execution-level. None of it changes a number: same maths,
same order of operations, same result bit for bit.

  loss accumulated on the GPU   `loss.item()` inside the step loop forces a
                                device sync every iteration -- the GPU stops
                                and waits for the CPU thousands of times per
                                epoch. Accumulated in a device tensor and read
                                once per epoch instead.
  validation AP on the GPU      pooled AP sorted 80M values on the CPU, pinning
                                16 cores while the GPU idled at 4%. Same
                                definition, measured 97x faster (21.4s -> 0.2s).
  no per-sample .cpu()          validation used to copy every prediction to
                                host memory one sample at a time. Kept on
                                device and concatenated once.
  persistent_workers            data workers survive between epochs instead of
                                being respawned.
  non_blocking transfers        with pinned memory, the copy overlaps compute.
  zero_grad(set_to_none=True)   frees gradient buffers instead of zeroing them.

NOT done, on purpose -- each of these WOULD change results:

  mixed precision (AMP)         fp16/bf16 accumulation changes the numbers. It
                                is the single biggest speedup available and it
                                is worth adopting, but as a declared choice
                                with the tables re-measured, not silently.
  torch.compile                 kernel fusion changes floating-point order.
  cudnn.benchmark               picks algorithms by timing, so a rerun can pick
                                a different one and drift.
  channels_last                 different memory layout, different accumulation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from paths import (AFSEQ, AFSEQ_SPLITS,
                   RUNS, SEED)  # noqa: E402
from evaluation.metrics import pooled_ap_gpu                     # noqa: E402

SINGLE_STEP = {"unet", "convnext", "convlstm", "utae"}
STATIC = {"unet", "convnext"}            # consume one frame, not a sequence
INTERNAL_SIGMOID = {"unet", "convnext"}  # these already output probabilities
ALL_MODELS = SINGLE_STEP | {"utae_ar"}
LOSSES = ("focal", "bce_dice", "bce", "dice", "jaccard")


# --------------------------------------------------------------------- model
def build_model(name: str, horizon: int, in_ch: int = 17):
    """Instantiate with the paper's architecture hyper-parameters."""
    if name == "unet":
        from unet import UNet
        return UNet(in_channels=in_ch, out_channels=1)
    if name == "convnext":
        from convnext import ConvNeXtUNet
        return ConvNeXtUNet(backbone_name="convnext_large", in_chans=in_ch,
                            out_channels=1)
    if name == "convlstm":
        from convlstm import ConvLSTM_Seg
        return ConvLSTM_Seg(num_classes=1, input_size=(64, 64), input_dim=in_ch,
                            hidden_dim=64, kernel_size=(3, 3))
    if name == "utae":
        from utae import UTAE
        return UTAE(input_dim=in_ch, encoder_widths=[64, 64, 128],
                    decoder_widths=[32, 32, 128], out_conv=[32, 20])
    if name == "utae_ar":
        from utae_ar import UTAE_AR
        return UTAE_AR(horizon=horizon, input_dim=in_ch)
    raise SystemExit(f"modelo desconocido: {name}")


def last_real_frame(X):
    """The day-t frame of each sample in the batch.

    `collate_padded` pads short histories with zero frames at the END of the
    sequence, so `X[:, -1]` is the day-t frame only for the samples that happen
    to be the longest in their batch. Measured on splits_v4_s23, 68% of test
    samples get an all-zero image that way -- the static models were being
    scored on a blank input two times out of three.

    AFSeq histories are variable by construction (`seq_dates = dates[max(0,
    idx-15) : idx+1]`), so this is not an edge case: 80% of training samples
    have fewer than the 16-day maximum.
    """
    # same criterion as with_vegfolds/cross_table.py:last_real_frame, which
    # evaluation already used -- only training was still reading X[:, -1]
    nz = (X.reshape(*X.shape[:2], -1).abs().sum(-1) > 0)       # (B,T)
    last = (nz.shape[1] - 1 - nz.flip(1).float().argmax(1)).clamp(min=0)
    return X[torch.arange(X.shape[0], device=X.device), last]


def forward(model, name, X, FC, Y, tf_ratio):
    """One forward pass, hiding four different signatures behind one call."""
    if name in STATIC:
        return model(last_real_frame(X))             # day t, skipping padding
    if name == "convlstm":
        return model(X)
    if name == "utae":
        # plain U-TAE takes the sequence plus a position index per frame
        b, t = X.shape[:2]
        bp = torch.arange(t, device=X.device).unsqueeze(0).repeat(b, 1)
        return model(X, batch_positions=bp)
    return model(X, FC, y_true=Y, teacher_forcing_ratio=tf_ratio)


def to_logits(out, name):
    """Normalise every model's output to (B, 1, H, W) LOGITS at t+1.

    U-Net and ConvNeXt apply a sigmoid inside, so their output is already a
    probability. Feeding that to a loss that expects logits would silently
    train a different objective, so it is inverted back here.
    """
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == 5:                               # (B, H, 1, h, w) seq2seq
        out = out[:, 0]
    if out.dim() == 3:
        out = out.unsqueeze(1)
    if out.dim() == 4 and out.shape[1] > 1:          # UTAE emits out_conv chans
        out = out[:, :1]
    if name in INTERNAL_SIGMOID:
        p = out.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))
    return out


# ---------------------------------------------------------------- validation
@torch.no_grad()
def val_ap(model, name, loader, dev, horizon, max_batches=0):
    """Pooled AP at t+1 on validation, free running. Everything stays on GPU."""
    model.eval()
    ys, ps = [], []
    for nb, (X, FC, Y, V, _) in enumerate(loader):
        if max_batches and nb >= max_batches:
            break
        X = X.to(dev, non_blocking=True)
        FC = FC.to(dev, non_blocking=True)
        Y = Y.to(dev, non_blocking=True)
        V = V.to(dev, non_blocking=True)
        if name in SINGLE_STEP:
            logit = to_logits(forward(model, name, X, FC, None, 0.0), name)[:, 0]
        else:
            logit = model(X, FC, y_true=None, teacher_forcing_ratio=0.0)[:, 0, 0]
        keep = V[:, 0] > 0.5
        if keep.any():
            ys.append((Y[keep, 0, 0] > 0.5).flatten())
            ps.append(torch.sigmoid(logit[keep]).flatten())
    model.train()
    if not ys:
        return 0.0
    return pooled_ap_gpu(torch.cat(ys), torch.cat(ps), device=dev)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(ALL_MODELS))
    ap.add_argument("--loss", default="bce_dice", choices=LOSSES,
                    help="bce_dice es la del paper; focal es la de WSTS+")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--opt", default="adam", choices=["adam", "adamw"])
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--max-history", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tf-mode", default="linear", choices=["linear", "ss", "none"],
                    help="teacher forcing: lineal 1.0->0.5, scheduled sampling, "
                         "o ninguno. Solo aplica a utae_ar")
    ap.add_argument("--k", type=float, default=3.0, help="k del scheduled sampling")
    ap.add_argument("--benchmark", default="afseq",
                    choices=["afseq", "wstsplus", "fused"],
                    help="which dataset to train on. wstsplus goes through "
                         "benchmarks/wstsplus_data.py, which wraps THEIR "
                         "loader; fused is the two joined per fold in the "
                         "shared Veg space, defined by experiments/exp4/")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3],
                    help="wstsplus only: which published 4-fold")
    ap.add_argument("--features", default="All", choices=["Veg", "Multi", "All"],
                    help="wstsplus only: feature set over the 40 channels")
    ap.add_argument("--history", default="fixed", choices=["fixed", "variable"],
                    help="wstsplus only: fixed = their protocol (n-history full "
                         "days, opening days of each fire dropped); variable = "
                         "every day with a future day, padded and masked")
    ap.add_argument("--n-history", type=int, default=5,
                    help="wstsplus only: days of history when history=fixed")
    ap.add_argument("--fusion-source", default="both",
                    choices=["both", "wstsplus", "afseq"],
                    help="fused only: which side to TRAIN on. 'both' is the "
                         "fusion; the other two train on one benchmark in the "
                         "same Veg space, which is what makes the rows of the "
                         "cross-benchmark table comparable")
    ap.add_argument("--require-fire-on-t", action="store_true",
                    help="wstsplus only: keep only windows with active fire on "
                         "day t, as AFSeq's sampler does. OFF by default: their "
                         "benchmark keeps ignition, extinction and empty "
                         "windows, and those are 57%% of the fold-0 test set")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0,
                    help="corta la epoca tras N pasos. Solo para comprobar que "
                         "un modelo arranca; 0 = epoca completa")
    ap.add_argument("--max-val-batches", type=int, default=0,
                    help="corta la validacion tras N lotes. Solo para smoke "
                         "tests: sin esto una prueba de 3 pasos tarda lo que "
                         "tarde la validacion entera. 0 = completa")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag-extra", default="")
    args = ap.parse_args()

    out = args.out or (RUNS / "release_training")
    out.mkdir(parents=True, exist_ok=True)
    H = 1 if args.model in SINGLE_STEP else args.horizon

    lr_tag = f"{args.lr:g}".replace("-", "").replace(".", "p")
    tag = f"{args.model}_{args.loss}_lr{lr_tag}"
    if args.benchmark != "afseq":
        if args.benchmark == "fused":
            tag = (f"{args.model}_fused_{args.fusion_source}_f{args.fold}"
                   f"_{args.loss}_lr{lr_tag}")
        else:
            tag = (f"{args.model}_{args.benchmark}_{args.features}_f{args.fold}"
                   f"_{args.history}_{args.loss}_lr{lr_tag}")
            if args.require_fire_on_t:
                tag += "_fireT"
    if args.tf_mode != "linear" and args.model not in SINGLE_STEP:
        tag += f"_tf{args.tf_mode}" + (f"k{args.k:g}" if args.tf_mode == "ss" else "")
    if args.seed != SEED:
        tag += f"_s{args.seed}"
    tag += args.tag_extra

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(out / f"{tag}.log", mode="w"),
                                  logging.StreamHandler()])
    # GDAL pipes a TIFF warning per file through the root logger: left alone it
    # wrote ~630 MB of identical lines per run and filled the disk quota
    for n in ("rasterio", "rasterio._env", "rasterio._io", "fiona"):
        logging.getLogger(n).setLevel(logging.ERROR)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- data ------------------------------------------------------------
    from dataset import afseq_loader
    from dataset.afseq_loader import (Seq2SeqActivePadded, collate_padded,
                                      dl_ids, load_ids)
    from dataset.afseq_cache import install_if_configured
    install_if_configured(afseq_loader)    # only if AFSEQ_CACHE is set
    import losses as L                     # training/losses.py, same dir

    if args.benchmark == "fused":
        # The fused benchmark is defined by the experiment that uses it, not by
        # the trainer: which fires join which fold is exp4's claim to make, so
        # the definition lives there and this only consumes it.
        sys.path.insert(0, str(ROOT / "experiments" / "exp4"))
        from fusion import N_VEG, collate, fused, single

        def split(name):
            part = {"Train": "train", "Val": "val", "Test": "test"}[name]
            if args.fusion_source == "both":
                return fused(args.fold, part, horizon=H, crop=args.patch)
            return single(args.fusion_source, args.fold, part,
                          horizon=H, crop=args.patch)

        coll = collate
        in_ch = N_VEG
    elif args.benchmark == "wstsplus":
        # Their data, their loader, our windowing -- see
        # benchmarks/wstsplus_data.py for the three axes it declares.
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from wstsplus_data import FEATURES, collate_wstsplus, make_dataset
        from paths import WSTS_HDF5

        def split(name):
            return make_dataset(
                args.fold, {"Train": "train", "Val": "val", "Test": "test"}[name],
                str(WSTS_HDF5), features=args.features, history=args.history,
                n_history=args.n_history, max_history=args.max_history,
                horizon=H, crop=args.patch,
                require_fire_on_t=args.require_fire_on_t)

        coll = collate_wstsplus
        keep = FEATURES[args.features]
        in_ch = 40 if keep is None else len(keep)
    else:
        def split(name):
            ids = dl_ids(load_ids(name, str(AFSEQ_SPLITS)), AFSEQ)
            return Seq2SeqActivePadded(ids, str(AFSEQ), args.patch,
                                       args.max_history, H)

        coll = lambda b: collate_padded(b, 0.0)                  # noqa: E731
        in_ch = 17
    common = dict(collate_fn=coll, pin_memory=True,
                  persistent_workers=args.workers > 0)
    tl = DataLoader(split("Train"), args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, **common)
    vl = DataLoader(split("Val"), args.batch_size, shuffle=False,
                    num_workers=max(2, args.workers // 2), **common)

    model = build_model(args.model, H, in_ch=in_ch).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    logging.info(f"{args.benchmark} {args.model} H={H} loss={args.loss} lr={args.lr:g} "
                 f"| {n_par:,} parametros | {dev}")
    if args.benchmark == "wstsplus":
        src = f"WSTS+ f{args.fold} {args.features} ({args.history})"
    elif args.benchmark == "fused":
        src = f"Veg f{args.fold} train-on={args.fusion_source}"
    else:
        src = str(AFSEQ)
    logging.info(f"  train={len(tl.dataset)} val={len(vl.dataset)} "
                 f"| in_ch={in_ch} | datos: {src}")

    Opt = torch.optim.AdamW if args.opt == "adamw" else torch.optim.Adam
    opt = Opt(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best, hist = -1.0, []
    best_path = out / f"best_{tag}.pth"

    for ep in range(args.epochs):
        if args.model in SINGLE_STEP or args.tf_mode == "none":
            tf = 0.0
        elif args.tf_mode == "ss":
            tf = float(args.k / (args.k + np.exp(ep / args.k)))
        else:
            tf = max(0.5, 1.0 - 0.5 * ep / max(1, args.epochs - 1))

        t0 = time.time()
        model.train()
        # accumulated on device: .item() per step would sync GPU and CPU on
        # every iteration
        run_loss = torch.zeros((), device=dev)
        nb = 0
        for step, (X, FC, Y, V, _) in enumerate(tl):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            X = X.to(dev, non_blocking=True)
            FC = FC.to(dev, non_blocking=True)
            Y = Y.to(dev, non_blocking=True)
            V = V.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            o = forward(model, args.model, X, FC, Y, tf)
            if args.model in SINGLE_STEP:
                loss = L.seq_loss(args.loss, to_logits(o, args.model).unsqueeze(1),
                                  Y[:, :1], V[:, :1], 1, pos_weight=args.pos_weight)
            else:
                loss = L.seq_loss(args.loss, o, Y, V, H,
                                  pos_weight=args.pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.detach()
            nb += 1

        train_loss = (run_loss / max(nb, 1)).item()              # one sync
        vap = val_ap(model, args.model, vl, dev, H, args.max_val_batches)
        sched.step(-vap)
        hist.append(dict(ep=ep + 1, train=train_loss, val_ap=vap, tf=tf))
        logging.info(f"Epoch {ep + 1}/{args.epochs} - Train {train_loss:.4f} | "
                     f"valAP {vap:.4f} | tf={tf:.2f} | {time.time() - t0:.0f}s")
        if vap > best:
            best = vap
            torch.save(model.state_dict(), best_path)
            logging.info(f"  best (valAP {vap:.4f}) -> {best_path.name}")

    (out / f"{tag}.json").write_text(json.dumps(
        dict(tag=tag, model=args.model, loss=args.loss, lr=args.lr,
             opt=args.opt, epochs=args.epochs, batch_size=args.batch_size,
             horizon=H, pos_weight=args.pos_weight, seed=args.seed,
             tf_mode=args.tf_mode, params=n_par, dataset=str(AFSEQ),
             # every knob an experiment varies has to land here: a run that
             # cannot say what it was is a run that has to be re-read off its
             # own filename, and tags are lossy on purpose
             k=args.k if args.tf_mode == "ss" else None,
             benchmark=args.benchmark,
             fold=args.fold if args.benchmark == "wstsplus" else None,
             features=args.features if args.benchmark == "wstsplus" else None,
             history_mode=args.history if args.benchmark == "wstsplus" else None,
             fusion_source=args.fusion_source if args.benchmark == "fused" else None,
             best_val_ap=best, history=hist), indent=1))
    logging.info(f"FIN best_val_ap={best:.4f} -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
