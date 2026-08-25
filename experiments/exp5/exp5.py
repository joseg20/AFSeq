#!/usr/bin/env python3
"""Experiment 5 -- UMAP of the encoder features, per data era.

    python experiments/exp5/exp5.py --ckpt <res18_unet.ckpt> --out fig6.png
    python experiments/exp5/exp5.py --ckpt ... --smoke     # 200 samples

Reproduces Figure 6 of the WSTS+ paper and extends it to AFSeq.

  panel (a)   their figure: WSTS+ samples encoded by the deepest layer of the
              best Res18-Unet encoder. BLUE = the original WSTS years
              (2018-2021), RED = the years WSTS+ added (2016, 2017, 2022, 2023).
  panel (b)   the same projection with AFSeq added in YELLOW.

WHAT THE FIGURE IS FOR. Their claim is that the added years are not more of the
same: if red sat inside blue, WSTS+ would be a bigger sample of one
distribution, and its extra difficulty would need another explanation. Panel (b)
asks the same question of AFSeq -- if yellow lands outside both, AFSeq is not a
third helping of the same fires either, which is the argument for having built
it. Whether that is what the plot shows is a question for the plot.

HOW THE FEATURES ARE TAKEN. `model.encoder(x)` returns one tensor per stage;
the deepest (`[-1]`, 512 channels for a ResNet-18) is spatially averaged to one
512-vector per sample. That is "the encoded features at the deepest layer" of
their caption. Single frame in, so a T=5 checkpoint is fed its last day.

TWO HONEST CAVEATS, since this figure is easy to over-read:

  - UMAP is not a metric space. Distance between clusters on the page is not
    distance between distributions; only the neighbour structure is meaningful.
    The random_state is pinned so the picture is at least reproducible.
  - Both panels share ONE fitted projection, computed on the union. Fitting
    each panel separately would produce two incomparable pictures and an
    apparent shift that is pure artefact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (ROOT, ROOT / "benchmarks", ROOT / "experiments" / "exp4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WSTS_ORIGINAL = (2018, 2019, 2020, 2021)      # the original WildfireSpreadTS
WSTS_ADDED = (2016, 2017, 2022, 2023)         # what WSTS+ added
COLORS = dict(original="#3b6fd4", added="#d43b3b", afseq="#e8c53a")
LABELS = dict(original="WSTS (2018-2021)", added="WSTS+ added years",
              afseq="AFSeq")


@torch.no_grad()
def encode(model, loader, dev, in_ch, n_veg, limit=0):
    """Deepest encoder stage, spatially averaged: one vector per sample."""
    feats, tags = [], []
    for n, (X, _FC, _Y, _V, meta) in enumerate(loader):
        if limit and n * X.shape[0] >= limit:
            break
        x = X.to(dev).float()
        # Feed the encoder what it was trained on: the whole history flattened
        # into the channel axis when conv1 expects T x C, the day-t frame when
        # it expects C. Getting this backwards is the silent-ImageNet failure.
        if in_ch == n_veg:
            inp = x[:, -1]
        else:
            inp = x[:, -(in_ch // n_veg):].flatten(1, 2)
        f = model.encoder(inp)[-1]            # (B, 512, h, w)
        feats.append(f.mean(dim=(2, 3)).cpu().numpy())
        tags.extend(meta)
    return np.concatenate(feats), tags


def era_of(tag):
    """Which era a sample belongs to, from its (source, fire, year) meta."""
    source = tag[0] if isinstance(tag, (tuple, list)) else "afseq"
    if source == "afseq":
        return "afseq"
    year = tag[2] if len(tag) > 2 else None
    return "added" if year in WSTS_ADDED else "original"


def infer_in_channels(state: dict) -> int:
    """How many channels the checkpoint's first conv expects.

    This is NOT cosmetic. The Res18-Unet of the cross-benchmark is trained with
    flatten_temporal_dimension=True over 5 days of Veg, so its conv1 takes
    7 x 5 = 35 channels, not 7. Building the model with the wrong count and
    loading with strict=False drops conv1 silently, leaving ImageNet weights in
    the very first layer -- the figure would then show an untrained encoder
    while claiming to show the best Res18-Unet.
    """
    for key in state:
        if key.endswith("encoder.conv1.weight"):
            return int(state[key].shape[1])
    raise SystemExit("no encuentro encoder.conv1.weight en el checkpoint")


def build_model(ckpt: Path, n_veg: int, dev: str):
    """Their SMPModel, with the encoder we are about to read features from.

    Returns (encoder-owning module, channels it expects).
    """
    sys.path.insert(0, str(ROOT / "benchmarks" / "wsts+"))
    from src.models.SMPModel import SMPModel

    if ckpt is None:
        print("SIN checkpoint: encoder ImageNet. Solo para probar el pipeline")
        in_ch = n_veg
        state = None
    else:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        in_ch = infer_in_channels(state)
        print(f"{ckpt.name}: conv1 espera {in_ch} canales "
              f"({in_ch // n_veg} dias x {n_veg} de Veg)")

    m = SMPModel(encoder_name="resnet18", encoder_weights="imagenet",
                 n_channels=in_ch, flatten_temporal_dimension=True,
                 pos_class_weight=236.0, loss_function="Focal")
    if state is not None:
        res = m.load_state_dict(state, strict=False)
        enc_missing = [k for k in res.missing_keys if "encoder" in k]
        if enc_missing:
            raise SystemExit(
                f"el checkpoint no cubre el encoder ({len(enc_missing)} claves "
                f"ausentes, p.ej. {enc_missing[0]}). La figura seria de un "
                f"encoder sin entrenar.")
        print(f"  encoder cargado entero ({len(res.unexpected_keys)} claves "
              f"sobrantes, normales: son del decoder)")
    return m.to(dev).eval().model, in_ch


def figure(emb, eras, out: Path, title_a, title_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eras = np.asarray(eras)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    panels = [("original", "added"), ("original", "added", "afseq")]
    for ax, keys, title in zip(axes, panels, (title_a, title_b)):
        for key in keys:
            m = eras == key
            if not m.any():
                continue
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.55,
                       c=COLORS[key], label=LABELS[key], linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=9, markerscale=2.2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, default=None,
                   help="checkpoint del mejor Res18-Unet. Sin el, encoder "
                        "ImageNet: sirve para probar el pipeline, no la figura")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="muestras por benchmark; 0 = todas")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out", type=Path, default=HERE / "fig6_umap.png")
    a = p.parse_args()
    if a.smoke:
        a.limit = a.limit or 200

    from fusion import N_VEG, T_HIST, collate, single
    from torch.utils.data import DataLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # single frame, T_HIST channels folded in: this is the data-level model
    model, in_ch = build_model(a.ckpt, N_VEG, dev)

    # ALL EIGHT YEARS, not one split. A WSTS+ fold holds out two years at a
    # time -- fold 0's test set is 2022+2023, both of them added years -- so
    # scoring one part would leave an entire era off the plot. The three parts
    # of a fold together are exactly the eight years, each appearing once.
    rng = np.random.default_rng(a.seed)
    feats, eras = [], []
    for bench, parts in (("wstsplus", ("train", "val", "test")),
                         ("afseq", ("test",))):
        got = 0
        # spread the budget over the parts so every year gets a share
        per_part = (a.limit // len(parts)) if a.limit else 0
        for part in parts:
            ds = single(bench, a.fold, part, horizon=1, crop=a.patch)
            room = per_part
            if room and room < len(ds):
                # A RANDOM subset, not the first N. These datasets are ordered
                # by year, so taking the head of one would return 2016 only and
                # silently drop an entire era from the figure.
                idx = rng.choice(len(ds), size=room, replace=False)
                ds = torch.utils.data.Subset(ds, sorted(idx.tolist()))
            dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.workers, collate_fn=collate)
            fv, tags = encode(model, dl, dev, in_ch, N_VEG, 0)
            feats.append(fv)
            eras.extend(era_of(t) for t in tags[:len(fv)])
            got += len(fv)
        print(f"{bench}: {got} muestras codificadas")
    X = np.concatenate(feats)
    import collections
    print("por era:", dict(collections.Counter(eras)))

    import umap
    # ONE projection for both panels: fitting them apart would make the shift
    # between panels an artefact of the fit rather than of the data
    emb = umap.UMAP(n_neighbors=25, min_dist=0.12, metric="euclidean",
                    random_state=a.seed).fit_transform(X)
    figure(emb, eras, a.out,
           "(a) WSTS+: anos originales frente a anadidos",
           "(b) los mismos, con AFSeq")


if __name__ == "__main__":
    main()
