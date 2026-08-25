#!/usr/bin/env python3
"""Experiment 2 -- the architecture table on AFSeq.

    python experiments/exp2.py --launch      # train the matrix
    python experiments/exp2.py --table       # score it on test

THE MATRIX: 6 trainable architectures x 3 splits = 18 runs, plus persistence,
which is measured rather than trained.

  Persistence   tomorrow's fire mask = today's. The floor of the table.
  U-Net         single frame in, one day out
  ConvNeXt      idem, convnext_large backbone (203M params -- verify against
                build_model, never against the constructor defaults, which
                would say 53M)
  ConvLSTM      sequence in, one day out
  UTAE          idem, temporal attention
  UTAE-Res18    THEIR model, imported from benchmarks/ and trained by
                benchmarks/train_afseq_theirs.py so it is unambiguous whose
                code produced the row. Deliberately not packaged into models/.
  UTAE-AR       sequence in, five days out

  splits        s5, s7, s23; mean +- std over the three.
  protocol      BCE + soft Dice, lr 1e-3, 20 epochs, checkpoint by validation
                AP. UTAE-Res18 is the exception: THEIR configuration (focal,
                AdamW, 10k steps), because the point of that row is their model
                as they trained it.

THE SCORING PROTOCOL, stated once and applied to every row
----------------------------------------------------------
  samples      the Test section of the split file, patch 64, max_history 16,
               horizon 1, require_fire_on_t=True (AFSeq's own sampler)
  scored       only samples whose t+1 target is valid (V[:,0] > 0.5), the same
               filter validation uses
  metric       pooled AP over every pixel of every scored sample, on GPU
               (evaluation/metrics.pooled_ap_gpu), never averaged per sample
  padding      re-checked at scoring time, see below

PADDING. Every model here must be invariant to appended pad frames: AFSeq
histories are variable and the collate pads at the END. Four of five were not,
and their rows measured something other than what they claimed -- U-Net and
ConvNeXt read X[:, -1] as "day t" when it was an all-zero frame 68% of the
time; ConvLSTM returned the cell state after the padded steps; UTAE and UTAE-AR
pooled padded steps into the LTAE's GroupNorm statistics. All are fixed, and
`pad_invariance` re-checks each model as it is scored so the failure cannot
come back silently.

PERSISTENCE is computed here too, from the same loader and the same filter, so
the floor of the table is measured rather than quoted: 0.167 +- 0.007 on the
v6 splits (1,194 events; dataset/curate.py says which 33 duplicates and 74 US
fires the build leaves out, and why). With every event of dataset_v6 it is 0.252, and the 0.198
of an earlier draft was a test set that no longer exists. What this prints is
the one that belongs next to the model rows it prints, which is the only
comparison the table actually makes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "training", ROOT / "benchmarks",
           ROOT / "experiments"):
    sys.path.insert(0, str(_p))

from runner import Job, run_queue                              # noqa: E402
from paths import AFSEQ, AFSEQ_ACTIVE, RUNS, SPLITS_DIR       # noqa: E402
from evaluation.metrics import pooled_ap_gpu                   # noqa: E402
from train import (SINGLE_STEP, build_model, forward,          # noqa: E402
                   last_real_frame, to_logits)

OURS = ("unet", "convnext", "convlstm", "utae", "utae_ar")
THEIRS = ("utae_res18",)
SPLITS = (5, 7, 23)
SPLIT_DIR = SPLITS_DIR                 # dataset/splits, shipped with the code
EPOCHS = 20
LOSS = "bce_dice"
LR = 1e-3
SEED_DEFAULT = 23          # paths.SEED: train.py omits it from the tag
FIRE_CHANNEL = 16          # AFSeq's active-fire mask

ROWS = ("persistence", "unet", "convnext", "convlstm", "utae",
        "utae_res18", "utae_ar")
SEQ_IN = {"convlstm", "utae", "utae_res18", "utae_ar"}
SEQ_OUT = {"utae_ar"}


# ------------------------------------------------------------------ launching
def tag_of(model: str, seed: int) -> str:
    lr_tag = f"{LR:g}".replace("-", "").replace(".", "p")
    tag = f"{model}_{LOSS}_lr{lr_tag}"
    if seed != SEED_DEFAULT:
        tag += f"_s{seed}"
    return tag + "_padfix"


def matrix(out: Path, splits: str = "v6") -> list[Job]:
    jobs = []
    for model in OURS:
        for seed in SPLITS:
            tag = tag_of(model, seed)
            jobs.append(Job(
                name=tag,
                cmd=[sys.executable, str(ROOT / "training" / "train.py"),
                     "--model", model, "--loss", LOSS, "--lr", str(LR),
                     "--epochs", str(EPOCHS), "--seed", str(seed),
                     "--workers", "4", "--out", str(out / model),
                     "--tag-extra", "_padfix"],
                log=out / "logs" / f"{model}_s{seed}.log",
                cwd=ROOT,
                output=out / model / f"{tag}.json",
                env={"AFSEQ_SPLITS": str(SPLIT_DIR / f"splits_{splits}_s{seed}.txt")}))
    for model in THEIRS:
        for seed in SPLITS:
            jobs.append(Job(
                name=f"{model}_s{seed}",
                cmd=[sys.executable,
                     str(ROOT / "benchmarks" / "train_afseq_theirs.py"),
                     "--model", model, "--features", "all",
                     "--workers", "4", "--seed", str(seed),
                     "--out", str(out / model)],
                log=out / "logs" / f"{model}_s{seed}.log",
                cwd=ROOT / "benchmarks",
                output=out / model / f"{model}_all_Tvar_s{seed}.json",
                env={"AFSEQ_SPLITS": str(SPLIT_DIR / f"splits_{splits}_s{seed}.txt")}))
    return jobs


# ------------------------------------------------------------------- scoring
def afseq_loader(splits_file, patch=64, max_history=16, horizon=1,
                 batch_size=4, workers=4):
    from dataset import afseq_loader
    from dataset.afseq_loader import (Seq2SeqActivePadded, collate_padded,
                                      dl_ids, load_ids)
    from dataset.afseq_cache import install_if_configured
    install_if_configured(afseq_loader)    # only if AFSEQ_CACHE is set

    ids = dl_ids(load_ids("Test", splits_file), AFSEQ_ACTIVE)
    ds = Seq2SeqActivePadded(ids, str(AFSEQ), str(AFSEQ_ACTIVE), patch,
                             max_history, horizon)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      collate_fn=lambda b: collate_padded(b, 0.0))


@torch.no_grad()
def score(model, name, loader, dev):
    """Pooled AP at t+1 over the scored samples. One pass, all on GPU."""
    ys, ps = [], []
    for X, FC, Y, V, _ in loader:
        X, FC = X.to(dev, non_blocking=True), FC.to(dev, non_blocking=True)
        Y, V = Y.to(dev, non_blocking=True), V.to(dev, non_blocking=True)
        keep = V[:, 0] > 0.5
        if not keep.any():
            continue
        if name == "persistence":
            p = last_real_frame(X)[:, FIRE_CHANNEL]
        elif name == "utae_res18":
            # theirs: (B,T,C,H,W) in, (B,1,H,W) logits out; pad frames are
            # masked inside SMPTempModelPadded, which is the whole point of it
            p = torch.sigmoid(model(X)[:, 0])
        else:
            out = forward(model, name, X, FC, None, 0.0)
            p = (torch.sigmoid(to_logits(out, name)[:, 0]) if name in SINGLE_STEP
                 else torch.sigmoid(out[:, 0, 0]))
        ys.append((Y[keep, 0, 0] > 0.5).flatten())
        ps.append(p[keep].flatten())
    if not ys:
        return float("nan"), 0
    n = sum(int(y.numel()) for y in ys)
    return pooled_ap_gpu(torch.cat(ys), torch.cat(ps), device=dev), n


def pad_invariance(model, name, dev, in_ch=17):
    """A model whose answer moves when a pad frame is appended is broken."""
    if name == "persistence":
        return 0.0
    torch.manual_seed(0)
    x = torch.rand(2, 4, in_ch, 64, 64, device=dev) + 0.01
    xp = torch.cat([x, torch.zeros(2, 3, in_ch, 64, 64, device=dev)], 1)
    fc = torch.rand(2, 4, 16, 64, 64, device=dev)
    fwd = ((lambda m, n, X, FC, Y, tf: m(X)) if name == "utae_res18"
           else forward)
    with torch.no_grad():
        a = fwd(model, name, x, fc, None, 0.0)
        b = fwd(model, name, xp, fc, None, 0.0)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    return (a - b).abs().max().item()


def load_model(name, runs, seed, dev):
    if name == "utae_res18":
        from afseq_models import SMPTempModelPadded
        js = list((runs / "utae_res18").glob(f"utae_res18_all_T*_s{seed}.json"))
        if not js:
            return None
        ck = json.loads(js[0].read_text())["best_ckpt"]
        m = SMPTempModelPadded(encoder_name="resnet18", encoder_weights=None,
                               n_channels=17, flatten_temporal_dimension=False,
                               pos_class_weight=236.0, loss_function="Focal")
        m.load_state_dict(torch.load(ck, map_location="cpu",
                                     weights_only=False)["state_dict"])
        return m.to(dev).eval()

    hits = sorted((runs / name).glob(f"best_{tag_of(name, seed)}.pth"))
    if not hits:
        return None
    # horizon 1 for every row: this table scores t+1 only, and UTAE-AR's first
    # step never touches fut_covars (they enter the rollout from k=1), so the
    # 5-step checkpoint loads into a 1-step model unchanged
    m = build_model(name, 1)
    m.load_state_dict(torch.load(hits[0], map_location="cpu"))
    return m.to(dev).eval()


def table(runs: Path, seeds, rows=ROWS, workers=4, batch_size=4, splits="v6"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = {s: afseq_loader(str(SPLIT_DIR / f"splits_{splits}_s{s}.txt"),
                               batch_size=batch_size, workers=workers)
               for s in seeds}
    res = {}
    for name in rows:
        per_seed = {}
        for s in seeds:
            model = None if name == "persistence" else load_model(name, runs, s, dev)
            if name != "persistence" and model is None:
                print(f"  {name} s{s}: sin checkpoint todavia")
                continue
            drift = pad_invariance(model, name, dev) if model is not None else 0.0
            ap_, n = score(model, name, loaders[s], dev)
            per_seed[s] = dict(ap=ap_, pixels=n, pad_drift=drift)
            flag = "" if drift < 1e-5 else f"  <-- padding drift {drift:.2e}"
            print(f"  {name:11s} s{s:<3d} AP={ap_:.4f}  ({n:,} px){flag}")
            del model
            torch.cuda.empty_cache()
        if per_seed:
            v = [d["ap"] for d in per_seed.values()]
            res[name] = dict(per_seed=per_seed, mean=statistics.mean(v),
                             std=statistics.stdev(v) if len(v) > 1 else 0.0,
                             n_seeds=len(v))

    print(f"\n{'Model':<14}{'Seq in':>8}{'Seq out':>9}{'AP':>18}{'seeds':>7}")
    for name in rows:
        if name not in res:
            continue
        r = res[name]
        cell = (f"{r['mean']:.3f} +- {r['std']:.3f}" if r["n_seeds"] > 1
                else f"{r['mean']:.3f}")
        print(f"{name:<14}{'si' if name in SEQ_IN else 'no':>8}"
              f"{'si' if name in SEQ_OUT else 'no':>9}{cell:>18}{r['n_seeds']:>7}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SUBSTR",
                    help="lanzar solo los trabajos cuyo nombre contenga esto")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SPLITS))
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="1 = sin padding en el batch; sirve para medir cuanto "
                         "cambia la AP de un modelo con el relleno")
    ap.add_argument("--out", type=Path, default=RUNS / "tabla2")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--splits", default="v6",
                    help="que splits_<tag>_s<seed>.txt entrenar/puntuar, buscado en "
                         "AFSEQ_SPLITS_DIR (dataset/splits: solo v6, el dataset "
                         "final). Con otro tag usa tambien otro --out")
    a = ap.parse_args()

    if a.launch:
        run_queue(matrix(a.out, a.splits), a.out / "done", dry_run=a.dry_run,
                  only=a.only)
    if a.table:
        res = table(a.out, a.seeds, tuple(a.rows), a.workers, a.batch_size,
                    a.splits)
        if a.json:
            a.json.write_text(json.dumps(res, indent=1))
            print(f"\n-> {a.json}")
    if not (a.launch or a.table):
        ap.print_help()


if __name__ == "__main__":
    main()
