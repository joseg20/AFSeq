#!/usr/bin/env python3
"""Experiment 1 -- UTAE-AR on WSTS+, across feature sets and folds.

    python experiments/exp1.py --launch          # train the matrix
    python experiments/exp1.py --launch --dry-run
    python experiments/exp1.py --table           # score it

THE MATRIX: 3 feature sets x 4 published folds x 2 history regimes = 24 runs.

  features   Veg / Multi / All, as index lists over the 40 processed channels
             (benchmarks/wstsplus_data.FEATURES)
  folds      the four PUBLISHED WSTS+ folds, the ones their Table 5 lives on --
             not the 12-fold table a deleted commit of their repo carried
  history    fixed     their protocol: 5 full days, which silently drops the
                       opening days of every fire (10,408 of 12,644 training
                       windows on fold 0)
             variable  every day that has a future day, with whatever history
                       exists behind it, zero-padded and MASKED. This is a
                       separate experiment, not a better version of the first:
                       it only became correct once utae.masked_group_norm
                       stopped pooling padded steps into the LTAE statistics

  All windows count. `require_fire_on_t` stays off, which is how the published
  anchor was measured: 5,135 test samples on fold 0, ignition and empty windows
  included, not the 2,203 that AFSeq's fire-on-t filter leaves.

THE ANCHOR. Trained from scratch these runs will not land exactly on Table 3 --
different initialisation, different batch order. What DOES reproduce, and what
should be checked whenever this file changes, is scoring the published
checkpoints through this same loader: best_AR_Veg_p4f{0..3}_bce_dice_lr0p001.pth
gives 0.3778 / 0.3089 / 0.4945 / 0.5385, mean 0.4299 against a published
0.4295. Deltas of 3-9 e-4 -- that is a reproduction, and it is what says the
release reads WSTS+ the same way the research tree did.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from runner import Job, run_queue                              # noqa: E402
from paths import RUNS                                         # noqa: E402

FEATURES = ("Veg", "Multi", "All")
FOLDS = (0, 1, 2, 3)
HISTORIES = ("fixed", "variable")
EPOCHS = 20
LOSS = "bce_dice"
LR = 1e-3


def tag_of(features: str, fold: int, history: str) -> str:
    """The tag training/train.py will write, so --table can find the run."""
    lr_tag = f"{LR:g}".replace("-", "").replace(".", "p")
    return f"utae_ar_wstsplus_{features}_f{fold}_{history}_{LOSS}_lr{lr_tag}"


def matrix(out: Path, histories=HISTORIES) -> list[Job]:
    """The runs, fixed-history first so a half-finished queue is still useful."""
    jobs = []
    for history in histories:
        for features in FEATURES:
            for fold in FOLDS:
                tag = tag_of(features, fold, history)
                jobs.append(Job(
                    name=tag,
                    cmd=[sys.executable, str(ROOT / "training" / "train.py"),
                         "--model", "utae_ar", "--benchmark", "wstsplus",
                         "--fold", str(fold), "--features", features,
                         "--history", history, "--loss", LOSS,
                         "--lr", str(LR), "--epochs", str(EPOCHS),
                         "--workers", "4", "--out", str(out)],
                    log=out / "logs" / f"{tag}.log",
                    cwd=ROOT,
                    output=out / f"{tag}.json"))
    return jobs


def table(out: Path):
    """Mean +- std over folds, per feature set and history regime."""
    print(f"{'features':<10}{'history':<10}" +
          "".join(f"{'f'+str(f):>9}" for f in FOLDS) + f"{'media':>10}{'std':>8}")
    for history in HISTORIES:
        for features in FEATURES:
            vals = []
            for fold in FOLDS:
                js = out / f"{tag_of(features, fold, history)}.json"
                vals.append(json.loads(js.read_text())["best_val_ap"]
                            if js.exists() else None)
            got = [v for v in vals if v is not None]
            if not got:
                continue
            cells = "".join(f"{v:>9.4f}" if v is not None else f"{'-':>9}"
                            for v in vals)
            m = statistics.mean(got)
            s = statistics.stdev(got) if len(got) > 1 else 0.0
            print(f"{features:<10}{history:<10}{cells}{m:>10.4f}{s:>8.4f}")
    print("\nNota: best_val_ap, no AP de test. El test se puntua con el mismo")
    print("protocolo del ancla (crop 128, AP pooled a t+1) en evaluation/.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SUBSTR",
                    help="lanzar solo los trabajos cuyo nombre contenga esto")
    ap.add_argument("--history", choices=HISTORIES, action="append",
                    help="limitar a un regimen; por defecto los dos")
    ap.add_argument("--out", type=Path, default=RUNS / "exp1")
    a = ap.parse_args()

    if a.launch:
        jobs = matrix(a.out, tuple(a.history) if a.history else HISTORIES)
        run_queue(jobs, a.out / "done", dry_run=a.dry_run, only=a.only)
    if a.table:
        table(a.out)
    if not (a.launch or a.table):
        ap.print_help()


if __name__ == "__main__":
    main()
