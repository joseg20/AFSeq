#!/usr/bin/env python3
"""Experiment 4 -- the cross-benchmark table.

    python experiments/cross/cross.py --launch [--dry-run]
    python experiments/cross/cross.py --table

What each model scores on the OTHER benchmark, and what it scores when trained
on both:

                            AFSeq      WSTS+
      Trained on WSTS+        x          x
      Trained on AFSeq        x          x
      Trained on both         x          x

    x 3 models x 4 published folds.

  UTAE-AR       ours. BCE + soft Dice, lr 1e-3 -- our configuration, the one
                every other table of the paper uses.
  UTAE-Res18    theirs. Their best model overall.
  Res50-Unet    theirs. Their best single-frame encoder after res18.
                Both run their repo unmodified with the configuration every
                published sweep of theirs passes; see cross/train_theirs.py.

EVERYTHING HAPPENS IN THE VEG SPACE (7 channels), which is what makes this
table commensurable with wstsplus.py: the WSTS+ column here and the Veg row of wstsplus.py
are the same data through the same slice. AFSeq is mapped in by cross/channels
.py -- 5 of the 7 slots have a counterpart, 2 are filled with the WSTS training
mean, and that fill is a declared confound: a model trained on the union can
tell the benchmarks apart by those two constant slots.

SCORED SEPARATELY, NEVER POOLED. A model is trained once and then scored on
each benchmark's own test set, in its own column. Pooling would let whichever
side contributes more pixels decide the number.

CONSISTENCY CHECK. The UTAE-AR / trained-on-WSTS+ / scored-on-WSTS+ cell should
land near the Veg row of wstsplus.py, and the published checkpoints scored through this
same loader give 0.4299 against Table 3's 0.4295. If a rewrite moves that cell,
the loader changed, not the model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (ROOT, ROOT / "experiments", str(HERE)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from runner import Job, run_queue                              # noqa: E402
from paths import RUNS                                         # noqa: E402

MODELS = ("utae_ar", "utae_res18", "res50_unet")
SOURCES = ("wstsplus", "afseq", "both")
FOLDS = (0, 1, 2, 3)
LOSS = "bce_dice"
LR = 1e-3
EPOCHS = 20


def tag_of(model: str, source: str, fold: int) -> str:
    if model == "utae_ar":
        lr_tag = f"{LR:g}".replace("-", "").replace(".", "p")
        return f"utae_ar_fused_{source}_f{fold}_{LOSS}_lr{lr_tag}"
    return f"{model}_fused_{source}_f{fold}"


def matrix(out: Path, models=MODELS, sources=SOURCES, folds=FOLDS,
           native: bool = True) -> list[Job]:
    """Ours through training/train.py, theirs through cross/train_theirs.py.

    `native` prepends the WSTS+-only Res18-Unet baseline, which runs THEIR repo
    end to end -- their loader, their windowing, their train.py. Everything else
    in this table goes through our Veg-space loader, so that row is the control
    that says the space conversion did not change what their model can do. It
    also produces the encoder the representation analysis reads from.
    """
    jobs = []
    if native:
        from native import matrix as native_matrix
        jobs += native_matrix(out / "native")
    for source in sources:
        for model in models:
            for fold in folds:
                tag = tag_of(model, source, fold)
                if model == "utae_ar":
                    cmd = [sys.executable, str(ROOT / "training" / "train.py"),
                           "--model", "utae_ar", "--benchmark", "fused",
                           "--fusion-source", source, "--fold", str(fold),
                           "--loss", LOSS, "--lr", str(LR),
                           "--epochs", str(EPOCHS), "--workers", "4",
                           "--out", str(out)]
                    output = out / f"{tag}.json"
                else:
                    cmd = [sys.executable, str(HERE / "train_theirs.py"),
                           "--model", model, "--source", source,
                           "--fold", str(fold), "--workers", "4",
                           "--out", str(out)]
                    output = out / f"{tag}.json"
                jobs.append(Job(name=tag, cmd=cmd,
                                log=out / "logs" / f"{tag}.log",
                                cwd=ROOT, output=output))
    return jobs


def table(out: Path, models=MODELS, sources=SOURCES, folds=FOLDS):
    """Mean +- std over the four folds, per model and training source."""
    print(f"{'trained on':<14}{'model':<14}{'val AP (mean +- std)':>24}{'folds':>7}")
    for source in sources:
        for model in models:
            vals = []
            for fold in folds:
                js = out / f"{tag_of(model, source, fold)}.json"
                if js.exists():
                    vals.append(json.loads(js.read_text())["best_val_ap"])
            if not vals:
                continue
            cell = (f"{statistics.mean(vals):.4f} +- {statistics.stdev(vals):.4f}"
                    if len(vals) > 1 else f"{statistics.mean(vals):.4f}")
            print(f"{source:<14}{model:<14}{cell:>24}{len(vals):>7}")
    print("\nNota: best_val_ap. La puntuacion por benchmark en test (columnas")
    print("AFSeq y WSTS+, nunca combinadas) se calcula sobre estos checkpoints.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SUBSTR",
                    help="lanzar solo los trabajos cuyo nombre contenga esto")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    ap.add_argument("--no-native", action="store_true",
                    help="omitir el baseline de su repo (4 folds)")
    ap.add_argument("--only-native", action="store_true",
                    help="lanzar SOLO ese baseline. Va primero en la cola "
                         "porque su encoder es el que alimenta el analisis "
                         "de representaciones, y no tiene "
                         "sentido esperar a las 36 corridas de la matriz")
    ap.add_argument("--out", type=Path, default=RUNS / "cross")
    a = ap.parse_args()

    if a.launch:
        if a.only_native:
            from native import matrix as native_matrix
            jobs = native_matrix(a.out / "native")
        else:
            jobs = matrix(a.out, tuple(a.models), tuple(a.sources),
                          tuple(a.folds), native=not a.no_native)
        run_queue(jobs, a.out / "done", dry_run=a.dry_run, only=a.only)
    if a.table:
        table(a.out, tuple(a.models), tuple(a.sources), tuple(a.folds))
    if not (a.launch or a.table):
        ap.print_help()


if __name__ == "__main__":
    main()
