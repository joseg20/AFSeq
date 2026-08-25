#!/usr/bin/env python3
"""Experiment 3 -- how fast should teacher forcing decay? (UTAE-AR, AFSeq)

    python experiments/exp3.py --launch
    python experiments/exp3.py --table

UTAE-AR rolls five days forward, feeding its own prediction back in. During
training it can instead be fed the ground truth -- teacher forcing -- which
makes the loss easier but trains the model in a regime it will never see at
test time. The schedule that decays it is therefore a real hyper-parameter of
the method, not a detail, and this is the ablation of it.

THE SCHEDULES, all over 20 epochs:

  none        tf = 0 from the start. Free running throughout: the model only
              ever sees its own predictions.
  linear      tf = max(0.5, 1 - 0.5 * ep / (E-1)). The paper's default. Note it
              floors at 0.5 -- it never stops teacher forcing entirely.
  ss(k)       tf = k / (k + exp(ep / k)), inverse-sigmoid decay (Bengio et al.
              2015). k sets how long the model is held on ground truth:

                  k     epoch 0   epoch 10   epoch 19
                   1      0.50      0.000      0.000
                   2      0.67      0.013      0.000
                   3      0.75      0.097      0.005
                   5      0.83      0.405      0.084
                  10      0.91      0.785      0.599
                  20      0.95      0.881      0.885

              So k=1 is almost free running, k=20 almost pure teacher forcing,
              and the interesting region is 3-10 where the crossover happens
              inside the run.

WHY AP PER HORIZON, not just t+1. Teacher forcing cannot affect the first step:
day t+1 is predicted before fut_covars or any fed-back mask is touched, so
every schedule gives an identical t+1 by construction. If a table of this
ablation shows t+1 moving, something else changed. The effect lives at t+2..t+5,
where the errors compound, which is exactly what the schedule is meant to
control.

SEEDS. One split (s23) by default: this is an ablation over 8 configurations,
and 3 splits would be 24 runs for a curve whose shape, not whose third decimal,
is the point. Use --seeds for the final version if a number goes in the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from runner import Job, run_queue                              # noqa: E402
from paths import RUNS, SPLITS_DIR                             # noqa: E402

KS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
MODES = ("none", "linear")          # the two non-ss baselines
SPLIT_DIR = SPLITS_DIR                 # dataset/splits, shipped with the code
EPOCHS = 20
LOSS = "bce_dice"
LR = 1e-3
SEED_DEFAULT = 23


def tf_at(mode: str, k: float, ep: int, epochs: int = EPOCHS) -> float:
    """The schedule, copied from training/train.py so --table can print it."""
    if mode == "none":
        return 0.0
    if mode == "ss":
        return float(k / (k + math.exp(ep / k)))
    return max(0.5, 1.0 - 0.5 * ep / max(1, epochs - 1))


def tag_of(mode: str, k: float, seed: int) -> str:
    """The tag training/train.py writes for this configuration."""
    lr_tag = f"{LR:g}".replace("-", "").replace(".", "p")
    tag = f"utae_ar_{LOSS}_lr{lr_tag}"
    if mode != "linear":
        tag += f"_tf{mode}" + (f"k{k:g}" if mode == "ss" else "")
    if seed != SEED_DEFAULT:
        tag += f"_s{seed}"
    return tag


def configs(ks=KS, modes=MODES):
    for mode in modes:
        yield mode, 0.0
    for k in ks:
        yield "ss", k


def matrix(out: Path, seeds, ks=KS, modes=MODES) -> list[Job]:
    jobs = []
    for seed in seeds:
        for mode, k in configs(ks, modes):
            tag = tag_of(mode, k, seed)
            cmd = [sys.executable, str(ROOT / "training" / "train.py"),
                   "--model", "utae_ar", "--loss", LOSS, "--lr", str(LR),
                   "--epochs", str(EPOCHS), "--seed", str(seed),
                   "--tf-mode", mode, "--workers", "4", "--out", str(out)]
            if mode == "ss":
                cmd += ["--k", f"{k:g}"]
            jobs.append(Job(name=tag, cmd=cmd,
                            log=out / "logs" / f"{tag}.log", cwd=ROOT,
                            output=out / f"{tag}.json",
                            env={"AFSEQ_SPLITS":
                                 str(SPLIT_DIR / f"splits_v6_s{seed}.txt")}))
    return jobs


def table(out: Path, seeds, ks=KS, modes=MODES):
    print(f"{'schedule':<12}{'tf ep0':>8}{'tf ep19':>9}{'val AP':>10}{'seeds':>7}")
    for mode, k in configs(ks, modes):
        vals = []
        for seed in seeds:
            js = out / f"{tag_of(mode, k, seed)}.json"
            if js.exists():
                vals.append(json.loads(js.read_text())["best_val_ap"])
        if not vals:
            continue
        name = mode if mode != "ss" else f"ss k={k:g}"
        cell = (f"{statistics.mean(vals):.4f}" if len(vals) == 1 else
                f"{statistics.mean(vals):.4f}+-{statistics.stdev(vals):.3f}")
        print(f"{name:<12}{tf_at(mode, k, 0):>8.2f}{tf_at(mode, k, EPOCHS-1):>9.2f}"
              f"{cell:>10}{len(vals):>7}")
    print("\nRecordatorio: t+1 es identico para todos los horarios por")
    print("construccion; el efecto esta en t+2..t+5.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SUBSTR",
                    help="lanzar solo los trabajos cuyo nombre contenga esto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED_DEFAULT])
    ap.add_argument("--ks", type=float, nargs="+", default=list(KS))
    ap.add_argument("--out", type=Path, default=RUNS / "exp3")
    a = ap.parse_args()

    if a.launch:
        run_queue(matrix(a.out, a.seeds, tuple(a.ks)), a.out / "done",
                  dry_run=a.dry_run, only=a.only)
    if a.table:
        table(a.out, a.seeds, tuple(a.ks))
    if not (a.launch or a.table):
        ap.print_help()


if __name__ == "__main__":
    main()
