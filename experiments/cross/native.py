"""Exp 4, part one: the native WSTS+ baseline (their model, their loader).

    python experiments/cross/native.py --launch
    (scored by cross/cross.py --table)

                          AFSeq      WSTS+
    Trained on WSTS+
      UTAE-AR              x           x      <- reproduces Table 3's anchor
      Res18-UNet           x           x      <- trained here, their repo
    Trained on AFSeq
      UTAE-AR              x           x
      Res18-UNet           x           x

WHOSE CODE IS WHOSE. The Res18 rows run THEIR repository unmodified
(benchmarks/wsts+), launched through their own src/train.py with the config
every published sweep of theirs uses: cfgs/unet/res18_monotemporal.yaml, i.e.
resnet18+imagenet, focal loss, AdamW at 1e-3, 10k steps, batch 64. It is not
re-implemented in models/ precisely so this stays unambiguous.

Their repo needs wsts+_compat.py applied first. Seven patches, all reversible,
all documented there. One of them is not a version fix but a bug: their
FireSpreadDataset's index lookup never stops at the fire it finds, so every
sample resolves into the LAST year of the split, measured on fold 0, all 401
sampled indices landed in 2021 and 272 named the wrong fire, leaving the 7,071
datapoints of 2016/2017/2020 unreachable. Without that patch a WSTS+ baseline
is trained and scored on one mangled year.

THE ANCHOR for the UTAE-AR / WSTS+ cell is not retrained: it reuses
best_AR_Veg_p4f{0..3}_bce_dice_lr0p001.pth. Scored through
benchmarks/wstsplus_data.py those give 0.3778 / 0.3089 / 0.4945 / 0.5385,
mean 0.4299 against Table 3's 0.4295.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from runner import Job, run_queue                              # noqa: E402
from paths import RUNS, WSTS_HDF5                              # noqa: E402

FOLDS = (0, 1, 2, 3)
VEG = "[0,1,2,3,4,38,39]"      # their Veg feature set over the 40 channels
MAX_STEPS = 10000              # their schedule
N_LEADING = 5


def matrix(out: Path) -> list[Job]:
    """Res18-UNet on WSTS+, one job per published fold, through their train.py."""
    theirs = ROOT / "benchmarks" / "wsts+"
    jobs = []
    for fold in FOLDS:
        jobs.append(Job(
            name=f"res18_wstsplus_f{fold}",
            cmd=[sys.executable, "src/train.py",
                 "-c", "cfgs/unet/res18_monotemporal.yaml",
                 "--trainer", "cfgs/trainer_single_gpu.yaml",
                 "--data", "cfgs/data_monotemporal_full_features.yaml",
                 "--data.data_dir", str(WSTS_HDF5),
                 "--data.additional_data", "true",
                 "--data.data_fold_id", str(fold),
                 "--data.features_to_keep", VEG,
                 "--data.n_leading_observations", str(N_LEADING),
                 "--data.num_workers", "4",
                 "--trainer.default_root_dir", str(out / f"fold{fold}"),
                 "--trainer.max_steps", str(MAX_STEPS)],
            log=out / "logs" / f"fold{fold}.log",
            cwd=theirs,
            # their train.py assumes a wandb sweep agent created the run;
            # offline keeps it self-contained. See the COMPAT patch.
            env={"WANDB_MODE": "offline", "PYTHONPATH": str(theirs)}))
    return jobs


def check_compat():
    """Refuse to launch against an unpatched repo: the index bug is silent."""
    fsd = (ROOT / "benchmarks" / "wsts+" / "src" / "dataloader"
           / "FireSpreadDataset.py").read_text()
    if "COMPAT-INDEX" not in fsd:
        raise SystemExit(
            "The WSTS+ repository is not patched: without COMPAT-INDEX every\n"
            "sample falls in the last year of the split. Run this first:\n"
            "  python benchmarks/wsts+_compat.py --apply")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=RUNS / "res18_wstsplus")
    a = ap.parse_args()

    if a.launch:
        check_compat()
        run_queue(matrix(a.out), a.out / "done", dry_run=a.dry_run)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
