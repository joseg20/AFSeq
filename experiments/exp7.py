#!/usr/bin/env python3
"""Experiment 7 -- UTAE-AR on the ORIGINAL WSTS, 12-fold protocol (supp. S3.1).

    python experiments/exp7.py --launch [--dry-run]
    python experiments/exp7.py --table

THE PROTOCOL. The original WildfireSpreadTS evaluates over the four years
2018-2021 with 12 folds: one year held out for test, one of the remaining three
for validation, the other two for training. WSTS+ replaced it with the 4-fold
table of eight years, which is what exp1 and the cross-benchmark use; this file
exists so the AFSeq paper can also report on the protocol the ORIGINAL benchmark
defined, which is the one most prior work cites.

Folds are addressed as 100+i in benchmarks/wstsplus_data (split12_years), so the
4-fold path stays byte-identical. Feature set is Veg -- the space the published
checkpoints and the cross-benchmark live in -- and history is "variable", ours.
"""
from __future__ import annotations

import argparse, json, os, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "experiments", ROOT / "benchmarks"):
    sys.path.insert(0, str(_p))
from paths import RUNS                                     # noqa: E402
from runner import Job, run_queue                          # noqa: E402
from wstsplus_data import FOLDS12, split12_years           # noqa: E402

FEATURES = "Veg"
HISTORY = "variable"
EPOCHS, LOSS, LR = 20, "bce_dice", 1e-3


def tag_of(fold: int) -> str:
    lr_tag = f"{LR:g}".replace("-", "").replace(".", "p")
    return f"AR_{FEATURES}_p{fold}f{fold}_{LOSS}_lr{lr_tag}"


def matrix(out: Path) -> list[Job]:
    jobs = []
    for i in range(len(FOLDS12)):
        fold = 100 + i
        tag = f"AR_{FEATURES}_wsts12_f{i}_{LOSS}"
        jobs.append(Job(
            name=tag,
            cmd=[sys.executable, str(ROOT / "training" / "train.py"),
                 "--model", "utae_ar", "--benchmark", "wstsplus",
                 "--fold", str(fold), "--features", FEATURES,
                 "--history", HISTORY, "--loss", LOSS, "--lr", str(LR),
                 "--epochs", str(EPOCHS),
                 "--workers", os.environ.get("WSP_WORKERS", "4"),
                 "--tag-extra", f"_wsts12f{i}",
                 "--out", str(out / f"f{i}")],
            log=out / "logs" / f"{tag}.log",
            cwd=ROOT,
            output=out / f"f{i}" / "done.json"))
    return jobs


def table(out: Path):
    rows = []
    for i in range(len(FOLDS12)):
        tr, va, te = split12_years(i)
        js = sorted((out / f"f{i}").glob("*.json"))
        js = [j for j in js if j.name != "done.json"]
        ap = None
        if js:
            d = json.loads(js[0].read_text())
            ap = d.get("test_AP") or d.get("best_val_ap")
        rows.append((i, tr, va[0], te[0], ap))
    print(f"{'fold':>5}{'train':>14}{'val':>7}{'test':>7}{'AP':>9}")
    for i, tr, va, te, ap in rows:
        s = f"{ap:.4f}" if ap is not None else "--"
        print(f"{i:>5}{str(tr):>14}{va:>7}{te:>7}{s:>9}")
    got = [r[4] for r in rows if r[4] is not None]
    if got:
        m = statistics.mean(got)
        sd = statistics.pstdev(got) if len(got) > 1 else 0.0
        print(f"\nmean +- std over {len(got)} folds: {m:.4f} +- {sd:.4f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--out", type=Path, default=RUNS / "exp7")
    a = ap.parse_args()
    if a.launch:
        run_queue(matrix(a.out), a.out / "done", dry_run=a.dry_run, only=a.only)
    if a.table:
        table(a.out)
    if not (a.launch or a.table):
        ap.print_help()
