#!/usr/bin/env python3
"""Experiment 6 -- leave-one-region-out on AFSeq (supplementary material).

    python dataset/make_splits.py --holdout Chile US S.Europe   # once: the splits
    python experiments/exp6.py --launch [--dry-run]
    python experiments/exp6.py --table [--json out.json]

THE QUESTION. The shipped splits (dataset/make_splits.py) are stratified by
region, so every number in exp2 is interpolation: the test fires come from
the same three regions the model trained on. This experiment holds a region
out entirely. Train and Val on the other two regions (Val drawn at random
from THEM, so the checkpoint is selected on the distribution it was trained
on), Test on every event of the third. Three regions x 2 models x 1 seed.

THE REGIONS are the ones dataset/curate.py assigns from the event centroid:
"Chile" is its label for the whole southern hemisphere (Chile 59, Argentina
254, Bolivia 140 events in v7), "US" (422) and "S.Europe" (319). Chile alone
would be too small a test set (59 fires), so it stays inside its region.

THE TABLE, one block per held-out region:

  persistence       the floor on that region's fires, as exp2 measures it
  <model> LORO      trained on the other two regions, scored on this one
  <model> random    exp2's checkpoints (3 seeds), scored on THIS region's fires
                    of their own random test split: the in-distribution
                    reference for the same fires. mean +- std over seeds
  delta             LORO - random

READ THE DELTA, NOT THE AP. The floors differ by region (dataset/curate.py:
US 0.307 after curation, Chile 0.174, S.Europe 0.170), so absolute APs across
blocks say more about the region than about the transfer. The delta holds
the fires fixed and moves only the training set.

TWO CAVEATS, stated rather than fixed:
  - provenance is confounded with geography: S.Europe comes from per-country
    VIIRS pools plus an early batch, the southern hemisphere from the
    argentina/bolivia pools; a drop on S.Europe mixes the two.
  - one seed. The splits are not seeded by name (a held-out region is a held-
    out region); --seed changes the training seed only.

Scoring is exp2's, imported, not copied: pooled AP at t+1, crop 128, the
padding-invariance check on every checkpoint.
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
for _p in (ROOT, ROOT / "experiments", ROOT / "dataset"):
    sys.path.insert(0, str(_p))

import exp2                                                    # noqa: E402
from exp2 import EPOCHS, LOSS, LR, PATCH, SEQ_IN, SEQ_OUT, tag_of  # noqa: E402
from make_splits import REGION_SLUG                            # noqa: E402
from paths import AFSEQ, RUNS, SPLITS_DIR                     # noqa: E402
from runner import Job, run_queue                              # noqa: E402

REGIONS = tuple(sorted(REGION_SLUG))          # ('Chile', 'S.Europe', 'US')
MODELS = ("utae_ar", "utae_res18")            # ours, and their best
ROWS = ("persistence",) + MODELS
SEED = 23                                     # training seed; exp2's default
EXP2_SEEDS = exp2.SPLITS                      # the reference's splits
STATS = SPLITS_DIR / "fire_stats_v7.json"     # region of every event, shipped


def split_file(region: str) -> Path:
    return SPLITS_DIR / f"splits_v7_loro_{REGION_SLUG[region]}.txt"


def region_of() -> dict:
    """{fire id: region}, from the statistics make_splits.py ships."""
    return {f: s["region"] for f, s in json.loads(STATS.read_text()).items()}


# ------------------------------------------------------------------ launching
def matrix(out: Path, regions=REGIONS, models=MODELS, seed=SEED) -> list[Job]:
    """One run per (held-out region, model), same commands as exp2's rows."""
    missing = [r for r in regions if not split_file(r).exists()]
    if missing:
        raise SystemExit(
            "faltan los splits de " + ", ".join(missing) + ": genera con\n"
            "    python dataset/make_splits.py --holdout " + " ".join(missing))
    jobs = []
    for region in regions:
        slug = REGION_SLUG[region]
        env = {"AFSEQ_SPLITS": str(split_file(region))}
        for model in models:
            if model == "utae_res18":
                jobs.append(Job(
                    name=f"{slug}_{model}",
                    cmd=[sys.executable,
                         str(ROOT / "benchmarks" / "train_afseq_theirs.py"),
                         "--model", model, "--features", "all",
                         "--workers", "4", "--seed", str(seed),
                         "--out", str(out / slug / model)],
                    log=out / "logs" / f"{slug}_{model}.log",
                    cwd=ROOT / "benchmarks",
                    output=out / slug / model / f"{model}_all_Tvar_s{seed}.json",
                    env=env))
            else:
                tag = tag_of(model, seed)
                jobs.append(Job(
                    name=f"{slug}_{model}",
                    cmd=[sys.executable, str(ROOT / "training" / "train.py"),
                         "--model", model, "--loss", LOSS, "--lr", str(LR),
                         "--epochs", str(EPOCHS), "--seed", str(seed),
                         "--workers", "4", "--out", str(out / slug / model),
                         "--tag-extra", "_padfix"],
                    log=out / "logs" / f"{slug}_{model}.log",
                    cwd=ROOT,
                    output=out / slug / model / f"{tag}.json",
                    env=env))
    return jobs


# ------------------------------------------------------------------- scoring
def loader_for(ids: list, batch_size: int, workers: int):
    """exp2's test loader over an explicit id list (a region's fires)."""
    from dataset import afseq_loader
    from dataset.afseq_cache import install_if_configured
    from dataset.afseq_loader import Seq2SeqActivePadded, collate_padded, dl_ids
    install_if_configured(afseq_loader)    # only if AFSEQ_CACHE is set
    ids = dl_ids(ids, AFSEQ)
    ds = Seq2SeqActivePadded(ids, str(AFSEQ), PATCH, 16, 1)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      collate_fn=lambda b: collate_padded(b, 0.0))


def _score_one(name, model, loader, dev):
    drift = exp2.pad_invariance(model, name, dev) if model is not None else 0.0
    ap, n = exp2.score(model, name, loader, dev)
    return dict(ap=ap, pixels=n, pad_drift=drift)


def table(out: Path, exp2_out: Path, regions=REGIONS, rows=ROWS, workers=4,
          batch_size=4, seed=SEED):
    from dataset.afseq_loader import load_ids
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    reg = region_of()
    res = {}
    for region in regions:
        f = split_file(region)
        if not f.exists():
            print(f"  {region}: sin splits ({f.name}); salta")
            continue
        test_ids = load_ids("Test", f)
        assert all(reg[i] == region for i in test_ids), f"{f.name}: test no es solo {region}"
        loader = loader_for(test_ids, batch_size, workers)
        block = dict(n_fires=len(test_ids), rows={})
        print(f"\n== held out: {region}  ({len(test_ids)} fires)")
        for name in rows:
            if name == "persistence":
                r = _score_one(name, None, loader, dev)
                print(f"  {name:11s}         AP={r['ap']:.4f}  ({r['pixels']:,} px)")
                block["rows"][name] = dict(loro=r)
                continue
            # LORO: trained on the other regions
            model = exp2.load_model(name, out / REGION_SLUG[region], seed, dev)
            if model is None:
                print(f"  {name:11s} LORO    sin checkpoint todavia")
                loro = None
            else:
                loro = _score_one(name, model, loader, dev)
                flag = "" if loro["pad_drift"] < 1e-5 else f"  <-- padding drift {loro['pad_drift']:.2e}"
                print(f"  {name:11s} LORO    AP={loro['ap']:.4f}  ({loro['pixels']:,} px){flag}")
                del model
                torch.cuda.empty_cache()
            # reference: exp2's checkpoints on this region's fires of THEIR test split
            ref = {}
            for s in EXP2_SEEDS:
                model = exp2.load_model(name, exp2_out, s, dev)
                if model is None:
                    continue
                ids = [i for i in load_ids("Test", SPLITS_DIR / f"splits_v7_s{s}.txt")
                       if reg[i] == region]
                ref[s] = _score_one(name, model, loader_for(ids, batch_size, workers), dev)
                ref[s]["n_fires"] = len(ids)
                del model
                torch.cuda.empty_cache()
            if ref:
                v = [d["ap"] for d in ref.values()]
                mean = statistics.mean(v)
                std = statistics.stdev(v) if len(v) > 1 else 0.0
                print(f"  {name:11s} random  AP={mean:.4f} +- {std:.4f}  "
                      f"({len(v)} seeds, {min(d['n_fires'] for d in ref.values())}-"
                      f"{max(d['n_fires'] for d in ref.values())} fires each)")
                if loro is not None:
                    print(f"  {name:11s} delta   {loro['ap'] - mean:+.4f}")
            block["rows"][name] = dict(
                loro=loro, random=dict(per_seed=ref, mean=mean, std=std) if ref else None,
                delta=(loro["ap"] - mean) if (loro is not None and ref) else None)
        res[region] = block

    print(f"\n{'held out':<11}{'model':<13}{'seq':>5}{'LORO':>9}{'random':>17}{'delta':>9}")
    for region, block in res.items():
        for name, r in block["rows"].items():
            if name == "persistence":
                print(f"{region:<11}{name:<13}{'-':>5}{r['loro']['ap']:>9.3f}"
                      f"{'':>17}{'':>9}")
                continue
            seq = "s2s" if name in SEQ_OUT else ("seq" if name in SEQ_IN else "1")
            loro = f"{r['loro']['ap']:.3f}" if r["loro"] else "-"
            rnd = (f"{r['random']['mean']:.3f} +- {r['random']['std']:.3f}"
                   if r["random"] else "-")
            delta = f"{r['delta']:+.3f}" if r["delta"] is not None else "-"
            print(f"{region:<11}{name:<13}{seq:>5}{loro:>9}{rnd:>17}{delta:>9}")
    print("\nNota: AP de test, pooled a t+1, crop 128 (exp2.score). 'random' = los "
          "checkpoints de exp2 sobre los incendios de ESTA region de su propio test.")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SUBSTR",
                    help="lanzar solo los trabajos cuyo nombre contenga esto")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS),
                    choices=list(REGIONS), metavar="REGION",
                    help="regiones a excluir, una corrida por region: "
                         + ", ".join(REGIONS))
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(exp2.OURS) + list(exp2.THEIRS), metavar="MODEL")
    ap.add_argument("--seed", type=int, default=SEED, help="semilla de entrenamiento")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", type=Path, default=RUNS / "exp6")
    ap.add_argument("--exp2-out", type=Path, default=RUNS / "tabla2",
                    help="donde estan los checkpoints de exp2 (la referencia)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    if a.launch:
        run_queue(matrix(a.out, tuple(a.regions), tuple(a.models), a.seed),
                  a.out / "done", dry_run=a.dry_run, only=a.only)
    if a.table:
        rows = ("persistence",) + tuple(a.models)
        res = table(a.out, a.exp2_out, tuple(a.regions), rows, a.workers,
                    a.batch_size, a.seed)
        if a.json:
            a.json.write_text(json.dumps(res, indent=1))
            print(f"\n-> {a.json}")
    if not (a.launch or a.table):
        ap.print_help()


if __name__ == "__main__":
    main()
