#!/usr/bin/env python3
"""The AFSeq splits: three seeds, stratified by region, fires as the unit.

    python dataset/make_splits.py --seeds 5 7 23
    -> dataset/splits/splits_v7_s{seed}.txt          (what ships with the repo)

Splits whatever the dataset tree contains. What is NOT in the tree is decided
upstream by `build_afseq.py --phase curate` (dataset/curate.py): duplicated
events and the 15% most persistent US fires leave the dataset there, not
here, so the download produces the dataset the paper uses and this file only
distributes it: 70 / 15 / 15 per region (Chile, US, S.Europe), one RNG per
seed, ids sorted before shuffling so the files are reproducible.

`--target LO HI` is optional and off by default: it swaps test and train fires
of the same region until the test's estimated persistence AP sits in the
window, keeping every region's share of fires, sampled days and burned pixels
within `--tol`. It exists for benchmark studies (what if the test floor were
X?) and is not what produced the shipped splits. Persistence of the shipped
splits, measured with exp2.py --table --rows persistence: 0.124 +- 0.013 (dataset_v7).

`--holdout REGION [REGION ...]` writes leave-one-region-out splits instead
(experiments/exp6.py): Test = EVERY event of that region, Train/Val = the other
regions, split at random with the same train:val ratio as the shipped splits
(70:15) and the same RNG discipline (ids sorted before shuffling, one RNG per
seed). Nothing is pinned. -> splits_<tag>_loro_<slug>.txt, slug in REGION_SLUG.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import AFSEQ, SPLITS_DIR  # noqa: E402
from dataset.curate import fire_stats, overlap, persistence_ap as ap_of  # noqa: E402

SPLIT_DIR = SPLITS_DIR                 # dataset/splits, shipped with the code
FRACS = (0.70, 0.15, 0.15)              # v4's
KEYS = ("fires", "days", "burned")


def shares(stats, fids) -> dict:
    """{region: {fires, days, burned}} as fractions of the given set."""
    tot = defaultdict(lambda: defaultdict(float))
    for f in fids:
        r = stats[f]["region"]
        tot[r]["fires"] += 1
        tot[r]["days"] += stats[f]["days"]
        tot[r]["burned"] += stats[f]["burned"]
    sums = {k: sum(tot[r][k] for r in tot) for k in KEYS}
    return {r: {k: tot[r][k] / max(sums[k], 1) for k in KEYS} for r in tot}


def imbalance(stats, fids, ref: dict) -> float:
    """Largest relative deviation of a region's share (any magnitude)."""
    sh = shares(stats, fids)
    return max(abs(sh.get(r, {}).get(k, 0.0) - ks[k]) / ks[k]
               for r, ks in ref.items() for k in KEYS)


def pin(stats, target_set: list, pool: list, ref, lo, hi, tol, rng,
        max_iter=2000):
    """Swap fires between `target_set` and `pool` (same region) until the
    persistence estimate of `target_set` is in [lo, hi]. Both lists mutate."""
    for it in range(max_iter):
        ap = ap_of(stats, target_set)
        if lo <= ap <= hi:
            return ap, it
        going_down = ap > hi
        # candidates: out = most/least overlapping in target, in = the other
        # extreme in the pool, same region, tried in order
        outs = sorted(target_set, key=lambda f: overlap(stats, f),
                      reverse=going_down)
        # a swap may not push the imbalance above tol -- or above where it
        # already is, so an unbalanced start can only get better
        cap = max(tol, imbalance(stats, target_set, ref))
        moved = False
        for fo in outs:
            r = stats[fo]["region"]
            ins = sorted((f for f in pool if stats[f]["region"] == r),
                         key=lambda f: overlap(stats, f), reverse=not going_down)
            for fi in ins[:60]:
                trial = [f for f in target_set if f != fo] + [fi]
                new = ap_of(stats, trial)
                if (new < ap if going_down else new > ap) and \
                        imbalance(stats, trial, ref) <= cap:
                    target_set.remove(fo); target_set.append(fi)
                    pool.remove(fi); pool.append(fo)
                    moved = True
                    break
            if moved:
                break
        if not moved:
            print(f"    sin swaps que mejoren: se queda en AP~{ap:.3f} "
                  f"(objetivo [{lo},{hi}], tol {tol})")
            return ap, it
    return ap_of(stats, target_set), max_iter


def build(stats, seed, lo, hi, tol, also_val):
    rng = np.random.default_rng(seed)
    fids = sorted(stats)
    ref = shares(stats, fids)
    by_region = defaultdict(list)
    for f in fids:
        by_region[stats[f]["region"]].append(f)
    tr, va, te = [], [], []
    for r in sorted(by_region):
        ids = sorted(by_region[r]); rng.shuffle(ids)
        n = len(ids); a = int(round(n * FRACS[0])); b = int(round(n * FRACS[1]))
        tr += ids[:a]; va += ids[a:a + b]; te += ids[a + b:]
    ap0 = ap_of(stats, te)
    if lo is None:
        ap_te, it_te, ap_va, it_va = ap0, 0, ap_of(stats, va), 0
    else:
        ap_te, it_te = pin(stats, te, tr, ref, lo, hi, tol, rng)
        ap_va, it_va = (pin(stats, va, tr, ref, lo, hi, tol, rng) if also_val
                        else (ap_of(stats, va), 0))
    return dict(train=sorted(tr), val=sorted(va), test=sorted(te),
                ap=dict(train=ap_of(stats, tr), val=ap_va, test=ap_te),
                test_ap_before=ap0, swaps=dict(test=it_te, val=it_va))


# file-name slug of each region label of dataset/curate.py:region_of. "Chile"
# is the label the curation gives the WHOLE southern hemisphere (Chile,
# Argentina, Bolivia; 59 / 254 / 140 events in v7): the slug says so.
REGION_SLUG = {"Chile": "southam", "US": "us", "S.Europe": "seurope"}


def build_holdout(stats, region: str, seed: int) -> dict:
    """Leave-one-region-out: Test = all of `region`, Train/Val = the rest.

    Val is drawn at random from the OTHER regions, stratified per region,
    with the train:val ratio of the shipped splits (FRACS[0]:FRACS[1]), so a
    model is selected on the distribution it was trained on and the held-out
    region is seen exactly once, at test time.
    """
    if region not in REGION_SLUG:
        raise SystemExit(f"region {region!r}: debe ser una de {sorted(REGION_SLUG)}")
    rng = np.random.default_rng(seed)
    fids = sorted(stats)
    by_region = defaultdict(list)
    for f in fids:
        by_region[stats[f]["region"]].append(f)
    if region not in by_region:
        raise SystemExit(f"region {region!r}: sin eventos en las estadisticas")
    val_frac = FRACS[1] / (FRACS[0] + FRACS[1])
    tr, va = [], []
    for r in sorted(by_region):
        if r == region:
            continue
        ids = sorted(by_region[r]); rng.shuffle(ids)
        b = int(round(len(ids) * val_frac))
        va += ids[:b]; tr += ids[b:]
    te = sorted(by_region[region])
    # the three sets partition the dataset, and test is the whole region
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert sorted(tr + va + te) == fids
    assert all(stats[f]["region"] == region for f in te)
    assert all(stats[f]["region"] != region for f in tr + va)
    return dict(train=sorted(tr), val=sorted(va), test=te,
                ap=dict(train=ap_of(stats, tr), val=ap_of(stats, va),
                        test=ap_of(stats, te)),
                test_ap_before=ap_of(stats, te), swaps=dict(test=0, val=0))


def report(stats, split):
    ref = shares(stats, sorted(stats))
    print(f"    {'':10s}" + "".join(f"{r:>22s}" for r in sorted(ref)))
    for name in ("train", "val", "test"):
        sh = shares(stats, split[name])
        cells = []
        for r in sorted(ref):
            s = sh.get(r, {})
            cells.append(f"{s.get('fires', 0):.2f}/{s.get('days', 0):.2f}/"
                         f"{s.get('burned', 0):.2f}")
        print(f"    {name:10s}" + "".join(f"{c:>22s}" for c in cells)
              + f"   n={len(split[name]):4d}  AP~{split['ap'][name]:.3f}")
    print(f"    {'dataset':10s}" + "".join(
        f"{ref[r]['fires']:.2f}/{ref[r]['days']:.2f}/{ref[r]['burned']:.2f}".rjust(22)
        for r in sorted(ref)) + "      (fires/days/burned)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 7, 23])
    ap.add_argument("--target", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="opcional: fijar la persistencia estimada del test en [LO, HI]")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="desvio relativo permitido por region y magnitud al fijar")
    ap.add_argument("--also-val", action="store_true")
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--stats", type=Path, default=SPLIT_DIR / "fire_stats_v7.json",
                    help="cache de las estadisticas por evento (se recalcula si falta)")
    ap.add_argument("--no-write", action="store_true", help="solo estimar")
    ap.add_argument("--holdout", nargs="+", metavar="REGION", default=None,
                    help="leave-one-region-out: Test = toda la region, Train/Val = "
                         "el resto (experiments/exp6.py). Regiones: "
                         + ", ".join(sorted(REGION_SLUG)))
    a = ap.parse_args()

    if a.stats.exists():
        stats = json.loads(a.stats.read_text())
    else:
        stats = fire_stats(AFSEQ)
        a.stats.write_text(json.dumps(stats))
    fids = sorted(stats)
    print(f"{len(fids)} eventos en {AFSEQ}; persistencia estimada del conjunto: "
          f"{ap_of(stats, fids):.3f}")
    if a.holdout:
        if a.target:
            raise SystemExit("--holdout no se combina con --target: nada se fija")
        seed = a.seeds[0]
        for region in a.holdout:
            sp = build_holdout(stats, region, seed)
            p = SPLIT_DIR / f"splits_{a.tag}_loro_{REGION_SLUG[region]}.txt"
            if not a.no_write:
                p.write_text("\n".join(["Train"] + sp["train"] + ["", "Val"] + sp["val"]
                                       + ["", "Test"] + sp["test"]) + "\n")
            print(f"\n  holdout {region} (s{seed}): test = {len(sp['test'])} eventos, "
                  f"AP~ {sp['ap']['test']:.3f}  -> {p}")
            report(stats, sp)
        return
    lo, hi = a.target if a.target else (None, None)
    for seed in a.seeds:
        sp = build(stats, seed, lo, hi, a.tol, a.also_val)
        p = SPLIT_DIR / f"splits_{a.tag}_s{seed}.txt"
        if not a.no_write:
            p.write_text("\n".join(["Train"] + sp["train"] + ["", "Val"] + sp["val"]
                                   + ["", "Test"] + sp["test"]) + "\n")
        print(f"\n  s{seed}: test AP~ {sp['test_ap_before']:.3f} -> "
              f"{sp['ap']['test']:.3f} en {sp['swaps']['test']} swaps"
              + (f"; val en {sp['swaps']['val']}" if a.also_val else "")
              + f"  -> {p}")
        report(stats, sp)


if __name__ == "__main__":
    main()
