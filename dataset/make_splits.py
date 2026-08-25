#!/usr/bin/env python3
"""The AFSeq v6 splits: region-stratified, with the persistence floor of WSTS+.

    python dataset/make_splits.py --seeds 5 7 23 --drop US=0.15 --target 0.16 0.18
    -> dataset/splits/splits_v6_s{seed}.txt          (what ships with the repo)

WHAT V6 IS. dataset_v6 holds 1,301 events; the v6 SPLITS use 1,194 of them:
33 are duplicates of another event (see duplicate_events) and the 15% most
persistent fires of the US are left out of every split (74 events, slow fronts of 6-30 days with day-to-day mask overlap 0.55-0.80). With
them in, the US floor was 0.307 against 0.174 in Chile and 0.170 in S.Europe
and the pooled floor 0.252; without them the three regions sit at 0.167 /
0.174 / 0.170, the measured test persistence is 0.167 +- 0.007 over the three
seeds and train is 0.164-0.170: same regime everywhere. The US share of burned
pixels goes from 52% to 45%. Chile and S.Europe are untouched.

WHY IT EXISTS. Under the exp2 protocol AFSeq's persistence floor is 0.252 and
WSTS+'s, sampled the same way (fire on day t), is 0.200. A reader who compares
the two tables sees a higher floor on our side and wonders whether the models'
0.47 is inertia. This split asks the question directly: take a test set whose
raw persistence sits where WSTS+'s does, keep every split balanced across
regions, and see what the models do. That is the benchmark's definition, so it lives
in the release next to the loader.

WHAT IT DOES NOT HIDE. The test set is chosen BY the metric: fires whose
consecutive daily masks overlap less. That makes the test distribution differ
from train's (train's persistence goes UP, the fires that leave the test set
have to go somewhere), and the paper has to say so if this split is ever used
for a reported number. `--also-val` pins the validation set too, so checkpoint
selection sees the same regime as the test. Everything else about the sampling
is v4's: fires are the unit, 70/15/15, no leakage constraint between splits
(the audit measured that at -0.016 AP).

HOW. Per fire, from the decoded cache (dataset/afseq_cache.py), over the days
the exp2 sampler would use (fire on t, an active mask on t+1): TP/FP/FN of
"tomorrow = today" on the full masks, positives and pixels. Pooled persistence
AP has a closed form for a binary score,

    AP = R1 * P1 + (1 - R1) * prevalence,   R1 = TP/(TP+FN),  P1 = TP/(TP+FP)

counted inside random 64-px crops as the loader scores them (see fire_stats).
So: start from a
region-stratified random split, then swap test fires for train fires of the
SAME region, highest-overlap out and lowest-overlap in, as long as every
region keeps its share of fires, sampled days and burned pixels in the test
within `--tol` (relative) of its share in the whole dataset. Stop when the
estimate is inside the target window; the real number is then measured with

    python experiments/exp2.py --table --rows persistence
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
from paths import AFSEQ, AFSEQ_ACTIVE, AFSEQ_CACHE, SPLITS_DIR  # noqa: E402

SPLIT_DIR = SPLITS_DIR                 # dataset/splits, shipped with the code
FRACS = (0.70, 0.15, 0.15)              # v4's
KEYS = ("fires", "days", "burned")


def region_of(fid: str) -> str:
    """Chile / US / S.Europe from the event centroid, as scan_event_quality."""
    g = AFSEQ / fid / "info.geojson"
    j = json.loads(g.read_text())
    geom = j["features"][0]["geometry"] if "features" in j else j["geometry"]
    c = geom["coordinates"]
    while isinstance(c[0], list):
        c = c[0]
    lon, lat = float(c[0]), float(c[1])
    if lat < 0:
        return "Chile"
    return "US" if lon < -30 else "S.Europe"


def fire_stats(cache: Path = AFSEQ_CACHE, patch: int = 64, crops: int = 4) -> dict:
    """Per fire: TP/FP/FN/P/N of persistence over the sampler's days.

    Counted inside random `patch`-sized crops, `crops` per day, because that is
    what gets scored: the loader pads a scene to at least `patch` and takes one
    random crop per sample, so a big fire contributes 64x64 windows that may
    miss the front entirely. Counting on full masks overestimated the pinned
    splits by 0.02-0.045 (0.176/0.185/0.185 estimated, 0.155/0.183/0.140
    measured); the crop-level count is what the closed form needs.
    """
    from datetime import datetime, timedelta
    rng = np.random.default_rng(0)
    out = {}
    for fid in sorted(p.name for p in cache.iterdir() if p.is_dir()):
        feat = {p.stem for p in (cache / fid).glob("*.npy")}
        masks = {p.stem: np.load(p) > 0.5
                 for p in (cache / fid / "active").glob("*.npy")}
        TP = FP = FN = P = N = days = 0
        for d, m in masks.items():
            nd = (datetime.strptime(d, "%Y-%m-%d")
                  + timedelta(days=1)).strftime("%Y-%m-%d")
            if nd not in masks or d not in feat or not m.any():
                continue
            y = masks[nd]
            days += 1
            th, tw = max(m.shape[0], patch), max(m.shape[1], patch)
            mp = np.zeros((th, tw), bool); mp[:m.shape[0], :m.shape[1]] = m
            yp = np.zeros((th, tw), bool); yp[:y.shape[0], :y.shape[1]] = y
            for _ in range(crops):
                t = rng.integers(0, th - patch + 1); l = rng.integers(0, tw - patch + 1)
                mc, yc = mp[t:t + patch, l:l + patch], yp[t:t + patch, l:l + patch]
                TP += int((mc & yc).sum()); FP += int((mc & ~yc).sum())
                FN += int((~mc & yc).sum()); P += int(yc.sum()); N += yc.size
        burned = int(np.any(np.stack(list(masks.values())), 0).sum()) if masks else 0
        out[fid] = dict(days=days, TP=TP, FP=FP, FN=FN, P=P, N=N, burned=burned,
                        region=region_of(fid))
    return out


def ap_of(stats: dict, fids) -> float:
    s = {k: sum(stats[f][k] for f in fids) for k in ("TP", "FP", "FN", "P", "N")}
    if s["TP"] + s["FN"] == 0:
        return 0.0
    r1 = s["TP"] / (s["TP"] + s["FN"])
    p1 = s["TP"] / max(s["TP"] + s["FP"], 1)
    return r1 * p1 + (1 - r1) * s["P"] / max(s["N"], 1)


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


def overlap(stats, f) -> float:
    """Jaccard of consecutive-day masks: what persistence has to work with."""
    s = stats[f]
    return s["TP"] / max(s["TP"] + s["FP"] + s["FN"], 1)


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


def duplicate_events(active: Path = AFSEQ_ACTIVE) -> dict:
    """{duplicate id: original id} for events that are the same fire twice.

    Found on 2026-08-25 while re-scanning the teaser candidates: 33 pairs of
    dataset_v6 share footprint, raster shape and the exact list of mask dates
    (25 of them have byte-identical climate.tif too) -- the expansion
    campaign re-downloaded events that were already there under a new id
    (id_2xxx / id_3xxx). On the first v6 splits 14-18 of those pairs sat in
    different splits: the same fire in train and test. The lower id (the
    original) stays; the other is excluded from every split.
    """
    import rasterio
    groups = {}
    for ev in sorted(p for p in Path(active).iterdir() if p.is_dir()):
        ds = tuple(sorted(p.stem for p in ev.glob("*.tif")))
        if not ds:
            continue
        with rasterio.open(ev / f"{ds[0]}.tif") as r:
            key = (tuple(round(b, 5) for b in r.bounds), r.shape, ds)
        groups.setdefault(key, []).append(ev.name)
    return {d: ids[0] for ids in groups.values() for d in ids[1:]}


def drop_most_persistent(stats, frac) -> dict:
    """Remove the most-overlapping fires from the dataset.

    `frac` is a number (the same fraction of every region, so the region shares
    stay what they were) or a {region: fraction} dict (e.g. {"US": 0.2}: only
    that region loses fires; its share of the dataset shrinks accordingly).

    The other lever. Swapping only moves persistence from the test to the
    train; dropping lowers it everywhere, so the three splits keep the same
    regime. US is where AFSeq's persistence lives (0.31 against 0.17 in Chile
    and S.Europe): its fires are long slow fronts of 6-30 days.
    """
    regions = sorted({v["region"] for v in stats.values()})
    fracs = frac if isinstance(frac, dict) else {r: frac for r in regions}
    keep = {}
    for r in regions:
        ids = sorted((f for f in stats if stats[f]["region"] == r),
                     key=lambda f: overlap(stats, f))
        n_drop = int(round(len(ids) * fracs.get(r, 0.0)))
        for f in ids[:len(ids) - n_drop]:
            keep[f] = stats[f]
    return keep


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
    ap_te, it_te = pin(stats, te, tr, ref, lo, hi, tol, rng)
    ap_va, it_va = (pin(stats, va, tr, ref, lo, hi, tol, rng) if also_val
                    else (ap_of(stats, va), 0))
    return dict(train=sorted(tr), val=sorted(va), test=sorted(te),
                ap=dict(train=ap_of(stats, tr), val=ap_va, test=ap_te),
                test_ap_before=ap0, swaps=dict(test=it_te, val=it_va))


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
    ap.add_argument("--target", type=float, nargs=2, default=(0.18, 0.19),
                    help="ventana para la estimacion (conteo en crops de 64)")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="desvio relativo permitido por region y magnitud")
    ap.add_argument("--also-val", action="store_true")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--stats", type=Path, default=SPLIT_DIR / "fire_stats_v6_crop64.json")
    ap.add_argument("--drop-frac", type=float, default=0.0,
                    help="quitar del dataset esta fraccion de fuegos por region, "
                         "los de mas solape dia a dia (baja la persistencia de "
                         "TODOS los splits, no solo del test)")
    ap.add_argument("--drop", nargs="+", metavar="REGION=FRAC", default=[],
                    help="idem pero solo en esas regiones, p.ej. --drop US=0.2")
    ap.add_argument("--no-write", action="store_true", help="solo estimar")
    a = ap.parse_args()

    if a.stats.exists():
        stats = json.loads(a.stats.read_text())
    else:
        stats = fire_stats()
        a.stats.write_text(json.dumps(stats))
    n_all = len(stats)
    dups = duplicate_events()
    stats = {f: v for f, v in stats.items() if f not in dups}
    print(f"{len(dups)} eventos duplicados excluidos (se queda el id menor): "
          + ", ".join(f"{d}={o}" for d, o in sorted(dups.items())[:5]) + ", ...")
    (SPLIT_DIR / "duplicates_v6.json").write_text(json.dumps(dups, indent=1))
    drop = ({k: float(v) for k, v in (d.split("=") for d in a.drop)}
            if a.drop else a.drop_frac)
    stats = drop_most_persistent(stats, drop)
    fids = sorted(stats)
    print(f"{len(stats)} fuegos" + (f" (de {n_all}, drop {drop})" if drop else "")
          + f"; persistencia estimada del dataset entero: {ap_of(stats, fids):.3f}")
    # feasibility: the 15% least-overlapping fires of every region
    low = []
    for r in sorted({stats[f]["region"] for f in fids}):
        ids = sorted((f for f in fids if stats[f]["region"] == r),
                     key=lambda f: overlap(stats, f))
        low += ids[:int(round(len(ids) * FRACS[2]))]
    print(f"cota: el 15% de fuegos con menos solape por region da AP~"
          f"{ap_of(stats, low):.3f} (desbalance {imbalance(stats, low, shares(stats, fids)):.2f})")
    for seed in a.seeds:
        sp = build(stats, seed, a.target[0], a.target[1], a.tol, a.also_val)
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
