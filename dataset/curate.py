#!/usr/bin/env python3
"""Curation of a built AFSeq tree: what leaves the dataset, and why.

    python dataset/build_afseq.py --out <tree> --phase curate      # normal way
    python dataset/curate.py --out <tree> --active <masks> [--dry-run]

Runs last in `build_afseq.py --phase all`, so the tree the download produces
is already the dataset the paper uses; `dataset/make_splits.py` only splits
what is left. Two rules, both decided on 2026-08-25 with the numbers below:

1. DUPLICATED EVENTS. Two ids with the same footprint, raster shape and the
   exact list of mask dates are the same fire twice (the expansion campaign
   re-downloaded events that already existed: 33 pairs in dataset_v6, 25 of
   them with a byte-identical climate.tif). The lower id stays. Left in, they
   land on different sides of a split: 14-18 of the 33 pairs did, i.e. the
   same fire in train and test.

2. THE MOST PERSISTENT US FIRES. AFSeq's pooled persistence AP (0.252 with
   everything) is not a property of the target: it is a handful of slow US
   fronts, 6-30 days long, whose day-to-day masks overlap 0.55-0.80. Per
   region the floor was US 0.307 against Chile 0.174 and S.Europe 0.170.
   Removing the 15% most persistent US events (74) puts the three regions at
   the same floor (0.168 / 0.173 / 0.156) and the dataset at 0.167, the level
   of WSTS+ under the same sampler (0.200). Chile and S.Europe are untouched.
   Persistence is ranked by the Jaccard of consecutive daily masks counted in
   random 64-px crops, four per day, the way the loader scores it; the count
   is deterministic (days sorted, one RNG per event seeded by its id).

Nothing is deleted: excluded events move to `<tree>_excluded/<reason>/<id>`
(and the same for the mask tree), and `<tree>/excluded.json` records every id
with its reason and the parameters. `--dry-run` only prints the lists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paths import AFSEQ, AFSEQ_ACTIVE  # noqa: E402

DROP_DEFAULT = {"US": 0.15}


def events_of(tree: Path) -> list[str]:
    return sorted(p.name for p in Path(tree).iterdir()
                  if p.is_dir() and p.name.startswith("id_"))


def region_of(tree: Path, fid: str) -> str:
    """Chile / US / S.Europe from the event centroid (info.geojson)."""
    j = json.loads((Path(tree) / fid / "info.geojson").read_text())
    geom = j["features"][0]["geometry"] if "features" in j else j["geometry"]
    c = geom["coordinates"]
    while isinstance(c[0], list):
        c = c[0]
    lon, lat = float(c[0]), float(c[1])
    if lat < 0:
        return "Chile"
    return "US" if lon < -30 else "S.Europe"


def _masks(active: Path, fid: str) -> dict:
    out = {}
    for p in sorted((Path(active) / fid).glob("*.tif")):
        with rasterio.open(p) as r:
            out[p.stem] = r.read(1) > 0.5
    return out


# ------------------------------------------------------------ rule 1: twins
def duplicate_events(active: Path) -> dict:
    """{duplicate id: original id}; same footprint, shape and mask dates."""
    groups = {}
    for fid in events_of(active):
        ds = tuple(sorted(p.stem for p in (Path(active) / fid).glob("*.tif")))
        if not ds:
            continue
        with rasterio.open(Path(active) / fid / f"{ds[0]}.tif") as r:
            key = (tuple(round(b, 5) for b in r.bounds), r.shape, ds)
        groups.setdefault(key, []).append(fid)
    return {d: ids[0] for ids in groups.values() for d in ids[1:]}


# ------------------------------------------------- rule 2: slow US fronts
def fire_stats(tree: Path, active: Path, fids=None, patch: int = 64,
               crops: int = 4) -> dict:
    """Per event: TP/FP/FN/P/N of "tomorrow = today", days, burned, region.

    Counted inside random `patch`-sized crops, `crops` per sampled day (fire
    on t, mask and covariates on t+1), because that is what gets scored.
    """
    tree, active = Path(tree), Path(active)
    out = {}
    for fid in (fids or events_of(tree)):
        rng = np.random.default_rng(zlib.crc32(fid.encode()))
        masks = _masks(active, fid)
        TP = FP = FN = P = N = days = 0
        for d in sorted(masks):
            nd = (datetime.strptime(d, "%Y-%m-%d")
                  + timedelta(days=1)).strftime("%Y-%m-%d")
            m = masks[d]
            if nd not in masks or not (tree / fid / d).is_dir() \
                    or not (tree / fid / nd / "fire-mask.tif").exists() \
                    or not m.any():
                continue
            y = masks[nd]
            days += 1
            th, tw = max(m.shape[0], patch), max(m.shape[1], patch)
            mp = np.zeros((th, tw), bool); mp[:m.shape[0], :m.shape[1]] = m
            yp = np.zeros((th, tw), bool); yp[:y.shape[0], :y.shape[1]] = y
            for _ in range(crops):
                t = rng.integers(0, th - patch + 1)
                l = rng.integers(0, tw - patch + 1)
                mc, yc = mp[t:t + patch, l:l + patch], yp[t:t + patch, l:l + patch]
                TP += int((mc & yc).sum()); FP += int((mc & ~yc).sum())
                FN += int((~mc & yc).sum()); P += int(yc.sum()); N += yc.size
        burned = int(np.any(np.stack(list(masks.values())), 0).sum()) if masks else 0
        out[fid] = dict(days=days, TP=TP, FP=FP, FN=FN, P=P, N=N, burned=burned,
                        region=region_of(tree, fid))
    return out


def overlap(stats: dict, fid: str) -> float:
    """Jaccard of consecutive-day masks: what persistence has to work with."""
    s = stats[fid]
    return s["TP"] / max(s["TP"] + s["FP"] + s["FN"], 1)


def most_persistent(stats: dict, drop: dict) -> dict:
    """{id: region} of the `drop[region]` most-overlapping events per region."""
    out = {}
    for region, frac in drop.items():
        ids = sorted((f for f in stats if stats[f]["region"] == region),
                     key=lambda f: overlap(stats, f), reverse=True)
        for f in ids[:int(round(len(ids) * frac))]:
            out[f] = region
    return out


def persistence_ap(stats: dict, fids) -> float:
    """Closed-form pooled AP of a binary score (validated against exp2)."""
    s = {k: sum(stats[f][k] for f in fids) for k in ("TP", "FP", "FN", "P", "N")}
    if s["TP"] + s["FN"] == 0:
        return 0.0
    r1 = s["TP"] / (s["TP"] + s["FN"])
    p1 = s["TP"] / max(s["TP"] + s["FP"], 1)
    return r1 * p1 + (1 - r1) * s["P"] / max(s["N"], 1)


# ---------------------------------------------------------------- driver
def curate(tree: Path, active: Path, drop: dict = DROP_DEFAULT,
           dry_run: bool = False) -> dict:
    tree, active = Path(tree), Path(active)
    before = events_of(tree)
    dups = duplicate_events(active)
    kept = [f for f in before if f not in dups]
    stats = fire_stats(tree, active, kept)
    slow = most_persistent(stats, drop)
    excluded = {**{d: dict(reason="duplicate", of=o) for d, o in dups.items()},
                **{f: dict(reason="persistent", region=r,
                           overlap=round(overlap(stats, f), 3))
                   for f, r in slow.items()}}
    left = [f for f in kept if f not in slow]
    by_region = {}
    for r in sorted({v["region"] for v in stats.values()}):
        ids = [f for f in left if stats[f]["region"] == r]
        by_region[r] = dict(events=len(ids), persistence=round(persistence_ap(stats, ids), 3))
    summary = dict(before=len(before), duplicates=len(dups), persistent=len(slow),
                   after=len(left), drop=drop, by_region=by_region,
                   persistence=round(persistence_ap(stats, left), 3))
    print(json.dumps(summary, indent=1))
    if dry_run:
        for reason in ("duplicate", "persistent"):
            ids = sorted(f for f, w in excluded.items() if w["reason"] == reason)
            print(f"{reason} ({len(ids)}): " + " ".join(ids))
        return dict(summary=summary, excluded=excluded)

    for fid, why in sorted(excluded.items()):
        for root in (tree, active):
            src = root / fid
            if not src.exists() and not src.is_symlink():
                continue
            dst = Path(str(root).rstrip("/") + "_excluded") / why["reason"] / fid
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    (tree / "excluded.json").write_text(json.dumps(
        dict(date=datetime.now().strftime("%F"), summary=summary,
             events=excluded), indent=1))
    print(f"{len(excluded)} eventos movidos a {tree}_excluded/ y "
          f"{active}_excluded/; registro en {tree}/excluded.json")
    return dict(summary=summary, excluded=excluded)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=AFSEQ, help="arbol del dataset")
    ap.add_argument("--active", type=Path, default=AFSEQ_ACTIVE,
                    help="arbol de mascaras activas diarias")
    ap.add_argument("--drop", nargs="+", metavar="REGION=FRAC",
                    default=[f"{k}={v}" for k, v in DROP_DEFAULT.items()])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    drop = {k: float(v) for k, v in (d.split("=") for d in a.drop)}
    curate(a.out, a.active, drop, a.dry_run)


if __name__ == "__main__":
    main()
