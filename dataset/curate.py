#!/usr/bin/env python3
"""Curation of a built AFSeq tree: what leaves the dataset, and why.

    python dataset/build_afseq.py --out <tree> --phase curate      # normal way
    python dataset/curate.py --out <tree> [--dry-run]

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
   64-px crops centred on the day-t fire with jitter, four per day, the way
   the loader samples; the count
   is deterministic (days sorted, one RNG per event seeded by its id).

Nothing is deleted: excluded events move to `<tree>_excluded/<reason>/<id>`,
and `<tree>/excluded.json` records every id with its reason and the
parameters. A tree that already has excluded.json is left alone (the rules
are not idempotent: a second pass would take 15% of what is left) unless
`--force`. `--dry-run` only prints the lists. Masks are read from
`<tree>/<id>/<date>/active-fire.tif`, the product build_afseq.py writes.
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
from paths import AFSEQ  # noqa: E402

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


def _masks(tree: Path, fid: str) -> dict:
    """{date: bool mask} from <tree>/<fid>/<date>/active-fire.tif."""
    out = {}
    for p in sorted((Path(tree) / fid).glob("*/active-fire.tif")):
        with rasterio.open(p) as r:
            out[p.parent.name] = r.read(1) > 0.5
    return out


# ------------------------------------------------------------ rule 1: twins
def duplicate_events(tree: Path) -> dict:
    """{duplicate id: original id}; same footprint, shape and mask dates."""
    groups = {}
    for fid in events_of(tree):
        ds = tuple(sorted(p.parent.name
                          for p in (Path(tree) / fid).glob("*/active-fire.tif")))
        if not ds:
            continue
        with rasterio.open(Path(tree) / fid / ds[0] / "active-fire.tif") as r:
            key = (tuple(round(b, 5) for b in r.bounds), r.shape, ds)
        groups.setdefault(key, []).append(fid)
    return {d: ids[0] for ids in groups.values() for d in ids[1:]}


# ------------------------------------------------- rule 2: slow US fronts
def fire_stats(tree: Path, fids=None, patch: int = 64, crops: int = 4) -> dict:
    """Per event: TP/FP/FN/P/N of "tomorrow = today", days, burned, region.

    Counted inside random `patch`-sized crops, `crops` per sampled day (fire
    on t, mask and covariates on t+1), because that is what gets scored.
    """
    tree = Path(tree)
    out = {}
    for fid in (fids or events_of(tree)):
        rng = np.random.default_rng(zlib.crc32(fid.encode()))
        masks = _masks(tree, fid)
        TP = FP = FN = P = N = days = 0
        for d in sorted(masks):
            nd = (datetime.strptime(d, "%Y-%m-%d")
                  + timedelta(days=1)).strftime("%Y-%m-%d")
            m = masks[d]
            if nd not in masks or not m.any():      # sampler: fire on t, mask on t+1
                continue
            y = masks[nd]
            days += 1
            th, tw = max(m.shape[0], patch), max(m.shape[1], patch)
            mp = np.zeros((th, tw), bool); mp[:m.shape[0], :m.shape[1]] = m
            yp = np.zeros((th, tw), bool); yp[:y.shape[0], :y.shape[1]] = y
            ys, xs = np.where(mp)      # crops centred on the day-t fire, as the loader
            cy = (int(ys.min()) + int(ys.max())) // 2
            cx = (int(xs.min()) + int(xs.max())) // 2
            j = patch // 4
            for _ in range(crops):
                t = int(np.clip(cy - patch // 2 + rng.integers(-j, j + 1), 0, th - patch))
                l = int(np.clip(cx - patch // 2 + rng.integers(-j, j + 1), 0, tw - patch))
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
def curate(tree: Path, drop: dict = DROP_DEFAULT, dry_run: bool = False,
           force: bool = False) -> dict:
    tree = Path(tree)
    if (tree / "excluded.json").exists() and not force and not dry_run:
        print(f"{tree} ya esta curado ({tree / 'excluded.json'}); --force para repetir")
        return json.loads((tree / "excluded.json").read_text())
    before = events_of(tree)
    dups = duplicate_events(tree)
    kept = [f for f in before if f not in dups]
    stats = fire_stats(tree, kept)
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
        src = tree / fid
        dst = Path(str(tree).rstrip("/") + "_excluded") / why["reason"] / fid
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    (tree / "excluded.json").write_text(json.dumps(
        dict(date=datetime.now().strftime("%F"), summary=summary,
             events=excluded), indent=1))
    print(f"{len(excluded)} eventos movidos a {tree}_excluded/; "
          f"registro en {tree}/excluded.json")
    return dict(summary=summary, excluded=excluded)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=AFSEQ, help="arbol del dataset")
    ap.add_argument("--drop", nargs="+", metavar="REGION=FRAC",
                    default=[f"{k}={v}" for k, v in DROP_DEFAULT.items()])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    drop = {k: float(v) for k, v in (d.split("=") for d in a.drop)}
    curate(a.out, drop, a.dry_run, a.force)


if __name__ == "__main__":
    main()
