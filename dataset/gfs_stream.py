"""Streaming GFS pipeline: download one date, extract the events that use it,
then delete the GRIB. It never holds more than a few hundred MB on disk.

The raw archive would be about 32 GB. What is actually needed is the per-event
crops, roughly 6 KB per day, because one fire window covers only a handful of
cells of the 0.25 degree grid.

    python dataset/gfs_stream.py [--keep-grib] [--workers N]
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402
from fetch_gfs import grab, DAILY, PRECIP_HOURS          # noqa: E402
import gfs_to_events as G                                 # noqa: E402

DATA = paths.AFSEQ
GFS = paths.DATA / "gfs"
OUT = paths.DATA / "gfs_events"
HOURS = sorted(set(DAILY) | set(PRECIP_HOURS))


def day_map():
    """date -> [events that have it as their day t]."""
    m = defaultdict(list)
    for p in DATA.glob("id_*/*/active-fire.tif"):
        m[p.parent.name].append(p.parent.parent.name)
    return {k: sorted(set(v)) for k, v in sorted(m.items())}


def extract_one(args):
    """One event on one date: the seven channels cropped on the native grid."""
    fid, date = args
    dst = OUT / fid / date / "gfs.npz"
    if dst.exists():
        return True
    try:
        info = json.loads((DATA / fid / "info.geojson").read_text())["properties"]
        lon0, lat0, lon1, lat1 = info["window"]
        box = (lon0 - G.PAD, lat0 - G.PAD, lon1 + G.PAD, lat1 + G.PAD)
        dd = GFS / date.replace("-", "")
        fc = []
        for h in G.HOURS:
            p = dd / f"f{h:03d}.grib2"
            if not p.exists():
                return False
            s = G.seven(G._open(p), box)
            if s is None:
                return False
            pr = G.day_precip(dd, box, h)
            if pr is not None and pr.shape == s[0].shape:
                s[6] = pr
            fc.append(s)
        an = None
        p0 = dd / "f000.grib2"
        if p0.exists():
            an = G.seven(G._open(p0), box)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, fcst=np.stack(fc).astype(np.float32),
                            anl=(an if an is not None else np.zeros(0, np.float32)),
                            bbox=np.array(box, np.float64))
        return True
    except Exception as exc:
        print(f"  extract {fid}/{date}: {type(exc).__name__} {exc}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-grib", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dm = day_map()
    dates = sorted(dm)
    if a.limit:
        dates = dates[:a.limit]
    print(f"{len(dates)} dates, {sum(len(v) for v in dm.values())} event-days", flush=True)

    t0 = time.time()
    n_ev = n_fail = 0
    for i, date in enumerate(dates, 1):
        d8 = date.replace("-", "")
        # 1. download whatever is still missing for this date
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda h: grab(d8, h), HOURS))
        # 2. extract every event that uses it
        jobs = [(fid, date) for fid in dm[date]]
        with ProcessPoolExecutor(max_workers=min(a.workers, 8)) as ex:
            for ok in ex.map(extract_one, jobs, chunksize=4):
                n_ev += bool(ok); n_fail += (not ok)
        # 3. throw the GRIB away
        if not a.keep_grib:
            shutil.rmtree(GFS / d8, ignore_errors=True)
        if i % 25 == 0 or i == len(dates):
            el = (time.time() - t0) / 60
            print(f"  {i}/{len(dates)} dates  {el:.1f} min  "
                  f"event-days ok={n_ev} fail={n_fail}", flush=True)
    print(f"done ok={n_ev} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
