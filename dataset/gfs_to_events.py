"""Turn the raw GFS archive into per-event, per-day covariates.

For every day t with fire in an event this takes the 00Z cycle of THAT day and
its f024 to f120 horizons, valid at t+1 to t+5, crops the event window and
stores the seven channels in the units normalize_climate expects:

    temp  C     (GFS TMP:surface,  K - 273.15)
    dew   C     (DPT:2m,           K - 273.15)
    rh    %     (RH:2m)
    ws    m/s   (|UGRD,VGRD:10m|)
    wdir  deg   (direction the wind comes from, 0 = north)
    pres  hPa   (PRES:surface / 100)
    precp mm    (APCP summed over the four 6 h sub-steps of the day)

Everything is stored on the native 0.25 degree grid rather than the 200 m one.
An event window covers only a few GFS cells, so a crop weighs kilobytes and the
interpolation onto the event grid happens on the fly in the loader. Storing it
already reprojected would cost about 127 GB; this way it costs about 0.2 GB.

    <out>/<id>/<date>/gfs.npz   fcst (5,7,h,w) float32 + f000 (7,h,w) + bbox
"""
from __future__ import annotations
import argparse, json, sys, tempfile, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

warnings.filterwarnings("ignore")
GFS = paths.DATA / "gfs"
DATA = paths.AFSEQ
PAD = 0.75          # degrees of margin, three GFS cells on every side
HOURS = (24, 48, 72, 96, 120)


def _open(path: Path):
    import xarray as xr
    out = {}
    for filt in ({"typeOfLevel": "surface"},
                 {"typeOfLevel": "heightAboveGround", "level": 2},
                 {"typeOfLevel": "heightAboveGround", "level": 10}):
        try:
            ds = xr.open_dataset(path, engine="cfgrib",
                                 backend_kwargs={"filter_by_keys": filt, "indexpath": ""})
            for v in ds.data_vars:
                out[v] = ds[v]
        except Exception:
            pass
    return out


def _crop(da, lon0, lat0, lon1, lat1):
    """Crop to a bounding box. GFS longitudes run from 0 to 360."""
    lon = da.longitude.values
    L0, L1 = (lon0 % 360), (lon1 % 360)
    lat_sel = (da.latitude >= lat0) & (da.latitude <= lat1)
    lon_sel = (lon >= L0) & (lon <= L1) if L0 <= L1 else ((lon >= L0) | (lon <= L1))
    return da.isel(latitude=np.where(lat_sel)[0], longitude=np.where(lon_sel)[0])


def seven(msgs, box):
    """The seven channels, in the units the loader expects."""
    g = {k: _crop(v, *box).values.astype(np.float32) for k, v in msgs.items()
         if k in ("t", "d2m", "r2", "u10", "v10", "sp", "tp")}
    if not {"t", "d2m", "r2", "u10", "v10", "sp"} <= set(g):
        return None
    u, v = g["u10"], g["v10"]
    ws = np.hypot(u, v)
    wdir = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0     # where it comes from
    return np.stack([g["t"] - 273.15, g["d2m"] - 273.15, g["r2"], ws, wdir,
                     g["sp"] / 100.0, g.get("tp", np.zeros_like(u))])


def day_precip(date_dir: Path, box, h_end: int):
    """Daily precipitation: the four 6 h accumulations that fall inside the day
    ending at h_end, summed."""
    tot = None
    for h in range(h_end - 18, h_end + 1, 6):
        p = date_dir / f"f{h:03d}.grib2"
        if not p.exists():
            continue
        m = _open(p)
        if "tp" not in m:
            continue
        a = _crop(m["tp"], *box).values.astype(np.float32)
        tot = a if tot is None else tot + a
    return tot


def one_event(fid: str, out_root: str):
    out_root = Path(out_root)
    info = json.loads((DATA / fid / "info.geojson").read_text())["properties"]
    lon0, lat0, lon1, lat1 = info["window"]
    box = (lon0 - PAD, lat0 - PAD, lon1 + PAD, lat1 + PAD)
    days = sorted(p.parent.name for p in (DATA / fid).glob("*/active-fire.tif"))
    n_ok = 0
    for d in days:
        dst = out_root / fid / d / "gfs.npz"
        if dst.exists():
            n_ok += 1
            continue
        dd = GFS / d.replace("-", "")
        if not dd.is_dir():
            continue
        try:
            fc = []
            for h in HOURS:
                p = dd / f"f{h:03d}.grib2"
                if not p.exists():
                    fc = []; break
                s = seven(_open(p), box)
                if s is None:
                    fc = []; break
                pr = day_precip(dd, box, h)
                if pr is not None and pr.shape == s[0].shape:
                    s[6] = pr
                fc.append(s)
            if len(fc) != len(HOURS):
                continue
            an = None
            p0 = dd / "f000.grib2"
            if p0.exists():
                an = seven(_open(p0), box)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, fcst=np.stack(fc).astype(np.float32),
                                anl=(an if an is not None else np.zeros(0, np.float32)),
                                bbox=np.array(box, np.float64))
            n_ok += 1
        except Exception as exc:
            print(f"  {fid}/{d}: {type(exc).__name__} {exc}", file=sys.stderr)
    return fid, n_ok, len(days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=paths.DATA / "gfs_events")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ids = sorted(p.name for p in DATA.glob("id_*") if p.is_dir())
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids)} events -> {a.out}", flush=True)
    done = tot = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one_event, i, str(a.out)) for i in ids]
        for n, f in enumerate(as_completed(futs), 1):
            _, ok, nd = f.result()
            done += ok; tot += nd
            if n % 100 == 0 or n == len(ids):
                print(f"  {n}/{len(ids)} events  days {done}/{tot}", flush=True)
    print(f"done, days with GFS: {done}/{tot}", flush=True)


if __name__ == "__main__":
    main()
