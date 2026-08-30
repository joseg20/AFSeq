#!/usr/bin/env python3
"""Convierte el archivo GFS crudo en covariables por evento y dia.

Para cada dia t con fuego de cada evento toma el ciclo 00Z de ESE dia y sus
horizontes f024..f120 (validos en t+1..t+5), recorta la ventana del evento y
guarda los siete canales en las unidades que espera normalize_climate:

    temp  C     (GFS TMP:surface,  K - 273.15)
    dew   C     (DPT:2m,           K - 273.15)
    rh    %     (RH:2m)
    ws    m/s   (|UGRD,VGRD:10m|)
    wdir  deg   (direccion de donde viene el viento, 0 = N)
    pres  hPa   (PRES:surface / 100)
    precp mm    (APCP acumulado de los cuatro sub-pasos de 6 h del dia)

Se guarda EN LA MALLA NATIVA de 0.25 grados, no en la de 200 m: la ventana de
un evento son unas pocas celdas GFS, asi que el recorte pesa kilobytes y la
interpolacion a la rejilla del evento se hace al vuelo en el loader. Guardar ya
reproyectado costaria ~127 GB; asi cuesta ~0.2 GB.

    <out>/<id>/<fecha>/gfs.npz   fcst (5,7,h,w) float32 + f000 (7,h,w) + bbox
"""
from __future__ import annotations
import argparse, json, sys, tempfile, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
GFS = Path("/workspace1/users/josegui/data/gfs")
DATA = Path("/workspace1/users/josegui/data/dataset_v7")
PAD = 0.75          # grados de margen: 3 celdas GFS a cada lado
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
    """Recorte por bbox; GFS viene en longitudes 0..360."""
    lon = da.longitude.values
    L0, L1 = (lon0 % 360), (lon1 % 360)
    lat_sel = (da.latitude >= lat0) & (da.latitude <= lat1)
    lon_sel = (lon >= L0) & (lon <= L1) if L0 <= L1 else ((lon >= L0) | (lon <= L1))
    return da.isel(latitude=np.where(lat_sel)[0], longitude=np.where(lon_sel)[0])


def seven(msgs, box):
    """Los siete canales, en las unidades del loader."""
    g = {k: _crop(v, *box).values.astype(np.float32) for k, v in msgs.items()
         if k in ("t", "d2m", "r2", "u10", "v10", "sp", "tp")}
    if not {"t", "d2m", "r2", "u10", "v10", "sp"} <= set(g):
        return None
    u, v = g["u10"], g["v10"]
    ws = np.hypot(u, v)
    wdir = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0     # de donde viene
    return np.stack([g["t"] - 273.15, g["d2m"] - 273.15, g["r2"], ws, wdir,
                     g["sp"] / 100.0, g.get("tp", np.zeros_like(u))])


def day_precip(date_dir: Path, box, h_end: int):
    """Precipitacion diaria: suma de los cuatro acumulados de 6 h que caen en
    el dia que termina en h_end."""
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
                    default=Path("/workspace1/users/josegui/data/gfs_events"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ids = sorted(p.name for p in DATA.glob("id_*") if p.is_dir())
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids)} eventos -> {a.out}", flush=True)
    done = tot = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one_event, i, str(a.out)) for i in ids]
        for n, f in enumerate(as_completed(futs), 1):
            _, ok, nd = f.result()
            done += ok; tot += nd
            if n % 100 == 0 or n == len(ids):
                print(f"  {n}/{len(ids)} eventos  dias {done}/{tot}", flush=True)
    print(f"FIN dias con GFS: {done}/{tot}", flush=True)


if __name__ == "__main__":
    main()
