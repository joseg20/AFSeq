#!/usr/bin/env python3
"""Descarga el archivo GFS 0.25 que necesita el contrato de covariables futuras.

Para cada fecha con fuego en AFSeq baja, del ciclo 00Z, solo los mensajes que
usan nuestros siete canales meteorologicos, por byte-range sobre el .idx: unos
5 MB por fichero en vez de los ~400 MB del GRIB completo.

  f000                      analisis  -> contrato gfs_anl
  f024 f048 f072 f096 f120  pronostico -> contrato gfs_fcst (t+1..t+5)
  APCP cada 6 h hasta f120  para poder acumular la precipitacion diaria

Cachea por (fecha, fhr); muchos eventos comparten dias, asi que 13.874
dias-evento se reducen a ~963 fechas. Reanudable: salta lo ya bajado.
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

# Una sesion por hilo: sin esto cada peticion abre un socket nuevo y con 14
# hilos se agota el ulimit (1024) a las pocas miles de descargas -> OSError.
_LOCAL = threading.local()


def _session():
    s = getattr(_LOCAL, "s", None)
    if s is None:
        s = requests.Session()
        ad = HTTPAdapter(pool_connections=4, pool_maxsize=8,
                         max_retries=Retry(total=3, backoff_factor=1.0,
                                           status_forcelist=(500, 502, 503, 504)))
        s.mount("https://", ad)
        _LOCAL.s = s
    return s

B = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT = paths.DATA / "gfs"
# (shortName GRIB, nivel) -> nuestros canales lst, dew, rh, wind_*, pressure
BASE = [("TMP", "surface"), ("DPT", "2 m above ground"), ("RH", "2 m above ground"),
        ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"),
        ("PRES", "surface")]
PRECIP = [("APCP", "surface")]
DAILY = (0, 24, 48, 72, 96, 120)          # analisis + t+1..t+5
PRECIP_HOURS = tuple(range(6, 121, 6))     # acumulados de 6 h


def messages_for(fhr: int):
    """Que mensajes hacen falta en este horizonte."""
    if fhr == 0:
        return BASE                        # el analisis no trae acumulados
    want = list(BASE) if fhr in DAILY else []
    if fhr in PRECIP_HOURS:
        want += PRECIP
    return want


def grab(date: str, fhr: int, cycle: str = "00", tries: int = 4) -> str:
    dst = OUT / date / f"f{fhr:03d}.grib2"
    if dst.exists() and dst.stat().st_size > 1000:
        return "cache"
    want = messages_for(fhr)
    if not want:
        return "skip"
    url = f"{B}/gfs.{date}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{fhr:03d}"
    for attempt in range(tries):
        try:
            r = _session().get(url + ".idx", timeout=40)
            if r.status_code == 404:
                return "404"
            r.raise_for_status()
            rows = [l.split(":") for l in r.text.splitlines()]
            offs = [int(x[1]) for x in rows]
            blob = b""
            for var, lev in want:
                for i, row in enumerate(rows):
                    if row[3] == var and row[4] == lev:
                        s = int(row[1])
                        e = (offs[i + 1] - 1) if i + 1 < len(offs) else ""
                        rr = _session().get(url, headers={"Range": f"bytes={s}-{e}"}, timeout=90)
                        rr.raise_for_status()
                        blob += rr.content
                        break
            if not blob:
                return "empty"
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(blob)
            tmp.rename(dst)
            return "ok"
        except Exception as exc:
            if attempt == tries - 1:
                return f"err:{type(exc).__name__}"
            time.sleep(2 ** attempt)
    return "err"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=Path, default=None, help="json con la lista de fechas")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.dates:
        dates = json.loads(a.dates.read_text())
    else:
        DATA = paths.AFSEQ
        dates = sorted({p.parent.name for p in DATA.glob("id_*/*/active-fire.tif")})
    if a.limit:
        dates = dates[:a.limit]
    hours = sorted(set(DAILY) | set(PRECIP_HOURS))
    jobs = [(d.replace("-", ""), h) for d in dates for h in hours]
    print(f"{len(dates)} fechas x {len(hours)} horizontes = {len(jobs)} ficheros", flush=True)

    stats = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(grab, d, h): (d, h) for d, h in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            stats[r] = stats.get(r, 0) + 1
            if n % 250 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f} min  "
                      + " ".join(f"{k}={v}" for k, v in sorted(stats.items())), flush=True)
    print("FIN", stats, flush=True)


if __name__ == "__main__":
    main()
