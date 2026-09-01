"""Download the 0.25 degree GFS archive the future-covariate contract needs.

For every date with fire in AFSeq this pulls, from the 00Z cycle, only the
messages our seven weather channels use, as byte ranges over the .idx file.
That is about 5 MB per file instead of the ~400 MB of the full GRIB.

  f000                      analysis, for the gfs_anl contract
  f024 f048 f072 f096 f120  forecast, for the gfs_fcst contract (t+1 to t+5)
  APCP every 6 h up to f120, so daily precipitation can be accumulated

The cache key is (date, fhr). Many events share days, so 13,874 event-days
come down to about 963 dates. Resumable: anything already downloaded is
skipped.
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

# One session per thread. Without this every request opens a fresh socket and,
# with 14 threads, the ulimit of 1024 runs out after a few thousand downloads
# and every later request raises OSError.
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
# (GRIB shortName, level) for our lst, dew, rh, wind_* and pressure channels
BASE = [("TMP", "surface"), ("DPT", "2 m above ground"), ("RH", "2 m above ground"),
        ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"),
        ("PRES", "surface")]
PRECIP = [("APCP", "surface")]
DAILY = (0, 24, 48, 72, 96, 120)          # analysis plus t+1 to t+5
PRECIP_HOURS = tuple(range(6, 121, 6))     # 6-hourly accumulations


def messages_for(fhr: int):
    """Which GRIB messages this forecast hour needs."""
    if fhr == 0:
        return BASE                        # the analysis carries no accumulation
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
    ap.add_argument("--dates", type=Path, default=None, help="json with the list of dates")
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
    print(f"{len(dates)} dates x {len(hours)} horizons = {len(jobs)} files", flush=True)

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
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
