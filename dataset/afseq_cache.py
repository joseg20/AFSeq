#!/usr/bin/env python3
"""A decoded copy of AFSeq, so training is not a GDAL benchmark.

    python dataset/afseq_cache.py --build            # once, ~5 min
    python dataset/afseq_cache.py --check 60         # cached == raw, bit for bit
    AFSEQ_CACHE=/path/to/cache python experiments/exp2.py --launch

WHY. Profiled on a training sample of Seq2SeqActivePadded (patch 64, history
up to 16): 186 ms per sample, 89% of it inside load_raster -- 57 rasterio opens
per sample, five .tif per day of history. The dataset is 843 MB and sits whole
in the page cache, iowait is 0: the cost is decoding, not disk. With four
workers that is 3 s per batch of 64 and ~4 min per epoch, GPU at 0-40%, and it
is why every AFSeq run here is CPU-bound while we are asked to use fewer cores.

WHAT IT STORES. Exactly what the loader would have computed, once:

    <cache>/<fid>/<date>.npy          get_features(...)[0]   float32 (17, H, W)
    <cache>/<fid>/active/<date>.npy   load_raster(<fid>/<date>/active-fire.tif)[0][0]
    <cache>/<fid>/DONE                the fire was fully processed

A day that get_features answers with None (no active-fire mask the next day, a
missing static layer, an exception it swallows) is simply absent, and the
cached call answers None too. The whole thing is 4.6 GB.

HOW IT PLUGS IN. `install(module)` replaces `get_features` and `load_raster`
in the namespace of dataset/afseq_loader.py with lookups into the cache. Nothing
else changes: same index, same crops, same padding, same random.seed(idx).
The four places that import the loader call `install_if_configured` after the
import, and it is a no-op unless AFSEQ_CACHE is set. `--check` proves the
equivalence the only way that counts: np.array_equal on every tensor of the
same samples, raw versus cached.

It refuses to install on a cache that is missing any fire of the dataset (no
DONE marker): a half-built cache would otherwise turn missing days into the
all-zero frames the loader uses for holes, silently.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import AFSEQ, AFSEQ_CACHE, SPLITS_DIR  # noqa: E402


# ------------------------------------------------------------------ building
def _fire_dirs(afseq: Path) -> list[Path]:
    return sorted(p for p in afseq.iterdir() if p.is_dir())


def _build_fire(args) -> tuple[str, int, int]:
    """Decode one fire. Returns (fid, days written, days answered None)."""
    fid, afseq, cache = args
    from dataset.afseq_loader import get_features, load_raster

    iddir = afseq / fid
    out = cache / fid
    (out / "active").mkdir(parents=True, exist_ok=True)
    written = missing = 0
    for day in sorted(p.name for p in iddir.iterdir() if p.is_dir()):
        X, _ = get_features(iddir, day)
        if X is None:
            missing += 1
        else:
            np.save(out / f"{day}.npy", X.astype(np.float32, copy=False))
            written += 1
        tif = iddir / day / "active-fire.tif"
        if tif.exists():
            m, _ = load_raster(tif)
            np.save(out / "active" / f"{day}.npy", m[0])
    (out / "DONE").touch()
    return fid, written, missing


def build(afseq: Path = AFSEQ, cache: Path = AFSEQ_CACHE, procs: int = 4,
          force: bool = False):
    cache.mkdir(parents=True, exist_ok=True)
    fids = [p.name for p in _fire_dirs(afseq)]
    todo = [f for f in fids if force or not (cache / f / "DONE").exists()]
    print(f"{len(fids)} fuegos, {len(fids) - len(todo)} ya en cache, "
          f"{len(todo)} por decodificar -> {cache}")
    t0 = time.time()
    jobs = [(f, afseq, cache) for f in todo]
    written = missing = 0
    with Pool(procs) as pool:
        for i, (fid, w, m) in enumerate(pool.imap_unordered(_build_fire, jobs), 1):
            written += w
            missing += m
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {written} dias, {missing} None  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    size = sum(p.stat().st_size for p in cache.rglob("*.npy"))
    print(f"listo: {size / 1e9:.2f} GB en {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- installing
def missing_fires(cache: Path = AFSEQ_CACHE, afseq: Path = AFSEQ) -> list[str]:
    """Fires of `afseq` with no DONE marker in `cache`; empty means usable."""
    if not cache.is_dir():
        return [p.name for p in _fire_dirs(afseq)]
    return [p.name for p in _fire_dirs(afseq)
            if not (cache / p.name / "DONE").exists()]


def install(module, cache: Path = AFSEQ_CACHE, afseq: Path = AFSEQ):
    """Point `module.get_features` / `module.load_raster` at the cache."""
    missing = missing_fires(cache, afseq)
    if missing:
        raise SystemExit(
            f"AFSEQ_CACHE={cache} no cubre {len(missing)} fuegos de {afseq} "
            f"(p.ej. {missing[0]}). Construyela primero:\n"
            f"  python dataset/afseq_cache.py --build")
    cache = Path(cache)

    def get_features(id_dir, fire_date):
        f = cache / Path(id_dir).name / f"{fire_date}.npy"
        if not f.exists():
            return None, None
        return np.load(f), None

    def load_raster(path):
        path = Path(path)                        # <fid>/<date>/active-fire.tif
        f = cache / path.parent.parent.name / "active" / (path.parent.name + ".npy")
        if not f.exists():
            raise FileNotFoundError(f"{path} no esta en la cache ({f})")
        return np.load(f)[None], None

    module.get_features = get_features
    module.load_raster = load_raster
    module.AFSEQ_CACHE_INSTALLED = str(cache)


def install_if_configured(module) -> bool:
    """No-op unless AFSEQ_CACHE is set. Returns whether the cache is in use."""
    if not os.environ.get("AFSEQ_CACHE"):
        return False
    if getattr(module, "AFSEQ_CACHE_INSTALLED", None):
        return True
    install(module)
    return True


# ------------------------------------------------------------------ checking
def check(n: int, splits_file, horizon: int = 5, patch=64, max_history=16):
    """np.array_equal on every output of `n` random samples per section.

    One dataset object; the only thing that changes between the two passes is
    which get_features / load_raster the module resolves at call time.
    """
    import random
    from dataset import afseq_loader as dap
    from dataset.afseq_loader import dl_ids, load_ids

    raw_fns = (dap.get_features, dap.load_raster)
    ok = True
    for section in ("Train", "Val", "Test"):
        ids = dl_ids(load_ids(section, splits_file), AFSEQ)
        ds = dap.Seq2SeqActivePadded(ids, str(AFSEQ), patch, max_history, horizon)
        random.seed(0)
        idxs = random.sample(range(len(ds)), min(n, len(ds)))

        dap.get_features, dap.load_raster = raw_fns
        t = time.time()
        a = [ds[i] for i in idxs]
        t_raw = (time.time() - t) / len(idxs)

        install(dap)
        t = time.time()
        b = [ds[i] for i in idxs]
        t_cached = (time.time() - t) / len(idxs)
        dap.get_features, dap.load_raster = raw_fns
        dap.AFSEQ_CACHE_INSTALLED = None

        bad = 0
        for i, x, y in zip(idxs, a, b):
            same = (all(np.array_equal(u, v) for u, v in zip(x[:4], y[:4]))
                    and x[4] == y[4])
            if not same:
                bad += 1
                print(f"  DIFIERE {section} idx {i}: {x[4]}")
        ok &= bad == 0
        print(f"{section:5s} {len(idxs)} muestras: {bad} difieren; "
              f"{t_raw * 1000:.0f} ms/muestra raw -> {t_cached * 1000:.1f} ms "
              f"cache ({t_raw / max(t_cached, 1e-9):.0f}x)", flush=True)
    print("IDENTICO" if ok else "FALLO: la cache no reproduce el loader")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--force", action="store_true", help="rehacer fuegos ya hechos")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--check", type=int, metavar="N",
                    help="comparar N muestras por seccion, raw vs cache")
    ap.add_argument("--splits", default=str(SPLITS_DIR / "splits_v7_s7.txt"))
    ap.add_argument("--cache", type=Path, default=AFSEQ_CACHE)
    a = ap.parse_args()
    if a.build:
        build(cache=a.cache, procs=a.procs, force=a.force)
    if a.check:
        sys.exit(0 if check(a.check, a.splits) else 1)
    if not (a.build or a.check):
        ap.print_help()


if __name__ == "__main__":
    main()
