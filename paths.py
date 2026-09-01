"""Every path and seed of the project, in one place.

Nothing else in the release hardcodes a directory. Every path here is an
environment variable with a default RELATIVE to the repository, so the code
runs on another machine by setting variables (see env/example.sh),
never by editing a file.

Default layout the code expects next to the repository:

    <parent>/
      release_paper/           this repository
      data/afseq/              AFSeq: covariates AND daily active masks (AFSEQ_DATA)
      data/afseq_cache/        decoded copy, optional  (AFSEQ_CACHE)
      data/wstsplus/hdf5/      WSTS+ HDF5, exp1/4/5    (WSTS_HDF5)
      runs/                    training outputs         (RUNS_DIR)

`WSP_DATA` moves the whole data/ directory at once; the individual variables
override one entry each.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _p(var, default):
    return Path(os.environ.get(var, default))


# --- data -----------------------------------------------------------------
DATA = _p("WSP_DATA", ROOT.parent / "data")
AFSEQ = _p("AFSEQ_DATA", DATA / "afseq")
# Decoded copy of AFSEQ (dataset/afseq_cache.py). Only used when the
# environment variable is SET: the default is where --build writes it.
AFSEQ_CACHE = _p("AFSEQ_CACHE", str(AFSEQ) + "_cache")
WSTS_HDF5 = _p("WSTS_HDF5", DATA / "wstsplus" / "hdf5")

# --- splits and dataset constants shipped WITH the code ---------------------
SPLITS_DIR = _p("AFSEQ_SPLITS_DIR", ROOT / "dataset" / "splits")
AFSEQ_SPLITS = _p("AFSEQ_SPLITS", SPLITS_DIR / "splits_s23.txt")
# per-channel TRAIN means used to impute holes (dataset/afseq_loader.py)
CHANNEL_MEANS = _p("AFSEQ_CHANNEL_MEANS", ROOT / "dataset" / "channel_means.json")

# --- outputs --------------------------------------------------------------
RUNS = _p("RUNS_DIR", ROOT.parent / "runs")
RESULTS = _p("RESULTS_DIR", ROOT / "results")

# --- earth engine ---------------------------------------------------------
# Your own Earth Engine cloud project; dataset/build_afseq.py needs it.
EE_PROJECT = os.environ.get("EE_PROJECT", "")

# --- reproducibility ------------------------------------------------------
SEED = int(os.environ.get("SEED", 23))

# --- the numbers any rewrite has to reproduce -----------------------------
# Acceptance tests, not documentation: if a refactor stops matching these, it
# has a bug, however clean it looks.
ANCHORS = {
    "utae_ar_veg_wstsplus_4fold_crop128": 0.4295,   # paper Table 3: 0.429
    "utae_ar_afseq_t1": 0.432,                      # 3-seed mean
    # 0.124 +- 0.013 (3 seeds; s5/s7/s23 = 0.1196/0.1136/0.1387):
    # exp2.py --table --rows persistence on the AFSeq tree (256-px windows, 10 km
    # margin, 128-px fire-centred patch, 34 channels, 1,194 events after
    # build_afseq --phase curate). With all 1,301 events on the old contour
    # windows and 64-px patches it was 0.252.
    "persistence_afseq_t1": 0.124,
    "persistence_wsts_t1": 0.193,
}
