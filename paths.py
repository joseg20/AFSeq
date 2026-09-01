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
      data/wstsplus/hdf5/      WSTS+ HDF5              (WSTS_HDF5)
      runs/                    training outputs         (RUNS_DIR)

`WSP_DATA` moves the whole data/ directory at once; the individual variables
override one entry each.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _p(var, default):
    return Path(os.environ.get(var, default))


# data
DATA = _p("WSP_DATA", ROOT.parent / "data")
AFSEQ = _p("AFSEQ_DATA", DATA / "afseq")
# Decoded copy of AFSEQ (dataset/afseq_cache.py). Only used when the
# environment variable is SET: the default is where --build writes it.
AFSEQ_CACHE = _p("AFSEQ_CACHE", str(AFSEQ) + "_cache")
WSTS_HDF5 = _p("WSTS_HDF5", DATA / "wstsplus" / "hdf5")

# splits and dataset constants shipped WITH the code
SPLITS_DIR = _p("AFSEQ_SPLITS_DIR", ROOT / "dataset" / "splits")
AFSEQ_SPLITS = _p("AFSEQ_SPLITS", SPLITS_DIR / "splits_s23.txt")
# per-channel TRAIN means used to impute holes (dataset/afseq_loader.py)
CHANNEL_MEANS = _p("AFSEQ_CHANNEL_MEANS", ROOT / "dataset" / "channel_means.json")

# outputs
RUNS = _p("RUNS_DIR", ROOT.parent / "runs")
RESULTS = _p("RESULTS_DIR", ROOT / "results")

# earth engine
# Your own Earth Engine cloud project; dataset/build_afseq.py needs it.
EE_PROJECT = os.environ.get("EE_PROJECT", "")

# reproducibility
SEED = int(os.environ.get("SEED", 23))
