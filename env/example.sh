# Paths of THIS machine. `source env/example.sh` before launching anything.
# Copy it, edit the four directories, and never edit a path inside the code:
# paths.py reads every location from these variables.

# Everything under one root: data/ and runs/ hang off it.
export WSP_DATA=/path/to/data

# AFSeq, as `dataset/build_afseq.py --out <tree>` leaves it, and its optional
# decoded cache (dataset/afseq_cache.py --build).
export AFSEQ_DATA=$WSP_DATA/afseq
export AFSEQ_CACHE=$WSP_DATA/afseq_cache

# WSTS+ as HDF5 (benchmarks/make_wsts_hdf5.py). Only exp1, exp4 and exp7 need it.
export WSTS_HDF5=$WSP_DATA/wstsplus/hdf5

# Where checkpoints, logs and result JSONs go.
export RUNS_DIR=/path/to/runs

# Your own Earth Engine cloud project. Only dataset/build_afseq.py needs it.
export EE_PROJECT=your-ee-project

# How the 128-px patch is placed: "maxfire" (default) takes the patch holding
# the most day-t fire when the front is wider than the patch; "centred" always
# centres on the bounding box, which is what the published tables used.
# export AFSEQ_CROP=centred

# Future-covariate contract: "era" (reanalysis, the default), "gfs_fcst" (the
# forecast available at run time) or "gfs_anl" (the GFS analysis).
# export AFSEQ_MET=era

# HDF5 over NFS: the WSTS+ loader's workers hang on flock without this.
export HDF5_USE_FILE_LOCKING=FALSE

# Loader workers per training process.
export WSP_WORKERS=8
