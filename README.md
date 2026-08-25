# Active fire spread forecasting -- release code

Five directories, one job each. Nothing imports across them except through
`paths.py`, and no file hardcodes a directory.

    dataset/      builds AFSeq from scratch: events, static layers, fire masks,
                  climate, reflectance, plus the audit that proves no hole was
                  filled with a plausible constant
    models/       UTAE-AR, self-contained
    training/     training loops
    evaluation/   HOW things are measured: pooled AP, the 128-px center crop,
                  persistence, fold averaging. One implementation, shared by
                  every experiment -- scoring the whole scene instead of the
                  crop moves UTAE-AR from 0.429 to 0.386, so this cannot be
                  reimplemented per experiment
    experiments/  WHAT is measured: one .py per table or figure of the paper
    benchmarks/   third-party code (WildfireSpreadTS+), kept unmodified, with
                  every compatibility change listed in its CAMBIOS_COMPAT.md

## The rule the dataset code enforces

A missing measurement is written as NaN and recorded in `provenance.json`; it
is never replaced by a plausible constant. The previous pipeline answered an
empty Earth Engine collection with `constant([0,0])` for reflectance and
`constant(273.15)` -- 0 C -- for land surface temperature. Both read as real
measurements once written, and they silently killed 83% of the reflectance and
34% of the LST without one error message. Imputation happens at load time, with
the channel mean over the training split.

## Acceptance tests

`paths.ANCHORS` lists numbers this code must reproduce. They are tests, not
trivia: a rewrite that stops matching them has a bug, however clean it reads.

## Running it somewhere else

The code hardcodes no directory. `paths.py` reads every location from the
environment, with defaults relative to the repository:

    <parent>/release_paper/        this repo
    <parent>/data/dataset_v6/      AFSeq day rasters          AFSEQ_DATA
    <parent>/data/fires_active_v4/ AFSeq daily active masks   AFSEQ_ACTIVE
    <parent>/data/dataset_v6_cache decoded copy (optional)    AFSEQ_CACHE
    <parent>/data/wstsplus/hdf5/   WSTS+ HDF5 (exp1/4/5 only) WSTS_HDF5
    <parent>/runs/                 outputs                    RUNS_DIR

`WSP_DATA` moves data/ as a whole. `env/lascar.sh` is our machine's set of
exports; write one per machine and `source` it before launching. Splits,
channel means and the fusion statistics ship inside the repo (dataset/splits,
dataset/channel_means_v5.json, experiments/exp4/afseq_channel_stats.json).
Nothing is imported from outside the repository any more: the AFSeq loader is
dataset/afseq_loader.py and the losses are training/losses.py.
`requirements.txt` lists the direct dependencies with the versions used here;
torch is unpinned because its build follows the driver.

To move: this repository, data/dataset_v6 and data/fires_active_v4 (850 MB
together) reproduce exp2 and exp3. exp1, exp4 and exp5 also need
the WSTS+ HDF5 (92 GB) and, for exp4's UTAE-AR/WSTS+ cell, the four
best_AR_Veg_p4f{0-3} checkpoints (82 MB).
