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
    <parent>/data/dataset_v6_cache decoded copy (optional)    AFSEQ_CACHE
    <parent>/data/wstsplus/hdf5/   WSTS+ HDF5 (exp1/4/5 only) WSTS_HDF5
    <parent>/runs/                 outputs                    RUNS_DIR

`WSP_DATA` moves data/ as a whole. `env/lascar.sh` is our machine's set of
exports; write one per machine and `source` it before launching. Splits,
channel means and the fusion statistics ship inside the repo (dataset/splits,
dataset/channel_means_v5.json, experiments/exp4/afseq_channel_stats.json).
Each event day holds `active-fire.tif` (the daily VIIRS active-fire mask, target
and fire channel), `climate.tif` and `reflectance.tif`; `dem.tif` and
`vegetation.tif` per event. Nothing is imported from outside the repository any more: the AFSeq loader is
dataset/afseq_loader.py and the losses are training/losses.py.
The dataset the paper uses is what `dataset/build_afseq.py --phase all`
leaves behind: its last phase, `curate` (dataset/curate.py), moves duplicated
events and the 15% most persistent US fires out of the tree and records them
in `<tree>/excluded.json`; `dataset/make_splits.py` only splits what is left.
`requirements.txt` lists the direct dependencies with the versions used here;
torch is unpinned because its build follows the driver.

To move: this repository and data/dataset_v6 (850 MB, covariates and the
daily active-fire masks in one tree) reproduce exp2 and exp3. exp1, exp4 and exp5 also need
the WSTS+ HDF5 (92 GB) and, for exp4's UTAE-AR/WSTS+ cell, the four
best_AR_Veg_p4f{0-3} checkpoints (82 MB).

## AFSeq v7: the 128-px window (the dataset the paper uses)

Until 2026-08-25 every event's rasters covered its final contour plus 0.01
deg: 91% of the events were smaller than the 64-px training patch (the loader
padded them with zeros) and the raster extent itself told a model where the
fire would end. `build_afseq.window_of` now makes the window the contour's
bounding box plus 5 km on every side, never smaller than 128 x 128 px at
200 m (25.6 km, twice the patch), and the loader crops the patch around the
day-t fire with jitter (`afseq_loader.fire_centred_crop`). Built on 2026-08-25 from the 1,301 contours in about 1 h 05 against Earth
Engine with 34 threads (the build is resumable); curation on the new windows
keeps 1,194 events (33 duplicates and the 15% most persistent US events out;
64 of those 74 coincide with the list computed on the old windows). 13,927
daily observations, 12,732 training days, 16 GB of GeoTIFF (the 30 m DEM is
13 GB of it), 20 GB decoded cache. Persistence 0.156 +- 0.011. To rebuild:

    source env/<machine>.sh
    python dataset/build_afseq.py --out <new tree> --phase events --from-tree $AFSEQ_DATA
    python dataset/build_afseq.py --out <new tree> --phase static
    python dataset/build_afseq.py --out <new tree> --phase fire    --workers 12 &
    python dataset/build_afseq.py --out <new tree> --phase climate --workers 12 &
    python dataset/build_afseq.py --out <new tree> --phase reflect --workers 12 &
    wait
    python dataset/build_afseq.py --out <new tree> --phase audit
    python dataset/build_afseq.py --out <new tree> --phase curate
    AFSEQ_DATA=<new tree> python dataset/make_splits.py --seeds 5 7 23
    AFSEQ_DATA=<new tree> python dataset/afseq_cache.py --build && ... --check 60

Every number of exp2 and exp3 is trained on this tree.

