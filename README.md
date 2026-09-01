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
    benchmarks/   third-party code (WildfireSpreadTS / WildfireSpreadTS+),
                  vendored under its own LICENSE, plus the bridges that let it
                  read AFSeq and that let our models read theirs

## Licence

This repository is CC BY 4.0 (`LICENSE`). The one exception is
`benchmarks/wsts+/`, a vendored copy of WildfireSpreadTS that keeps its own MIT
licence in `benchmarks/wsts+/LICENSE`; `benchmarks/wsts+/UPSTREAM.txt` records
the vendored commit and the reversible compatibility patches applied to it.

## The rule the dataset code enforces

A missing measurement is written as NaN and recorded in `provenance.json`; it
is never replaced by a plausible constant, which would read as a real
measurement once written. Imputation happens at load time, with the channel
mean over the training split.

## Acceptance tests

`paths.ANCHORS` lists numbers this code must reproduce. They are tests, not
trivia: a rewrite that stops matching them has a bug, however clean it reads.

## Running it somewhere else

The code hardcodes no directory. `paths.py` reads every location from the
environment, with defaults relative to the repository:

    <parent>/release_paper/        this repo
    <parent>/data/afseq/          AFSeq day rasters          AFSEQ_DATA
    <parent>/data/afseq_cache/    decoded copy (optional)    AFSEQ_CACHE
    <parent>/data/wstsplus/hdf5/   WSTS+ HDF5 (exp1/4/7 only) WSTS_HDF5
    <parent>/runs/                 outputs                    RUNS_DIR

`WSP_DATA` moves data/ as a whole. `env/example.sh` is the template; copy it,
edit the four directories and `source` it before launching. Splits,
channel means and the fusion statistics ship inside the repo (dataset/splits,
dataset/channel_means_v7.json, experiments/exp4/afseq_channel_stats.json).
Each event day holds `active-fire.tif` (the daily VIIRS active-fire mask, target
and fire channel), `climate.tif` (7 bands) and `reflectance.tif` (I1, I2, M11); `dem.tif` and
`vegetation.tif` per event. The loader turns them into 34 model channels
(`dataset/afseq_loader.CHANNELS`: terrain, land cover one-hot, NDVI/EVI/ET,
seven weather channels, I1/I2/M11, active fire). Nothing is imported from outside the repository any more: the AFSeq loader is
dataset/afseq_loader.py and the losses are training/losses.py.
The dataset the paper uses is what `dataset/build_afseq.py --phase all`
leaves behind: its last phase, `curate` (dataset/curate.py), moves duplicated
events and the 15% most persistent US fires out of the tree and records them
in `<tree>/excluded.json`; `dataset/make_splits.py` only splits what is left.
`requirements.txt` lists the direct dependencies with the versions used here;
torch is unpinned because its build follows the driver.

To move: this repository and the AFSeq tree (~19 GB of GeoTIFF, covariates and
the daily active-fire masks together) reproduce exp2 and exp3. exp1, exp4 and
exp7 also need the WSTS+ HDF5 (92 GB) and, for exp4's UTAE-AR/WSTS+ cell, the
four published best_AR_Veg_p4f{0-3} checkpoints (82 MB).

## The crop: AFSEQ_CROP

The loader trains and scores on a 128-px patch centred on the day-t fire. When
the front is WIDER than the patch its bounding box can be bimodal -- two
clusters at the ends, nothing in the middle -- and the centred patch then holds
no fire at all: 7.1% of train, 6.5% of val and 7.9% of test samples. Those
rollouts start from an empty feedback mask and score near zero.

`AFSEQ_CROP=v8`, the default here, takes instead the patch with the most day-t
fire, found exactly with an integral image, deterministically. It consumes the
same two draws of `random` that the centred crop does, so every sample whose
front FITS the patch is bit-identical between the two modes; only the wide
fronts move. `AFSEQ_CROP=v7` restores the centred crop, which is the sampling
the tables of the paper were produced with.

## AFSeq: the dataset the paper uses

THE WINDOW IS NOT THE SCAR. `build_afseq.window_of` makes each event's raster
the bounding box of its final contour plus 10 km on every side, never smaller
than 256 x 256 px at 200 m -- twice the 128-px training patch, so the
fire-centred crop has room to move (`afseq_loader.fire_centred_crop`). A window
cut to the contour would let the raster extent itself tell a model where the
fire is going to end.

Built from 1,301 contours: 34 model channels, DEM on the 200 m grid, detection
hour as band 2 of the fire mask (the cross-benchmark fusion is the only consumer).
Curation keeps 1,194 events -- 33 duplicates and the 15% most persistent US
events go out -- for 13,874 daily observations, ~19 GB of GeoTIFF and a 29 GB
decoded cache (uint16). Persistence on the 128-px fire-centred patch:
0.124 +- 0.013. To rebuild:

    source env/example.sh
    python dataset/build_afseq.py --out <tree> --phase events --contours fires.geojson
    python dataset/build_afseq.py --out <tree> --phase all --workers 12

`--phase all` runs everything after the events phase in one go: fire, static,
climate, reflect, a repair pass over the daily layers, the audit, the curation
(duplicates and the 15% most persistent US events out), the splits, the
decoded cache with its identity check, and the channel statistics. Each step
is also its own `--phase`, and the whole thing is resumable: a raster that is
already healthy is never downloaded twice. To reproduce an existing tree's
event list instead of a contour file: `--phase events --from-tree <old> --fresh`.

Every number of exp2 and exp3 is trained on this tree.

