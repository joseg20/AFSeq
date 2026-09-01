# Active fire spread forecasting: release code

Five directories, one job each. Nothing imports across them except through
`paths.py`, and no file hardcodes a directory.

    dataset/      builds AFSeq from scratch: events, static layers, fire masks,
                  climate, reflectance, plus the audit that proves no hole was
                  filled with a plausible constant
    models/       UTAE-AR, self-contained
    training/     training loops
    evaluation/   HOW things are measured: pooled AP, the 128-px centre crop,
                  persistence, fold averaging
    experiments/  WHAT is measured: one .py per table of the paper
    benchmarks/   third-party code (WildfireSpreadTS and WildfireSpreadTS+),
                  vendored under its own licence, plus the bridges that let it
                  read AFSeq and that let our models read theirs

There is one implementation of the metric, shared by every experiment, and it
is not meant to be reimplemented per experiment: scoring the whole scene
instead of the crop moves UTAE-AR from 0.429 to 0.386.

## Licence

This repository is CC BY 4.0 (`LICENSE`). The one exception is
`benchmarks/wsts+/`, a vendored copy of WildfireSpreadTS that keeps its own MIT
licence in `benchmarks/wsts+/LICENSE`. The vendored commit and the reversible
compatibility patches applied to it are recorded in
`benchmarks/wsts+/UPSTREAM.txt`.

## The rule the dataset code enforces

A missing measurement is written as NaN and recorded in `provenance.json`. It
is never replaced by a plausible constant, which would read as a real
measurement once written. Imputation happens at load time instead, with the
channel mean over the training split.

## Acceptance tests

`paths.ANCHORS` lists numbers this code must reproduce. They are tests, not
trivia: a rewrite that stops matching them has a bug, however clean it reads.

## Running it somewhere else

The code hardcodes no directory. `paths.py` reads every location from the
environment, with defaults relative to the repository:

    <parent>/release_paper/        this repo
    <parent>/data/afseq/           AFSeq day rasters          AFSEQ_DATA
    <parent>/data/afseq_cache/     decoded copy (optional)    AFSEQ_CACHE
    <parent>/data/wstsplus/hdf5/   WSTS+ HDF5 (exp1/4/7 only) WSTS_HDF5
    <parent>/runs/                 outputs                    RUNS_DIR

`WSP_DATA` moves `data/` as a whole. `env/example.sh` is the template: copy it,
edit the four directories and `source` it before launching. Splits, channel
means and the fusion statistics ship inside the repo, in `dataset/splits`,
`dataset/channel_means_v7.json` and `experiments/exp4/afseq_channel_stats.json`.

Each event day holds `active-fire.tif` (the daily VIIRS active-fire mask, which
is both the target and the fire channel), `climate.tif` with seven bands and
`reflectance.tif` with I1, I2 and M11. `dem.tif` and `vegetation.tif` are per
event rather than per day. The loader turns all of that into the 34 model
channels of `dataset/afseq_loader.CHANNELS`: terrain, land cover as one-hot,
NDVI, EVI and ET, the seven weather channels, I1, I2, M11, and the active fire.
Nothing is imported from outside the repository: the AFSeq loader is
`dataset/afseq_loader.py` and the losses are `training/losses.py`.

The dataset the paper uses is what `dataset/build_afseq.py --phase all` leaves
behind. Its last phase, `curate` (`dataset/curate.py`), moves duplicated events
and the 15% most persistent US fires out of the tree and records them in
`<tree>/excluded.json`, and `dataset/make_splits.py` only splits what is left.
`requirements.txt` lists the direct dependencies with the versions used here;
torch is unpinned because its build has to follow the driver.

To move the whole thing: this repository and the AFSeq tree (about 19 GB of
GeoTIFF, covariates and daily active-fire masks together) are enough for exp2
and exp3. exp1, exp4 and exp7 also need the WSTS+ HDF5 (92 GB), and exp4's
UTAE-AR on WSTS+ needs the four published `best_AR_Veg_p4f{0-3}` checkpoints
(82 MB).

## The crop: AFSEQ_CROP

The loader trains and scores on a 128-px patch centred on the day-t fire. When
the front is wider than the patch, its bounding box can be bimodal, with two
clusters at the ends and nothing in the middle, and the centred patch then
holds no fire at all. That happens in 7.1% of train, 6.5% of val and 7.9% of
test samples, and those rollouts start from an empty feedback mask and score
near zero.

`AFSEQ_CROP=v8`, the default here, takes instead the patch with the most day-t
fire, found exactly with an integral image and deterministically. It consumes
the same two draws of `random` that the centred crop does, so every sample
whose front fits inside the patch is bit-identical between the two modes and
only the wide fronts move. `AFSEQ_CROP=v7` restores the centred crop, which is
the sampling the tables of the paper were produced with.

## AFSeq: the dataset the paper uses

The window is not the scar. `build_afseq.window_of` makes each event's raster
the bounding box of its final contour plus 10 km on every side, and never
smaller than 256 x 256 px at 200 m. That is twice the 128-px training patch, so
the fire-centred crop has room to move (`afseq_loader.fire_centred_crop`). A
window cut to the contour would let the raster extent itself tell a model where
the fire is going to end.

The tree is built from 1,301 contours, with the 34 model channels, the DEM on
the 200 m grid, and the detection hour as band 2 of the fire mask, which only
the cross-benchmark fusion reads. Curation keeps 1,194 events, dropping 33
duplicates and the 15% most persistent US fires, for 13,874 daily observations,
about 19 GB of GeoTIFF and a 29 GB decoded cache in uint16. Persistence on the
128-px fire-centred patch scores 0.124 ± 0.013. To rebuild:

    source env/example.sh
    python dataset/build_afseq.py --out <tree> --phase events --contours fires.geojson
    python dataset/build_afseq.py --out <tree> --phase all --workers 12

`--phase all` runs everything after the events phase in one go: fire, static,
climate, reflect, a repair pass over the daily layers, the audit, the curation,
the splits, the decoded cache with its identity check, and the channel
statistics. Each step is also its own `--phase`, and the whole thing is
resumable, so a raster that is already healthy is never downloaded twice. To
reproduce an existing tree's event list instead of starting from a contour
file, use `--phase events --from-tree <old> --fresh`.

Every number of exp2 and exp3 is trained on this tree.
