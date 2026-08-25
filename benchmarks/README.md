# benchmarks/ — third-party code

## wsts+/

The WildfireSpreadTS+ repository of Lahrichi et al., **as published**, with its
original configurations. Nothing in it is edited: the bytes here are theirs.

Weights: <https://huggingface.co/saadlahrichi/WSTSPlus>

### Running it here

`wsts+_compat.py --apply` makes three import-level fixes, and `--revert` undoes
them. They do not touch the model, the loss, the hyper-parameters, the splits
or the evaluation:

1. `T_co` is no longer exported by `torch.utils.data.dataset` since torch 2.1.
   It is a typing `TypeVar`.
2. `src/models/__init__.py` imports `TransUnetLightning`, **a file that is not
   in the public repository**. The package therefore does not import as
   published; this is a defect of their release, not an incompatibility with
   our environment.
3. `SegFormerLightning` drags in `transformers`, unused for `SMPModel`.

### Their protocol, unchanged

Their code already supports both designs; we only pass the flag:

    --data.additional_data true    4 folds over WSTS+ (2016-2023), the
                                   published protocol: 4 train / 2 val / 2 test
    (default, false)               12 folds over WSTS only (2018-2021), with 2
                                   train / 1 val / 1 test

The released HuggingFace weights (`Res18UTAE_T5/Veg`, fold0..fold11, mean AP
0.478) are the **12-fold, WSTS-only** design. They cannot be used for the
4-fold WSTS+ column of our tables: different training years, different test
sets.

`cfgs/unet/res18_monotemporal.yaml` states in its first line that it "contains
best parameters determined in grid search", so their tuned configuration is
what ships — no grid search of ours is involved:

    AdamW, lr 1e-3, Focal loss, pos_class_weight 236,
    resnet18 encoder pretrained on ImageNet, seed 0, 10000 steps

### One thing to know about their Focal

`alpha_focal` is commented out, so their default applies: `BaseModel.py`
normalizes `pos_class_weight` and then passes `alpha = 1 - weight`, which makes
alpha 0.0042 instead of 0.9958. In torchvision alpha weights the POSITIVE
class, so their Focal penalizes a fire mistake ~236x less than a background
one — the opposite of what their paper describes. We do not correct it: their
published numbers come from this code, and the question these tables ask is how
*their* model transfers, not how a fixed version would.

## Getting WSTS+ (only exp1, exp4 and exp5 need it)

1. `benchmarks/download_wstsplus.sh <dir>` -- Zenodo record 17584629,
   WSTSPlus.zip, 19.9 GB, md5 42da7598cc33a170064e78d8027148c9. Resumable by
   an external loop on purpose: `curl --retry` together with `-C -` truncates
   the file and restarts from zero (that is how 6 GiB were lost once).
2. `unzip` it; the tree is `<year>/<fire>/*.tif`, 23 bands, no CRS.
3. `benchmarks/make_wsts_hdf5.py --data_dir <unzipped> --target_dir $WSTS_HDF5
   --years 2016 2017 2018 2019 2020 2021 2022 2023` -- their
   CreateHDF5Dataset.py with the years unpinned and two fixes (no CRS, no torch
   import); 92 GB, `<year>/<fire>.hdf5` with a `data` dataset of
   (days, 23, H, W). `wstsplus_data.py` reads exactly that.

