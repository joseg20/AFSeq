"""Convert the .tif files of WSTS+ to HDF5.

This reproduces CreateHDF5Dataset.py and get_generator_for_hdf5 of the
upstream repository (src/preprocess/CreateHDF5Dataset.py and
src/dataloader/FireSpreadDataset.py:598-631), with three differences:

  1. The years are an argument. Theirs is hardcoded to 2018-2021, which leaves
     out the four years WSTS+ adds (2016, 2017, 2022, 2023).
  2. ds.lnglat() is wrapped in try/except. The WSTS+ .tif files carry no CRS
     (40 of 40 sampled files give `ds.crs is None`), so ds.lnglat() raises
     rasterio.errors.CRSError and the original script aborts on the first
     fire. Here it stores (nan, nan) and records `has_lnglat=False`. lnglat is
     metadata only: neither the dataloader nor the training reads it.
  3. No torch dependency. The original imports FireSpreadDataset just to use
     its glob and its generator, and that module does not import under
     torch >= 2.x (`from torch.utils.data.dataset import T_co` is gone). The
     glob is reimplemented here, which is literally sorting file names.

The preprocessing is identical to theirs (FireSpreadDataset.py:627-630):
  - active fire channel (band 22): NaN -> 0, since NaN means "no detection"
  - active fire channel: detection time from hhmm to hh (floor_divide 100)
Every other band is stored raw; standardisation happens in the dataloader.

Output: <target_dir>/<year>/<fire_name>.hdf5, dataset "data" (n_imgs, 23, H, W).

Usage (any year; --years defaults to the four new ones, pass 2016 ... 2023 to
rebuild all eight at once):
  python benchmarks/make_wsts_hdf5.py --data_dir <unzipped WSTSPlus.zip> \
      --target_dir $WSTS_HDF5 --years 2016 2017 2018 2019 2020 2021 2022 2023

This is the script the HDF5 used by wstsplus.py and cross/ was built with.
"""
import argparse
import glob
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

NEW_YEARS = [2016, 2017, 2022, 2023]


def read_list_of_images(data_dir, years):
    """Equivalente a FireSpreadDataset.read_list_of_images con load_from_hdf5=False."""
    imgs_per_fire = {}
    for year in years:
        imgs_per_fire[year] = {}
        fires_in_year = sorted(glob.glob(f"{data_dir}/{year}/*/"))
        for fire_dir in fires_in_year:
            fire_name = fire_dir.split("/")[-2]
            paths = sorted(glob.glob(f"{fire_dir}/*.tif"))
            imgs_per_fire[year][fire_name] = paths
            if not paths:
                warnings.warn(f"Fuego {year}:{fire_name} sin imagenes.", RuntimeWarning)
    return imgs_per_fire


def generator(imgs_per_fire):
    """Copia de get_generator_for_hdf5 con ds.lnglat() protegido."""
    for year, fires_in_year in imgs_per_fire.items():
        for fire_name, img_files in fires_in_year.items():
            if not img_files:
                continue
            imgs = []
            lnglat = None
            for img_path in img_files:
                with rasterio.open(img_path, "r") as ds:
                    imgs.append(ds.read())
                    if lnglat is None:
                        try:
                            lnglat = ds.lnglat()
                        except Exception:
                            lnglat = None  # WSTS+ .tif sin CRS -> CRSError
            x = np.stack(imgs, axis=0)

            img_dates = [
                p.split("/")[-1].split("_")[0].replace(".tif", "") for p in img_files
            ]

            # Active fire channel (band 23, index 22).
            # Careful: the encoding changes between eras.
            #   WSTS 2018-2021 : hhmm (754, 800, ... 2218), NaN for no detection
            #   the newer WSTS+: hh   (6..22),              0   for no detection
            # Applying floor_divide(., 100) to the newer years would set EVERY
            # detection to 0 and destroy both the input channel and the target.
            # The encoding is detected and logged.
            af = np.nan_to_num(x[:, -1, ...], nan=0)
            pos = af[af > 0]
            if pos.size and pos.max() >= 100:
                af = np.floor_divide(af, 100)          # era hhmm -> hh
                enc = "hhmm"
            else:
                enc = "hh"                              # ya venia en horas
            x[:, -1, ...] = af

            has_lnglat = lnglat is not None
            if not has_lnglat:
                lnglat = (float("nan"), float("nan"))

            yield year, fire_name, img_dates, lnglat, has_lnglat, enc, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Directorio con subcarpetas por anio (WSTSPlus/)")
    parser.add_argument("--target_dir", required=True,
                        help="Where the HDF5 files go; put them on fast local storage, not NFS.")
    parser.add_argument("--years", type=int, nargs="*", default=NEW_YEARS)
    args = parser.parse_args()

    imgs_per_fire = read_list_of_images(args.data_dir, args.years)
    n_fires = sum(len(v) for v in imgs_per_fire.values())
    n_imgs = sum(len(p) for v in imgs_per_fire.values() for p in v.values())
    print(f"{n_fires} fuegos, {n_imgs} imagenes en {args.years}")

    for y in args.years:
        Path(f"{args.target_dir}/{y}").mkdir(parents=True, exist_ok=True)

    n_done = n_skip = n_nocrs = 0
    n_fire_px = 0
    enc_counts = {"hh": 0, "hhmm": 0}
    for year, fire_name, img_dates, lnglat, has_lnglat, enc, imgs in tqdm(
            generator(imgs_per_fire), total=n_fires):
        h5_path = f"{args.target_dir}/{year}/{fire_name}.hdf5"
        if Path(h5_path).is_file():
            n_skip += 1
            continue

        with h5py.File(h5_path, "w") as f:
            dset = f.create_dataset("data", imgs.shape, data=imgs)
            dset.attrs["year"] = year
            dset.attrs["fire_name"] = fire_name
            dset.attrs["img_dates"] = img_dates
            dset.attrs["lnglat"] = lnglat
            dset.attrs["has_lnglat"] = has_lnglat
            dset.attrs["af_encoding"] = enc

        n_done += 1
        n_nocrs += (not has_lnglat)
        enc_counts[enc] += 1
        n_fire_px += int((imgs[:, -1, ...] > 0).sum())

    print(f"escritos={n_done} ya_existian={n_skip} sin_crs={n_nocrs}")
    print(f"codificacion fuego activo: {enc_counts}")
    print(f"pixeles de fuego totales escritos: {n_fire_px:,}")
    if n_done and n_fire_px == 0:
        raise SystemExit("ERROR: canal de fuego vacio. Revisa la codificacion hh/hhmm.")


if __name__ == "__main__":
    main()
