#!/usr/bin/env python3
"""Convierte los .tif de los 4 anios NUEVOS de WSTS+ (2016, 2017, 2022, 2023) a HDF5.

Replica exactamente CreateHDF5Dataset.py + get_generator_for_hdf5 del fork de
Lahrichi (theirs_repo/src/preprocess/CreateHDF5Dataset.py y
theirs_repo/src/dataloader/FireSpreadDataset.py:598-631), con tres diferencias:

  1. years = [2016, 2017, 2022, 2023]  (el suyo esta hardcodeado a 2018-2021).
  2. ds.lnglat() envuelto en try/except: los .tif de WSTS+ NO tienen CRS
     (comprobado: 40 de 40 ficheros muestreados dan `ds.crs is None`) y
     ds.lnglat() lanza rasterio.errors.CRSError. El script original aborta en el
     primer fuego. Aqui se guarda (nan, nan) y se marca con `has_lnglat=False`.
     lnglat solo es metadato: ni el dataloader ni el entrenamiento lo leen.
  3. Sin dependencia de torch. El original importa FireSpreadDataset solo para
     usar su glob y su generador; ese modulo no importa con torch >= 2.x
     (`from torch.utils.data.dataset import T_co` ya no existe). Aqui se
     reimplementa el glob, que es literalmente ordenar ficheros.

Preprocesado aplicado (identico al suyo, FireSpreadDataset.py:627-630):
  - canal de fuego activo (banda 22): NaN -> 0  (NaN significa "sin deteccion")
  - canal de fuego activo: hora de deteccion de hhmm -> hh  (floor_divide 100)
El resto de bandas se guardan crudas; la estandarizacion ocurre en el dataloader.

Salida: <target_dir>/<anio>/<fire_name>.hdf5, dataset "data" (n_imgs, 23, H, W).

Uso (cualquier anio; --years por defecto son los 4 nuevos, pasa 2016 ... 2023
para reconstruir los ocho de una vez):
  python benchmarks/make_wsts_hdf5.py --data_dir <unzip de WSTSPlus.zip> \
      --target_dir $WSTS_HDF5 --years 2016 2017 2018 2019 2020 2021 2022 2023

Movido al release el 25/08 desde with_wstsplus/make_hdf5_new_years.py; es el
script con el que se construyo el HDF5 que usan wstsplus.py y cross/.
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

            # --- canal de fuego activo (banda 23 / indice 22) ---
            # OJO: la codificacion cambia entre eras.
            #   WSTS 2018-2021 : hhmm (754, 800, ... 2218), NaN si no hay deteccion
            #   WSTS+ nuevos   : hh   (6..22),               0   si no hay deteccion
            # Aplicar floor_divide(.,100) a los anios nuevos pondria TODAS las
            # detecciones a 0 y destruiria el canal de entrada Y el target.
            # Se autodetecta y se registra.
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
