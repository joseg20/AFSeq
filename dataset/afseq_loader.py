"""How AFSeq is read: the sequence dataset, its collate, and the raster stack.

    from dataset.afseq_loader import (Seq2SeqActivePadded, collate_padded,
                                      load_ids, dl_ids)

Ported VERBATIM on 2026-08-25 from the research tree (model/experiments_v2/
dataset_active_padded.py, model/utils/utils.py:get_features and the functions
it calls, with_wstsplus/train_afseq_proto.py:load_ids/dl_ids), so that the
release stands alone; nothing in it imports that tree any more. Equality with
the original was checked sample by sample (np.array_equal on every output) on
Train/Val/Test of splits_v4_s7 the day of the port.

THE SAMPLE. One item is (X_hist, fut_covars, Y, valid, meta):
    X_hist      (T, 17, p, p)   the history up to day t, T <= max_seq_len;
                                channels 0..15 are the covariates and 16 is
                                the daily active-fire mask
    fut_covars  (H-1, 16, p, p) covariates of days t+1 .. t+H-1 (no fire)
    Y           (H, 1, p, p)    active-fire masks of t+1 .. t+H
    valid       (H,)            1 where Y[k] exists; the loss masks the rest
    meta        (fid, day t, (H0, W0) of the scene)
Histories have variable length; `collate_padded` pads them with zero frames at
the END, and every model reads that padding through a pad mask (see the
PADDING note in experiments/exp2.py). require_fire_on_t=True is AFSeq's
protocol: a sample needs fire on day t and at least one observed day after.

THE RASTER STACK (get_features). Per fire day: dem.tif (elevation, slope,
aspect), vegetation.tif (land cover, NDVI, EVI, ET), climate.tif (7 bands),
reflectance.tif (VIIRS I1, I2, bilinearly resampled to the fire grid) and the
day's fire-mask.tif, every band min-max normalised into [0, 1] with the fixed
ranges below. A hole is NaN on disk (dataset/build_afseq.py never writes a
plausible constant) and is imputed here with the channel's TRAIN mean
(dataset/channel_means_v5.json): 0 would be the bottom of every range. The
normalised zero of reflectance is 0.006211, not 0.

If AFSEQ_CACHE is set, dataset/afseq_cache.py swaps `get_features` and
`load_raster` in this module for lookups into a decoded copy: 26-34x faster,
bit-identical (its --check proves it).
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paths import CHANNEL_MEANS  # noqa: E402


def _add(d, n):
    return (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=n)).strftime("%Y-%m-%d")


# ----------------------------------------------------------------- rasters
# static layers (dem, vegetation) do not change with the day: decoded once
_STATIC_CACHE = {}
_CHANNEL_MEANS = None


def _channel_means():
    """Per-channel TRAIN means, loaded once. Missing file is a hard error."""
    global _CHANNEL_MEANS
    if _CHANNEL_MEANS is None:
        p = Path(CHANNEL_MEANS)
        if not p.exists():
            raise FileNotFoundError(
                f"faltan las medias de canal ({p}): sin ellas no se puede "
                "imputar, y rellenar con 0 sesga los canales hacia el minimo.")
        _CHANNEL_MEANS = np.asarray(json.loads(p.read_text())["mean"],
                                    dtype=np.float32)
    return _CHANNEL_MEANS


def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read()  # carga todas las bandas
        transform = src.transform
    return data, transform

# ajustar la cantidad de pixeles


def downsample_average(raster, target_shape):
    """
    raster: np.array de forma (bands, height, width)
    target_shape: (target_height, target_width)
    """
    bands, h, w = raster.shape
    th, tw = target_shape
    factor_h = h // th
    factor_w = w // tw

    # recortar para que sea divisible
    h_crop = th * factor_h
    w_crop = tw * factor_w
    raster_cropped = raster[:, :h_crop, :w_crop]

    # reshape y promediar
    raster_down = raster_cropped.reshape(
        bands,
        th, factor_h,
        tw, factor_w
    ).mean(axis=(2, 4)).astype(int)

    return raster_down


def upsample(raster, target_shape):
    """
    Amplía un raster al tamaño target_shape usando interpolación bilineal.
    
    raster: np.array de forma (bands, height, width)
    target_shape: (target_height, target_width)
    """
    bands, _, _ = raster.shape
    th, tw = target_shape
    raster_up = np.zeros((bands, th, tw), dtype=raster.dtype)

    for b in range(bands):
        raster_up[b] = resize(
            raster[b],
            (th, tw),
            order=1,           # 1 = bilineal
            mode='reflect',
            anti_aliasing=False,
            preserve_range=True
        )

    return raster_up


def normalize_dem(dem):
    elevation, slope, aspect = dem

    elev_max = 4000 # max dataset: 3380, satellite max: 6500
    elev_min = -100
    slope_max = 90
    aspect_max = 360
    # Normalización por banda
    elevation = (elevation - elev_min) / (elev_max - elev_min)
    slope     = slope / slope_max
    aspect    = aspect / aspect_max

    # Clipping para evitar valores fuera de rango
    elevation = np.clip(elevation, 0, 1)
    slope     = np.clip(slope, 0, 1)
    aspect    = np.clip(aspect, 0, 1)

    return np.stack([elevation, slope, aspect])


def normalize_vegetation(veg):
    lc, ndvi, evi, et = veg

    min_lc = 1
    max_lc = 17
    min_vg_index = -2000   
    max_vg_index = 10000 
    max_et = 1000 # max dataset: 569
    # Normalización por banda
    lc = (lc - min_lc) / (max_lc - min_lc)
    ndvi = (ndvi - min_vg_index) / (max_vg_index - min_vg_index)
    evi = (evi - min_vg_index) / (max_vg_index - min_vg_index)
    et = et / max_et

    # Clipping para evitar valores fuera de rango
    lc = np.clip(lc, 0, 1)
    ndvi = np.clip(ndvi, 0, 1)
    evi = np.clip(evi, 0, 1)
    et = np.clip(et, 0, 1)

    return np.stack([lc, ndvi, evi, et])


def normalize_reflectance(ref):
    I1, I2 = ref
    max_ref = 1.6
    min_ref = -0.01
    I1 = np.clip((I1 - min_ref) / (max_ref - min_ref), 0, 1)
    I2 = np.clip((I2 - min_ref) / (max_ref - min_ref), 0, 1)
    return np.stack([I1, I2])


def normalize_climate(climate):
    temp, dew, rh, ws, wdir, pres, precp = climate

    max_T = 60 # temp: -2.6 to 79
    min_dew, max_dew = -20, 20
    max_rh = 100
    max_ws = 25
    max_pres = 1000
    max_precp = 100
    temp = np.clip(temp / max_T, 0, 1)
    dew = np.clip((dew - min_dew) / (max_dew - min_dew), 0, 1)
    rh = np.clip(rh / max_rh, 0, 1)
    ws = np.clip(ws / max_ws, 0, 1)
    wdir = np.clip(wdir / 360, 0, 1)
    pres = np.clip(pres / max_pres, 0, 1)
    precp = np.clip(precp / max_precp, 0, 1)

    return np.stack([temp, dew, rh, ws, wdir, pres, precp])


def impute_missing(X):
    """Replace NaN holes with the channel mean (neutral), never with 0.

    0 is the BOTTOM of every min-max normalized channel, so filling with it
    asserts the coldest temperature and the darkest reflectance. The mean says
    "nothing special known about this pixel", which is the honest statement and
    what WildfireSpreadTS does after z-scoring.
    """
    if not np.isnan(X).any():
        return X
    means = _channel_means()
    n = min(X.shape[0], means.shape[0])
    for ch in range(n):
        v = X[ch]
        m = np.isnan(v)
        if m.any():
            v[m] = means[ch]
    if np.isnan(X).any():                      # channels beyond the table
        X = np.nan_to_num(X, nan=0.0)
    return X


def get_features(id_dir, fire_date):
    date_path = id_dir / fire_date
    if not date_path.exists(): 
        return None, None # Retorno explícito si no existe la carpeta de la fecha
    
    # Verificar máscara del día siguiente (etiqueta Y)
    next_date = (datetime.strptime(fire_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    fire_mask_path = id_dir / next_date / 'fire-mask.tif'
    if not fire_mask_path.exists():
        return None, None

    try:
        fire_mask, _ = load_raster(fire_mask_path)
        target_shape = fire_mask.shape[1:]

        # --- LECTURA DE DATOS ESTÁTICOS ---
        fire_id = id_dir.name
        if fire_id not in _STATIC_CACHE:
            dem_path = id_dir / "dem.tif"
            veg_path = id_dir / "vegetation.tif"
            
            if not dem_path.exists() or not veg_path.exists():
                return None, None
                
            dem_data, _ = load_raster(dem_path)
            dem_norm = normalize_dem(downsample_average(dem_data, target_shape))
            
            veg_data, _ = load_raster(veg_path)
            veg_norm = normalize_vegetation(veg_data)
            
            _STATIC_CACHE[fire_id] = {'dem': dem_norm, 'veg': veg_norm}

        dem_normalized = _STATIC_CACHE[fire_id]['dem']
        vegetation = _STATIC_CACHE[fire_id]['veg']

        # --- DATOS DINÁMICOS ---
        climate_path = date_path / "climate.tif"
        if climate_path.exists():
            climate = normalize_climate(load_raster(climate_path)[0])
        else:
            # NO zero-fill: a missing file is a hole, and 0 means 0 C here.
            # impute_missing puts the channel mean in below.
            climate = np.full((7, *target_shape), np.nan)

        refl_path = date_path / "reflectance.tif"
        if refl_path.exists():
            reflectance = normalize_reflectance(upsample(load_raster(refl_path)[0], target_shape))
        else:
            # NO zero-fill: 0 is zero reflectance, a real (very dark) value
            reflectance = np.full((2, *target_shape), np.nan)

        # Máscara de hoy (feature)
        today_path = date_path / "fire-mask.tif"
        if not today_path.exists(): return None, None
        today_fire_mask = load_raster(today_path)[0]

        X = np.concatenate([dem_normalized, vegetation, climate, reflectance, today_fire_mask], axis=0)
        X = impute_missing(X)
        return X.astype(np.float32), fire_mask[0].astype(np.float32)

    except Exception as e:
        logging.warning(f"⚠️ Error procesando {id_dir.name}/{fire_date}: {e}")
        return None, None


# ----------------------------------------------------------------- dataset
class Seq2SeqActivePadded(Dataset):
    def __init__(self, ids, DATASET, ACTIVE, patch_size=64, max_seq_len=16,
                 horizon=8, require_fire_on_t=True):
        # require_fire_on_t=True is AFSeq's protocol and the default, so every
        # existing experiment is untouched. False also emits the windows WSTS
        # emits and we never do -- ignition (nothing at t, fire at t+1),
        # extinction and empty -- which are 30.9% of their test and 0% of ours.
        self.DATASET = Path(DATASET); self.ACTIVE = Path(ACTIVE)
        self.ph = patch_size; self.max_seq_len = max_seq_len; self.H = horizon
        self.samples = []
        for fid in ids:
            adir = self.ACTIVE / fid; iddir = self.DATASET / fid
            if not adir.is_dir() or not iddir.is_dir():
                continue
            dates = sorted(p.stem for p in adir.glob("*.tif") if (iddir / p.stem).is_dir())
            has_fire = {}
            for dt in dates:
                try:
                    m, _ = load_raster(adir / f"{dt}.tif"); has_fire[dt] = bool((m[0] > 0.5).any())
                except Exception:
                    has_fire[dt] = False
            for idx, date in enumerate(dates):
                if require_fire_on_t and not has_fire.get(date, False):
                    continue
                # días futuros consecutivos (por calendario) con máscara activa, hasta H
                future, d = [], date
                for _ in range(self.H):
                    nd = _add(d, 1)
                    if not (adir / f"{nd}.tif").exists():
                        break
                    future.append(nd); d = nd
                if len(future) < 1:     # necesita >=1 día futuro (no >=H)
                    continue
                start = max(0, idx - (max_seq_len - 1))
                self.samples.append((fid, dates[start: idx + 1], date, future))

    def __len__(self):
        return len(self.samples)

    def _active(self, fid, date):
        m, _ = load_raster(self.ACTIVE / fid / f"{date}.tif")
        return (m[0] > 0.5).astype(np.float32)

    def __getitem__(self, idx):
        fid, seq_dates, last_date, future = self.samples[idx]
        iddir = self.DATASET / fid; ph = self.ph; H = self.H; C = 17
        navail = len(future)
        try:
            y0 = self._active(fid, future[0]); Ho, Wo = y0.shape
            tH = max(Ho, ph); tW = max(Wo, ph)
            random.seed(idx); top = random.randint(0, tH - ph); left = random.randint(0, tW - ph)

            def crop(arr):
                if arr.shape[1] < tH or arr.shape[2] < tW:
                    arr = np.pad(arr, ((0, 0), (0, tH - arr.shape[1]), (0, tW - arr.shape[2])), mode="constant")
                return arr[:, top:top + ph, left:left + ph]

            def crop2(m):
                return crop(m[None])[0]

            # historia: features 0..15 + activo en ch16
            X_seq = []
            for d in seq_dates:
                res = get_features(iddir, d)
                if res[0] is not None:
                    feat = res[0].copy(); am = self._active(fid, d)
                    if am.shape == feat.shape[1:]:
                        feat[16] = am
                    else:
                        h = min(am.shape[0], feat.shape[1]); w = min(am.shape[1], feat.shape[2])
                        feat[16] = 0.0; feat[16, :h, :w] = am[:h, :w]
                    X_seq.append(crop(feat))
                else:
                    X_seq.append(np.zeros((C, ph, ph), dtype=np.float32))
            X_hist = np.stack(X_seq).astype(np.float32)

            # fut_covars (H-1): disponibles t+1..t+navail-1, resto ceros (enmascarado en loss)
            fut = []
            for j in range(H - 1):
                if j < navail - 1:
                    res = get_features(iddir, future[j])
                    fut.append(crop(res[0])[:16] if res[0] is not None else np.zeros((16, ph, ph), np.float32))
                else:
                    fut.append(np.zeros((16, ph, ph), dtype=np.float32))
            fut_covars = np.stack(fut).astype(np.float32) if H > 1 else np.zeros((0, 16, ph, ph), np.float32)

            # targets + valid_mask
            Ys, valid = [], np.zeros(H, dtype=np.float32)
            for k in range(H):
                if k < navail:
                    Ys.append(crop2(self._active(fid, future[k]))); valid[k] = 1.0
                else:
                    Ys.append(np.zeros((ph, ph), dtype=np.float32))
            Y = np.stack(Ys)[:, None].astype(np.float32)
        except Exception as e:
            logging.error(f"❌ padded __getitem__ {fid}: {e}")
            X_hist = np.zeros((len(seq_dates), C, ph, ph), np.float32)
            fut_covars = np.zeros((max(H - 1, 0), 16, ph, ph), np.float32)
            Y = np.zeros((H, 1, ph, ph), np.float32); valid = np.zeros(H, np.float32)
            Ho, Wo = ph, ph
        return X_hist, fut_covars, Y, valid, (fid, last_date, (Ho, Wo))


def collate_padded(batch, pad_value=0.0):
    seqs = [b[0] for b in batch]; futs = [b[1] for b in batch]
    ys = [b[2] for b in batch]; vs = [b[3] for b in batch]; metas = [b[4] for b in batch]
    Tmax = max(s.shape[0] for s in seqs); B = len(seqs); _, C, H, W = seqs[0].shape
    X = np.full((B, Tmax, C, H, W), pad_value, np.float32)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s
    return (torch.from_numpy(X),
            torch.from_numpy(np.stack(futs).astype(np.float32)),
            torch.from_numpy(np.stack(ys).astype(np.float32)),
            torch.from_numpy(np.stack(vs).astype(np.float32)),   # (B,H) valid mask
            metas)


# ------------------------------------------------------------------ splits
def load_ids(section: str, splits_file) -> list:
    """Fire ids under the `Train` / `Val` / `Test` heading of a splits file."""
    lines = Path(splits_file).read_text().splitlines()
    s = [i for i, l in enumerate(lines) if l.startswith(section)][0] + 1
    out = []
    for l in lines[s:]:
        if l.strip() == "" or l.startswith(("Train", "Val", "Test")):
            break
        out.append(l.strip())
    return out


def dl_ids(ids: list, active) -> list:
    """Keep the ids that have at least one active-fire mask on disk."""
    active = Path(active)
    return [i for i in ids
            if (active / i).is_dir() and any((active / i).glob("*.tif"))]
