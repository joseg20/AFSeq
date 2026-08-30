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
    X_hist      (T, 34, p, p)   the history up to day t, T <= max_seq_len;
                                CHANNELS below: 0..32 covariates (land cover
                                one-hot in 3..19), 33 the daily active-fire mask
    fut_covars  (H-1, 33, p, p) covariates of days t+1 .. t+H-1 (no fire)
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
day's active-fire.tif (SNPP + NOAA-20 detections of that day, 0 elsewhere),
every band min-max normalised into [0, 1] with the fixed
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

# THE CHANNEL LAYOUT, the one place it is defined. Everything that needs a
# count or an index imports it from here (train.py, afseq_datamodule.py,
# exp2.py, exp4/channels.py). Changed on 2026-08-25 from 17 to 34 channels:
# land cover went from one ordinal scalar to 17 one-hot classes (it is a
# category, and that is how WSTS encodes it) and VNP09GA M11 joined I1/I2
# (the one Veg channel of WSTS+ that AFSeq lacked).
LANDCOVER_CLASSES = 17                       # MCD12Q1 IGBP classes 1..17
CHANNELS = (["elevation", "slope", "aspect"]
            + [f"landcover_{k}" for k in range(1, LANDCOVER_CLASSES + 1)]
            + ["ndvi", "evi", "et",
               "lst", "dew", "rh", "wind_speed", "wind_dir", "pressure",
               "precipitation", "refl_i1", "refl_i2", "refl_m11", "fire"])
N_CHANNELS = len(CHANNELS)                   # 34
FIRE_CHANNEL = CHANNELS.index("fire")        # 33, always last
N_COVARIATES = N_CHANNELS - 1                # what fut_covars carries
N_STATIC = 3 + LANDCOVER_CLASSES + 3         # terrain + land cover + NDVI/EVI/ET: constant per event
# THE PATCH: what the model sees per sample, in pixels of the 200 m grid
# (128 px = 25.6 km). Fixed on 2026-08-25: with the v7 windows, 48% of the
# fire days have a day-t front larger than 64 px and those days hold 81% of
# the fire pixels; a 64-px patch saw half of the day's fire. 128 is also the
# crop WSTS/WSTS+ train and evaluate on.
PATCH = 128
# Optional extra channel, NOT in CHANNELS: the hour of detection (UTC, 0-23,
# 0 = no fire) carried as band 2 of active-fire.tif. Only the cross-benchmark
# fusion asks for it (Seq2SeqActivePadded(with_fire_hour=True)), to fill the
# one Veg slot of WSTS+ that AFSeq had no counterpart for; it is appended
# after the fire channel, so X_hist then has N_CHANNELS + 1 channels.
FIRE_HOUR_CHANNEL = N_CHANNELS
LANDCOVER = slice(3, 3 + LANDCOVER_CLASSES)
CH = {name: i for i, name in enumerate(CHANNELS)}



# ---------------------------------------------------------------- contrato GFS
# AFSEQ_MET elige de donde salen los siete canales meteorologicos de las
# covariables FUTURAS (fut_covars). "era" es lo que se ha usado siempre:
# reanalisis observado del dia t+k, un techo que en operacion no existe.
# "gfs_fcst" usa el pronostico emitido el dia t (f024..f120) y "gfs_anl" el
# analisis GFS del propio dia, que aisla el error de pronostico del efecto de
# la resolucion mas gruesa. La historia (X) nunca cambia: siempre es observada.
MET_MODE = os.environ.get("AFSEQ_MET", "era")
GFS_EVENTS = Path(os.environ.get(
    "AFSEQ_GFS", "/workspace1/users/josegui/data/gfs_events"))
MET_SLICE = slice(CH["lst"], CH["precipitation"] + 1)      # canales 23..29



_EVENT_TRANSFORM = {}


def event_transform(iddir, date):
    """Transform del raster del evento, leido del GeoTIFF de disco.

    No usa load_raster a proposito: con AFSEQ_CACHE instalada esa funcion sirve
    .npy de la cache y devuelve None como transform.
    """
    fid = Path(iddir).name
    tr = _EVENT_TRANSFORM.get(fid)
    if tr is None:
        with rasterio.open(Path(iddir) / date / "active-fire.tif") as src:
            tr = src.transform
        _EVENT_TRANSFORM[fid] = tr
    return tr


def _gfs_regrid(arr, bbox, transform, top, left, ph):
    """De la malla GFS de 0.25 grados a la rejilla del crop, bilineal.

    arr: (..., h, w) en la malla nativa; bbox: (lon0, lat0, lon1, lat1) del
    recorte, con latitudes DECRECIENTES como las sirve GFS.
    """
    lon0, lat0, lon1, lat1 = bbox
    h, w = arr.shape[-2:]
    # centros de celda del crop, en coordenadas del raster del evento
    cols = left + np.arange(ph) + 0.5
    rows = top + np.arange(ph) + 0.5
    lon = transform.c + transform.a * cols
    lat = transform.f + transform.e * rows
    # posicion fraccionaria dentro de la malla GFS
    gx = (lon - lon0) / max(lon1 - lon0, 1e-9) * (w - 1)
    gy = (lat1 - lat) / max(lat1 - lat0, 1e-9) * (h - 1)
    gx = np.clip(gx, 0, w - 1); gy = np.clip(gy, 0, h - 1)
    x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (gx - x0)[None, :]; fy = (gy - y0)[:, None]
    a = arr[..., y0[:, None], x0[None, :]]
    b = arr[..., y0[:, None], x1[None, :]]
    c = arr[..., y1[:, None], x0[None, :]]
    d = arr[..., y1[:, None], x1[None, :]]
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)


def gfs_future_met(fid, day_t, transform, top, left, ph, horizon):
    """(horizon, 7, ph, ph) con los canales meteorologicos del contrato GFS,
    o None si ese dia no tiene pronostico descargado."""
    f = GFS_EVENTS / fid / day_t / "gfs.npz"
    if not f.exists():
        return None
    try:
        z = np.load(f)
        bbox = z["bbox"]
        src = z["fcst"] if MET_MODE == "gfs_fcst" else None
        if MET_MODE == "gfs_anl":
            an = z["anl"]
            if an.size == 0:
                return None
            src = np.repeat(an[None], len(z["fcst"]), axis=0)
        if src is None:
            return None
        out = _gfs_regrid(src.astype(np.float32), bbox, transform, top, left, ph)
        if out.shape[0] < horizon:
            out = np.concatenate([out, np.repeat(out[-1:], horizon - out.shape[0], 0)])
        return out[:horizon]
    except Exception as exc:
        if os.environ.get("AFSEQ_MET_STRICT"):
            raise
        print(f"[gfs] {fid}/{day_t}: {type(exc).__name__} {exc}", file=sys.stderr)
        return None


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
    """4 raw bands -> 20 channels: land cover one-hot (17), NDVI, EVI, ET.

    Land cover is categorical (IGBP 1..17); it used to be written as one
    scalar (class-1)/16, which puts "grassland" between "savanna" and
    "cropland" on a scale that means nothing. One-hot, as WSTS does. Class 0
    / nodata gives all zeros.
    """
    lc, ndvi, evi, et = veg
    lc_oh = (np.rint(lc)[None] == np.arange(1, LANDCOVER_CLASSES + 1)[:, None, None]
             ).astype(np.float32)

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

    return np.concatenate([lc_oh, np.stack([ndvi, evi, et])])


def normalize_reflectance(ref):
    """I1, I2, M11 (VNP09GA reflectance floats), same min-max for the three.

    A 2-band file (trees built before M11 was added) gets M11 as NaN, which
    impute_missing fills with the channel mean like any other hole.
    """
    max_ref = 1.6
    min_ref = -0.01
    if ref.shape[0] == 2:
        ref = np.concatenate([ref, np.full((1, *ref.shape[1:]), np.nan)])
    return np.stack([np.clip((b - min_ref) / (max_ref - min_ref), 0, 1)
                     for b in ref[:3]])


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


def normalize_climate_stack(g):
    """normalize_climate aplicado a (T, 7, H, W)."""
    return np.stack([normalize_climate(g[i]) for i in range(g.shape[0])])


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
    fire_mask_path = id_dir / next_date / 'active-fire.tif'
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
            reflectance = np.full((3, *target_shape), np.nan)

        # Máscara de hoy (feature)
        today_path = date_path / "active-fire.tif"
        if not today_path.exists(): return None, None
        # binary from here on: EE writes -inf as nodata on the edge pixels of
        # a float GeoTIFF, and the dataset overwrites this channel with the
        # same `> 0.5` mask anyway (see Seq2SeqActivePadded.__getitem__)
        today_fire_mask = (load_raster(today_path)[0][:1] > 0.5).astype(np.float32)   # band 1 only: band 2 is the detection hour

        X = np.concatenate([dem_normalized, vegetation, climate, reflectance, today_fire_mask], axis=0)
        assert X.shape[0] == N_CHANNELS, f"{id_dir.name}/{fire_date}: {X.shape[0]} canales, esperados {N_CHANNELS}"
        X = impute_missing(X)
        return X.astype(np.float32), fire_mask[0].astype(np.float32)

    except Exception as e:
        logging.warning(f"⚠️ Error procesando {id_dir.name}/{fire_date}: {e}")
        return None, None


# ----------------------------------------------------------------- dataset
def fire_centred_crop(mask_t, ph: int, tH: int, tW: int):
    """(top, left) of the patch: centred on the day-t fire, jittered.

    Rasters are now bigger than the patch (window_of in build_afseq.py), so a
    uniform crop would often hold no fire at all. The patch is centred on the
    bounding box of the day-t active mask and shifted by a uniform jitter of
    up to a quarter patch in each axis (less if the front would not fit),
    clipped to the canvas; with no fire on
    day t it falls back to a uniform crop. Consumes exactly two draws of the
    `random` module after random.seed(idx), so samples stay reproducible. On
    a raster that fits inside the patch in both axes it returns (0, 0), which
    is what the old uniform crop returned too: unchanged samples there.
    """
    ys, xs = np.where(mask_t > 0.5)
    if len(ys) == 0:
        return random.randint(0, tH - ph), random.randint(0, tW - ph)
    cy = (int(ys.min()) + int(ys.max())) // 2
    cx = (int(xs.min()) + int(xs.max())) // 2
    # jitter up to a quarter patch, but never so much that a fire which fits
    # in the patch is pushed out of it: the day-t front stays whole whenever
    # its bounding box is smaller than the patch (85% of the fire days at
    # 128 px). Larger fronts get whatever jitter their overhang leaves (0).
    bh = int(ys.max()) - int(ys.min()) + 1
    bw = int(xs.max()) - int(xs.min()) + 1
    jy = max(0, min(ph // 4, (ph - bh) // 2))
    jx = max(0, min(ph // 4, (ph - bw) // 2))
    top = cy - ph // 2 + random.randint(-jy, jy)
    left = cx - ph // 2 + random.randint(-jx, jx)
    return (int(np.clip(top, 0, tH - ph)), int(np.clip(left, 0, tW - ph)))


class Seq2SeqActivePadded(Dataset):
    def __init__(self, ids, DATASET, patch_size=PATCH, max_seq_len=16,
                 horizon=8, require_fire_on_t=True, with_fire_hour=False):
        # require_fire_on_t=True is AFSeq's protocol and the default, so every
        # existing experiment is untouched. False also emits the windows WSTS
        # emits and we never do -- ignition (nothing at t, fire at t+1),
        # extinction and empty -- which are 30.9% of their test and 0% of ours.
        self.DATASET = Path(DATASET)
        self.with_fire_hour = with_fire_hour
        self.ph = patch_size; self.max_seq_len = max_seq_len; self.H = horizon
        self.samples = []
        for fid in ids:
            iddir = self.DATASET / fid
            if not iddir.is_dir():
                continue
            # a day exists when it has its daily active-fire mask
            dates = sorted(p.parent.name for p in iddir.glob("*/active-fire.tif"))
            has_fire = {}
            for dt in dates:
                try:
                    m, _ = load_raster(iddir / dt / "active-fire.tif"); has_fire[dt] = bool((m[0] > 0.5).any())
                except Exception:
                    has_fire[dt] = False
            for idx, date in enumerate(dates):
                if require_fire_on_t and not has_fire.get(date, False):
                    continue
                # días futuros consecutivos (por calendario) con máscara activa, hasta H
                future, d = [], date
                for _ in range(self.H):
                    nd = _add(d, 1)
                    if not (iddir / nd / "active-fire.tif").exists():
                        break
                    future.append(nd); d = nd
                if len(future) < 1:     # necesita >=1 día futuro (no >=H)
                    continue
                start = max(0, idx - (max_seq_len - 1))
                self.samples.append((fid, dates[start: idx + 1], date, future))
        if not self.samples:
            # Silent emptiness is how a wrong AFSEQ_DATA turns into a model that
            # "trains" on nothing. Say what was looked for and where.
            raise RuntimeError(
                f"Seq2SeqActivePadded: 0 muestras. {len(ids)} ids pedidos, "
                f"DATASET={self.DATASET}; cada dia necesita "
                "<id>/<fecha>/active-fire.tif (build_afseq.py --phase fire, o "
                "--phase adopt-active para un arbol de mascaras antiguo).")

    def __len__(self):
        return len(self.samples)

    def _active(self, fid, date):
        m, _ = load_raster(self.DATASET / fid / date / "active-fire.tif")
        return (m[0] > 0.5).astype(np.float32)

    def _hour(self, fid, date):
        """Detection hour (band 2); zeros for trees built before it existed."""
        m, _ = load_raster(self.DATASET / fid / date / "active-fire.tif")
        return (m[1] if m.shape[0] > 1 else np.zeros_like(m[0])).astype(np.float32)

    def __getitem__(self, idx):
        fid, seq_dates, last_date, future = self.samples[idx]
        iddir = self.DATASET / fid; ph = self.ph; H = self.H; C = N_CHANNELS + int(self.with_fire_hour)
        navail = len(future)
        try:
            y0 = self._active(fid, future[0]); Ho, Wo = y0.shape
            tH = max(Ho, ph); tW = max(Wo, ph)
            random.seed(idx)
            top, left = fire_centred_crop(self._active(fid, last_date), ph, tH, tW)

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
                        feat[FIRE_CHANNEL] = am
                    else:
                        h = min(am.shape[0], feat.shape[1]); w = min(am.shape[1], feat.shape[2])
                        feat[FIRE_CHANNEL] = 0.0; feat[FIRE_CHANNEL, :h, :w] = am[:h, :w]
                    if self.with_fire_hour:
                        hr = self._hour(fid, d)
                        hh = np.zeros(feat.shape[1:], np.float32)
                        h = min(hr.shape[0], hh.shape[0]); w = min(hr.shape[1], hh.shape[1])
                        hh[:h, :w] = hr[:h, :w]
                        feat = np.concatenate([feat, hh[None]])
                    X_seq.append(crop(feat))
                else:
                    X_seq.append(np.zeros((C, ph, ph), dtype=np.float32))
            X_hist = np.stack(X_seq).astype(np.float32)

            # fut_covars (H-1): disponibles t+1..t+navail-1, resto ceros (enmascarado en loss)
            fut = []
            for j in range(H - 1):
                if j < navail - 1:
                    res = get_features(iddir, future[j])
                    fut.append(crop(res[0])[:N_COVARIATES] if res[0] is not None else np.zeros((N_COVARIATES, ph, ph), np.float32))
                else:
                    fut.append(np.zeros((N_COVARIATES, ph, ph), dtype=np.float32))
            fut_covars = np.stack(fut).astype(np.float32) if H > 1 else np.zeros((0, N_COVARIATES, ph, ph), np.float32)

            # contrato de covariables futuras: sustituir la meteorologia
            # observada por la del pronostico GFS emitido el dia t
            if MET_MODE != "era" and fut_covars.shape[0] > 0:
                _tr = event_transform(iddir, last_date)
                _g = gfs_future_met(fid, last_date, _tr, top, left, ph,
                                    fut_covars.shape[0])
                if _g is not None:
                    fut_covars[:, MET_SLICE] = normalize_climate_stack(_g)

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
            fut_covars = np.zeros((max(H - 1, 0), N_COVARIATES, ph, ph), np.float32)
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


def dl_ids(ids: list, dataset) -> list:
    """Keep the ids that have at least one daily active-fire mask on disk."""
    dataset = Path(dataset)
    return [i for i in ids
            if (dataset / i).is_dir() and any((dataset / i).glob("*/active-fire.tif"))]
