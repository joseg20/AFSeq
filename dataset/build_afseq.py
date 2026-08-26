#!/usr/bin/env python3
"""Build the AFSeq dataset end to end, from event geometries to day rasters.

ONE entry point. Run it against an empty directory and it downloads everything;
run it against an existing tree and it fills only what is missing or sick. The
repairs that were applied as one-off scripts during August 2026 are folded in
here, so this file -- not a sequence of patches -- is what defines the dataset.

    python build_afseq.py --out /path/new --phase events --contours fires.geojson
    python build_afseq.py --out /path/new --phase events --from-tree /path/old_tree
                                    # same 1,194 events, new window (bbox + 5 km, >= 128 px)
    python build_afseq.py --out /path/new --phase all     # everything, start to finish:
        # events must exist (see above); then fire, static, climate, reflect, a repair
        # pass, audit, curate, splits, cache (+ identity check), channel stats
    python build_afseq.py --out /path/tree --phase climate --workers 16
    python build_afseq.py --out /path/tree --phase audit
    python build_afseq.py --out /path/tree --phase curate   # last step of "all"

Layout produced, per event and per day:
    <out>/<id>/info.geojson  contour.geojson  dem.tif  vegetation.tif
    <out>/<id>/<YYYY-MM-DD>/active-fire.tif  climate.tif  reflectance.tif (I1, I2, M11)

THE RULE THIS PIPELINE EXISTS TO ENFORCE: never write a plausible constant for
missing data. The previous version answered an empty collection with
constant([0,0]) for reflectance and constant(273.15) -- 0 C -- for LST, and both
are indistinguishable from a measurement once written. That silently killed 83%
of the reflectance and 34% of the LST without a single error message. Here a
hole is written as NaN and recorded in provenance.json, and the loader imputes
it with the channel mean (see model/utils/utils.py:impute_missing).

SOURCES, and why each one:

  fire      NASA/LANCE/SNPP_VIIRS/C2, Bright_ti5 > 279 K. VIIRS only, by
            design: this is the target channel and it stays single-sensor.
  LST       MODIS/061/MYD11A1 (Aqua, ~13:30 local), NOT MOD11A1 (Terra,
            ~10:30). Measured: Aqua runs +10.1 C warmer than Terra, so the two
            cannot be mixed in one channel; and 13:30 is closer to the hour a
            fire makes its run, with more thermal contrast between burn scar
            and live fuel. Coverage is 68% vs Terra's 70% -- a rounding error.
  LST hole  ERA5-Land skin_temperature sampled at 13:30 LOCAL (UTC derived from
            longitude). Measured error against real Aqua LST: 4.0 C at 13:30
            versus 11.2 C as a daily mean and 9.06 C for imputing the global
            mean. The clock dominates everything else in this channel. ERA5 is
            9 km against a 7.7 km patch, so it is flat inside the crop -- but
            the real intra-patch variation is only 2.3 C, smaller than ERA5's
            own 4.0 C error, so resampling it to 1 km changes nothing (measured
            identical) and a DEM/NDVI downscaling would polish noise.
  weather   ECMWF/ERA5_LAND/HOURLY, CHIRPS daily for precipitation.
  reflect   NASA/VIIRS/002/VNP09GA -- Collection 2. Collection 1 stopped being
            produced on 2024-06-16, which is what the old fallback was hiding.
            Collection 2 serves I1/I2 already as reflectance floats: do NOT
            apply the x0.0001 rescale of the old code.
  static    SRTM elevation, MODIS MOD13Q1 (NDVI/EVI), MCD12Q1 (land cover),
            MOD16A2 (ET). One pair of rasters per event, phase "static".

WHAT STAYS UPSTREAM: which fires are in the dataset. Phase "events" turns a
GeoJSON of contours into the directory skeleton, but producing that GeoJSON --
MODIS burned-area seeding, the cropland and duration filters, the clustering --
is a research decision and lives in create-dataset/get_fires.py and
expansion/. Give this script the agreed list and it builds everything else.

Kept single-sensor on purpose: the fire mask is VIIRS and nothing else, and the
LST stays inside MODIS + ERA5, the two families already present in climate.tif.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import shutil
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import Resampling, reproject

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
from paths import EE_PROJECT as PROJECT, AFSEQ as DEFAULT_OUT  # noqa: E402
FIRE_PRODUCT = "NASA/LANCE/SNPP_VIIRS/C2"
FIRE_PRODUCT_N20 = "NASA/LANCE/NOAA20_VIIRS/C2"   # same instrument, 50 min apart
LST_PRODUCT = "MODIS/061/MYD11A1"          # Aqua 13:30, see module docstring
LST_BAND = "LST_Day_1km"
LST_HOUR_LOCAL = 13.5
ERA5 = "ECMWF/ERA5_LAND/HOURLY"
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
REFL_PRODUCT = "NASA/VIIRS/002/VNP09GA"
SCALE_FIRE = 200
SCALE_CLIMATE = 200
SCALE_REFL = 500
MAX_TRIES = 5

_print_lock = threading.Lock()

# Two Earth Engine queries per day return the SAME value for every day of an
# event: the centroid (which fixes the UTC hour) and the pixel count of the
# crop. Repeating them per day was a third of the request budget for nothing.
_event_cache = {}
_cache_lock = threading.Lock()


def event_const(event_id, key, compute):
    """Memoize a per-event constant across the whole run, thread-safe."""
    ck = (event_id, key)
    with _cache_lock:
        if ck in _event_cache:
            return _event_cache[ck]
    value = compute()
    with _cache_lock:
        _event_cache[ck] = value
    return value


class NoData(RuntimeError):
    """The collection has nothing for this day and region. Never faked."""


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def retry(fn, *a, **kw):
    """Run fn with exponential backoff on transient Earth Engine failures."""
    for attempt in range(MAX_TRIES):
        try:
            return fn(*a, **kw)
        except NoData:
            raise
        except Exception as e:                             # noqa: BLE001
            if "http 4" in str(e):          # 400/403/404: not transient
                raise
            if attempt == MAX_TRIES - 1:
                raise
            time.sleep(min(60, 2 ** attempt) * (1 + random.random()))


def download(img, region, scale, ref_path=None, resampling=Resampling.bilinear):
    """GeoTIFF bytes for img, optionally reprojected onto ref_path's grid."""
    url = img.reproject(crs="EPSG:4326", scale=scale).getDownloadURL(
        {"scale": scale, "region": region, "format": "GEO_TIFF"})
    r = requests.get(url, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
    if ref_path is None:
        return r.content
    with rasterio.open(ref_path) as d:
        prof = d.profile.copy()
        shape, transform, crs = (d.height, d.width), d.transform, d.crs
    with rasterio.open(io.BytesIO(r.content)) as s:
        out = np.zeros((s.count, *shape), dtype="float64")
        for b in range(s.count):
            reproject(source=rasterio.band(s, b + 1), destination=out[b],
                      src_transform=s.transform, src_crs=s.crs,
                      dst_transform=transform, dst_crs=crs,
                      resampling=resampling)
    prof.update(count=out.shape[0], dtype="float64")
    return out, prof


def write_tif(path: Path, arr, profile):
    """Atomic write: a crash never leaves a half raster that reads as data."""
    profile = dict(profile)
    profile.update(count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
                   dtype="float64")
    tmp = path.with_suffix(".tmp")
    with rasterio.open(tmp, "w", **profile) as w:
        w.write(arr.astype("float64"))
    tmp.rename(path)


# ----------------------------------------------------------------- fire mask
def fetch_fire(region, date, out_path, ref_path=None):
    """The DAILY ACTIVE-FIRE mask: VIIRS detections of that one day only.

    SNPP and NOAA-20 together: same instrument, same 375 m, two looks per day
    instead of one (recovers 11.5% of the gaps that a single satellite enters
    as "no fire" when it means "not observed"). This is the target and the
    fire channel of every model; there is no cumulative product any more.
    Pixels with no detection are 0, never NaN: absence of fire is a value.
    """
    start = ee.Date(date)
    col = (ee.ImageCollection(FIRE_PRODUCT).filterBounds(region)
           .filterDate(start, start.advance(1, "day"))
           .merge(ee.ImageCollection(FIRE_PRODUCT_N20).filterBounds(region)
                  .filterDate(start, start.advance(1, "day"))))
    if retry(lambda: col.size().getInfo()) == 0:
        raise NoData("sin paso de VIIRS")
    fire = col.map(lambda im: im.select("Bright_ti5").gt(279).rename("burned")).mosaic()
    # Band 2: hour of detection (UTC, 0-23), the latest pass where SNPP and
    # NOAA-20 both saw the pixel; 0 where there is no fire, as WSTS stores
    # it. NOT a model channel (dataset/afseq_loader.CHANNELS stays at 34):
    # only the cross-benchmark fusion reads it, to fill the one Veg slot of
    # WSTS+ that AFSeq had no counterpart for.
    hour = (col.map(lambda im: im.select("acq_time").divide(3600).floor()
                    .updateMask(im.select("Bright_ti5").gt(279)).rename("hour"))
            .reduce(ee.Reducer.max()).rename("hour"))
    # unmask(0): otherwise EE writes -inf (nodata) on edge pixels the clip
    # leaves partially outside; the loader binarises with > 0.5 either way
    mask = (ee.Image.constant(0).rename("burned").blend(fire)
            .addBands(ee.Image.constant(0).rename("hour").blend(hour))
            .clip(region).unmask(0).toFloat())
    if ref_path is not None and Path(ref_path).exists():
        # ON THE EVENT'S EXISTING GRID. A fresh download lands on a grid EE
        # rounds on its own, one pixel off the earlier layers for some
        # events (310 vs 309 columns on id_0001 when fire was re-downloaded
        # for the hour band). Nearest neighbour: both bands are categorical.
        arr, prof = retry(download, mask, region, SCALE_FIRE, ref_path, Resampling.nearest)
    else:
        data = retry(download, mask, region, SCALE_FIRE)
        with rasterio.open(io.BytesIO(data)) as s:
            arr, prof = s.read().astype("float64"), s.profile.copy()
    write_tif(out_path, arr, prof)
    return "viirs_snpp+n20+hour"


# -------------------------------------------------------------------- climate
def utc_hour_for(region, local_hour, event_id):
    """Local solar hour -> UTC hour, from longitude at 15 deg per hour.

    Cached per event: every day of a fire shares the same footprint, so the
    centroid never changes and asking Earth Engine again is pure latency.
    """
    lon = event_const(event_id, "lon",
                      lambda: retry(
                          lambda: region.centroid(1).coordinates().getInfo())[0])
    return int(round(local_hour - lon / 15.0)) % 24


def fetch_climate(region, date, out_path, ref_path, event_id):
    """7 bands: temp_c, dew_c, rh, wind_ms, wind_deg, pressure_hpa, precip_mm.

    Band 0 is Aqua LST when Aqua sees the ground, else ERA5 skin_temperature at
    13:30 local, else NaN. Provenance is returned so the choice is auditable.
    """
    d0 = ee.Date(date)
    d1 = d0.advance(1, "day")
    era_day = ee.ImageCollection(ERA5).filterDate(d0, d1).filterBounds(region)
    if retry(lambda: era_day.size().getInfo()) == 0:
        raise NoData("ERA5 sin cobertura")

    temp_era = era_day.select("temperature_2m").mean()
    dew = era_day.select("dewpoint_temperature_2m").mean()
    u = era_day.select("u_component_of_wind_10m").mean()
    v = era_day.select("v_component_of_wind_10m").mean()
    sp = era_day.select("surface_pressure").mean()

    # --- band 0: LST, Aqua first, ERA5 at the SAME hour as fallback ---------
    # A present image is NOT a present measurement: MYD11A1 ships fully
    # cloud-masked rasters, which download as all-zero and would land as a
    # brand new faked constant -- the exact bug this pipeline exists to kill.
    # So the valid-pixel fraction is checked before the band is accepted.
    aqua = (ee.ImageCollection(LST_PRODUCT).filterDate(d0, d1)
            .filterBounds(region))
    source = "nan"
    lst = None
    if retry(lambda: aqua.size().getInfo()) > 0:
        cand = aqua.mean().select(LST_BAND)
        valid = retry(lambda: cand.reduceRegion(
            ee.Reducer.count(), region, 1000, maxPixels=1e7,
            bestEffort=True).getInfo())
        total = event_const(event_id, "npix", lambda: retry(
            lambda: ee.Image.constant(1).reduceRegion(
                ee.Reducer.count(), region, 1000, maxPixels=1e7,
                bestEffort=True).getInfo()))
        nv = next(iter(valid.values()), 0) or 0
        nt = total.get("constant", 1) or 1
        if float(nv) / float(max(nt, 1)) > 0.5:
            lst = cand.multiply(0.02).subtract(273.15)
            source = "aqua"
    if lst is None:
        utc = utc_hour_for(region, LST_HOUR_LOCAL, event_id)
        hourly = (ee.ImageCollection(ERA5)
                  .filterDate(d0.advance(utc, "hour"),
                              d0.advance(utc + 1, "hour"))
                  .filterBounds(region))
        if retry(lambda: hourly.size().getInfo()) > 0:
            lst = hourly.select("skin_temperature").mean().subtract(273.15)
            source = "era5_1330"          # reanalysis: no cloud gaps by design
        else:
            lst = ee.Image.constant(float("nan"))          # declared, not faked
    lst = lst.rename("temperatura_c")

    chirps = ee.ImageCollection(CHIRPS).filterDate(d0, d1).filterBounds(region)
    precip = (chirps.mean().select("precipitation").rename("precipitacion_mm")
              if retry(lambda: chirps.size().getInfo()) > 0
              else ee.Image.constant(float("nan")).rename("precipitacion_mm"))

    rh = temp_era.expression(
        "100 * exp(17.625 * td / (243.04 + td)) / exp(17.625 * t / (243.04 + t))",
        {"t": temp_era.subtract(273.15), "td": dew.subtract(273.15)}
    ).rename("humedad_relativa")

    combined = ee.Image.cat([
        lst,
        dew.subtract(273.15).rename("punto_rocio_c"),
        rh,
        u.hypot(v).rename("viento_velocidad_ms"),
        u.atan2(v).multiply(180 / math.pi).add(180).rename("viento_direccion_grados"),
        sp.divide(100).rename("presion_hpa"),
        precip,
    ])
    if ref_path and Path(ref_path).exists():
        arr, prof = retry(download, combined, region, SCALE_CLIMATE, ref_path)
    else:
        data = retry(download, combined, region, SCALE_CLIMATE)
        with rasterio.open(io.BytesIO(data)) as s:
            arr, prof = s.read().astype("float64"), s.profile.copy()
    # Last line of defence, band by band. ERA5-Land has no data over water, so
    # a coastal patch (Calabria, the Aegean) can come back empty for EVERY
    # band, not just LST -- and an empty band written as 0 is a faked constant
    # again: 0 C, 0 m/s, 0 hPa. Any band that is a flat zero is declared NaN.
    # ONLY when the WHOLE raster is empty. A single flat-zero band is usually
    # legitimate: precipitation is exactly 0 on a dry day, and CHIRPS is daily
    # and global, so that zero is a measurement (76% of fire-days are dry).
    # Declaring those NaN would turn "it did not rain" into "unknown", which is
    # the opposite mistake to the one this pipeline was written to fix.
    todas_vacias = all(np.all(np.abs(np.nan_to_num(arr[b])) < 1e-9)
                       for b in range(arr.shape[0]))
    if todas_vacias:
        # ERA5-Land has no data over water: a coastal patch mostly over sea
        # comes back empty in every band. Nothing here is a measurement.
        arr[:] = np.nan
        source = f"nan (descartado {source}: raster entero vacio)"
    elif arr.shape[0] > 0 and np.all(np.abs(np.nan_to_num(arr[0])) < 1e-9):
        arr[0] = np.nan                      # LST alone: the 273.15 fallback
        source = f"nan (descartado {source}: banda LST vacia)"
    write_tif(out_path, arr, prof)
    return source


# ---------------------------------------------------------------- reflectance
def fetch_reflectance(region, date, out_path, ref_path):
    d0 = ee.Date(date)
    # I1 (0.64 um) and I2 (0.86 um) on the 500 m grid, M11 (1.6 um SWIR) on
    # the 1 km grid: the three VNP09GA bands WSTS uses, in the order the
    # loader unpacks them. M11 was added on 2026-08-25 (it was the one Veg
    # channel AFSeq lacked; exp4 had to fill it with a constant).
    col = (ee.ImageCollection(REFL_PRODUCT).filterDate(d0, d0.advance(1, "day"))
           .filterBounds(region).select(["I1", "I2", "M11"]))
    if retry(lambda: col.size().getInfo()) == 0:
        raise NoData("VNP09GA sin imagen")
    # Collection 2 already serves reflectance floats -- no x0.0001 here
    img = col.mean().rename(["I1", "I2", "M11"])
    if ref_path and Path(ref_path).exists():
        arr, prof = retry(download, img, region, SCALE_REFL, ref_path)
    else:
        data = retry(download, img, region, SCALE_REFL)
        with rasterio.open(io.BytesIO(data)) as s:
            arr, prof = s.read().astype("float64"), s.profile.copy()
    if float(np.nanmax(np.abs(arr))) == 0.0:
        raise NoData("imagen vacia")                       # never stage a zero
    write_tif(out_path, arr, prof)
    return "vnp09ga_c2"



# ------------------------------------------------------------------ static
def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def fetch_static(region, event_dir: Path, force=False, ref=None):
    """dem.tif and vegetation.tif, one pair per event (they never change).

    Both replicate the ORIGINAL feature scripts (features/dem.py and
    features/vegetation.py), band for band, because the loader unpacks them
    positionally (afseq_loader.normalize_dem / normalize_vegetation):

      dem.tif         3 bands, float32, ON THE FIRE GRID (200 m, EPSG:4326):
                      elevation (m), slope (deg), aspect (deg, 0 = N,
                      clockwise). SRTM GL1 v003 downloaded in EPSG:4326 at
                      30 m, reprojected to UTM (bilinear), slope and aspect
                      by Horn's kernels / (8 * pixel size) at 30 m, then
                      block-averaged onto the fire grid (Resampling.average).
      vegetation.tif  4 bands at 200 m, EPSG:4326, in THIS order:
                      land cover (MCD12Q1 LC_Type1, mode of the two years
                      before the event), NDVI, EVI (MOD13Q1 mean of the 30
                      days before start_date, raw x1e4), ET (MOD16A2 mean of
                      the same 30 days).

    2026-08-25: the first version of this function wrote elevation alone and
    the vegetation in the order [NDVI, EVI, landcover, ET], averaged over the
    whole collection. Nothing caught it until dataset_v7 was the first tree
    built entirely by this file: every day failed to unpack the DEM and the
    fire channel came out empty. The band order is now asserted at write time.
    No fallback: if a product has nothing for this footprint the event is
    reported, not filled.
    """
    from scipy.ndimage import convolve
    from rasterio.warp import calculate_default_transform
    made = []
    info = json.loads((event_dir / "info.geojson").read_text())
    props = info.get("properties") or info["features"][0]["properties"]
    start = datetime.strptime(props["start_date"], "%Y-%m-%d")

    dem_p = event_dir / "dem.tif"
    if force or not dem_p.exists():
        srtm = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(region)
        # 30 m, or coarser for the few events whose 30 m DEM exceeds EE's
        # 50 MB download cap (HTTP 400): 60 m, then 90 m. Slope and aspect
        # are then computed at that resolution; the result is averaged onto
        # the 200 m fire grid either way, and the scale used is returned in
        # the provenance.
        dem_scale = None
        for sc in (30, 60, 90):
            try:
                data = retry(download, srtm, region, sc)
                dem_scale = sc
                break
            except RuntimeError as e:
                if "400" not in str(e):
                    raise
        if dem_scale is None:
            raise NoData("DEM: incluso a 90 m supera el tope de descarga de EE")
        with rasterio.open(io.BytesIO(data)) as s_:
            dem = s_.read(1).astype("float32")
            src_tr, src_crs, b = s_.transform, s_.crs, s_.bounds
        dst_crs = f"EPSG:{utm_epsg((b.left + b.right) / 2, (b.top + b.bottom) / 2)}"
        dst_tr, w, h = calculate_default_transform(src_crs, dst_crs, dem.shape[1],
                                                   dem.shape[0], *b)
        dem_utm = np.zeros((h, w), dtype="float32")
        reproject(source=dem, destination=dem_utm, src_transform=src_tr,
                  src_crs=src_crs, dst_transform=dst_tr, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
        dx, dy = dst_tr.a, -dst_tr.e
        dzdx = convolve(dem_utm, kx, mode="reflect") / (8.0 * dx)
        dzdy = convolve(dem_utm, ky, mode="reflect") / (8.0 * dy)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        aspect = (np.degrees(np.arctan2(dzdy, -dzdx)) + 360) % 360
        arr = np.stack([dem_utm, slope, aspect]).astype("float32")
        assert arr.shape[0] == 3
        if ref is not None:
            # STORED ON THE FIRE GRID (200 m). Slope and aspect are computed
            # at 30 m in UTM, where they mean something, then block-averaged
            # onto the event's grid: that average is all the loader ever used
            # (downsample_average), and keeping the 30 m raster made the DEM
            # 80% of the dataset (13 GB of 16 at 128-px windows, ~50 GB at
            # 256). Same numbers, a fraction of the bytes.
            with rasterio.open(ref) as r_:
                prof = r_.profile.copy(); shape, tr_out, crs_out = (r_.height, r_.width), r_.transform, r_.crs
            out = np.zeros((3, *shape), dtype="float32")
            for b in range(3):
                reproject(source=arr[b], destination=out[b], src_transform=dst_tr,
                          src_crs=dst_crs, dst_transform=tr_out, dst_crs=crs_out,
                          resampling=Resampling.average)
            prof.update(count=3, dtype="float32", compress="deflate", nodata=None)
            with rasterio.open(dem_p, "w", **prof) as d:
                d.write(out)
                d.descriptions = ("elevation_m", "slope_deg", "aspect_deg")
        else:
            prof = dict(driver="GTiff", height=h, width=w, count=3, dtype="float32",
                        crs=dst_crs, transform=dst_tr, compress="deflate")
            with rasterio.open(dem_p, "w", **prof) as d:
                d.write(arr)
                d.descriptions = ("elevation_m", "slope_deg", "aspect_deg")
        made.append("dem" if dem_scale == 30 else f"dem@{dem_scale}m")

    veg_p = event_dir / "vegetation.tif"
    if force or not veg_p.exists():
        d0 = (start - timedelta(days=30)).strftime("%Y-%m-%d")
        d1 = start.strftime("%Y-%m-%d")
        d_lc = (start - timedelta(days=730)).strftime("%Y-%m-%d")
        mod13 = ee.ImageCollection("MODIS/061/MOD13Q1").filterBounds(region).filterDate(d0, d1)
        mcd12 = ee.ImageCollection("MODIS/061/MCD12Q1").filterBounds(region).filterDate(d_lc, d1)
        # MCD12Q1 is annual and its last image is dated 2024-01-01: for an
        # event starting after 2026-01-31 the two-year window opens after it
        # and matches nothing, and ee.Image.cat would silently drop the band
        # (76 events of 2026 in dataset_v7). Same rule the expansion campaign
        # used: fall back to the two most recent images (mode of the two).
        if retry(lambda: mcd12.size().getInfo()) == 0:
            mcd12 = (ee.ImageCollection("MODIS/061/MCD12Q1").filterBounds(region)
                     .sort("system:time_start", False).limit(2))
        mod16 = ee.ImageCollection("MODIS/061/MOD16A2").filterBounds(region).filterDate(d0, d1)
        if retry(lambda: mod13.size().getInfo()) == 0:
            raise NoData("MOD13Q1 sin cobertura en los 30 dias previos")
        veg = ee.Image.cat([
            mcd12.select("LC_Type1").mode().rename("landcover"),
            mod13.select("NDVI").mean().rename("NDVI"),
            mod13.select("EVI").mean().rename("EVI"),
            mod16.select("ET").mean().rename("ET"),
        ]).clip(region)
        if ref is not None:
            # ON THE FIRE GRID. Downloaded on its own, EE hands vegetation a
            # grid that is off by one row or column from the fire raster's
            # for 60% of the events (float rounding of the region bounds),
            # and get_features cannot stack them. Nearest neighbour: land
            # cover is categorical, and nearest is what EE's own reprojection
            # did for every layer of the earlier datasets.
            # Scale ladder: for the largest windows EE answers "User memory
            # limit exceeded" (HTTP 400) at 200 m; 250 m is MOD13Q1's native
            # grid and 500 m MCD12Q1/MOD16A2's, so nothing real is lost
            # there. The scale used goes into the provenance.
            veg_scale = None
            for sc in (200, 250, 500):
                try:
                    arr, prof = retry(download, veg, region, sc, ref, Resampling.nearest)
                    veg_scale = sc
                    break
                except RuntimeError as e:
                    if "400" not in str(e):
                        raise
            if veg_scale is None:
                raise NoData("vegetacion: EE sin memoria incluso a 500 m")
        else:
            veg_scale = 200
            data = retry(download, veg, region, 200)
            with rasterio.open(io.BytesIO(data)) as s_:
                arr, prof = s_.read().astype("float64"), s_.profile.copy()
        # EE's GeoTIFF carries no band names; the order is ee.Image.cat's
        assert arr.shape[0] == 4, arr.shape
        write_tif(veg_p, arr, prof)
        with rasterio.open(veg_p, "r+") as d:
            d.descriptions = ("landcover", "NDVI", "EVI", "ET")
        made.append("vegetation" if veg_scale == 200 else f"vegetation@{veg_scale}m")
    return "+".join(made) if made else "ya estaba"


def run_static(out: Path, workers: int, force: bool):
    events = [out / e for e in sorted(os.listdir(out))
              if e.startswith("id_") and (out / e).is_dir()]
    todo = [e for e in events
            if force or not (e / "dem.tif").exists()
            or not (e / "vegetation.tif").exists()]
    log(f"fase static: {len(todo)} eventos de {len(events)}")
    if not todo:
        return {}
    stats, lock = {}, threading.Lock()

    def one(edir):
        region, ref = region_of(edir)      # ref: a day raster, the fire grid
        return fetch_static(region, edir, force, ref)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, e): e for e in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:                        # noqa: BLE001
                res = f"error: {str(e)[:50]}"
            with lock:
                k = res.split(":")[0]
                stats[k] = stats.get(k, 0) + 1
            if i % 100 == 0:
                log(f"  [{i}/{len(todo)}]  " + str(stats))
    log(f"fase static terminada: {stats}")
    return stats


# ------------------------------------------------------------------ events
MARGIN_KM = 10.0       # around the final contour, every side
MIN_PX = 256           # window never smaller than this, at SCALE_FIRE


def window_of(geometry, margin_km: float = MARGIN_KM, min_px: int = MIN_PX,
              scale_m: int = SCALE_FIRE) -> list:
    """[west, south, east, north] of the event window, in degrees.

    THE WINDOW IS NOT THE SCAR. Until 2026-08-25 each event's rasters covered
    its final contour plus 0.01 deg, so the raster extent itself told a model
    where the fire would end, and 91% of the events were smaller than the
    64-px training patch, which the loader filled with zeros. Now: the
    contour's bounding box, `margin_km` on every side, and never less than
    `min_px` pixels of `scale_m` in either axis (256 px = 51 km: twice the
    128-px patch, the window:patch ratio of WSTS+, whose ~284-px windows hold
    128-px crops), grown symmetrically about the box centre. A window twice
    the patch is what makes the fire-centred crop's jitter real.
    """
    c = geometry["coordinates"]
    pts = []
    stack = [c]
    while stack:
        x = stack.pop()
        if isinstance(x[0], (int, float)):
            pts.append((float(x[0]), float(x[1])))
        else:
            stack.extend(x)
    lons, lats = zip(*pts)
    w, e, s, n = min(lons), max(lons), min(lats), max(lats)
    lat0 = (s + n) / 2.0
    km_lat = 110.54
    km_lon = 111.32 * math.cos(math.radians(lat0))
    w, e = w - margin_km / km_lon, e + margin_km / km_lon
    s, n = s - margin_km / km_lat, n + margin_km / km_lat
    min_km = min_px * scale_m / 1000.0
    if (e - w) * km_lon < min_km:
        cx = (w + e) / 2.0
        w, e = cx - min_km / km_lon / 2, cx + min_km / km_lon / 2
    if (n - s) * km_lat < min_km:
        cy = (s + n) / 2.0
        s, n = cy - min_km / km_lat / 2, cy + min_km / km_lat / 2
    return [w, s, e, n]


def _make_event(out: Path, fid: str, feature: dict, days: list,
                margin_km: float, min_px: int) -> None:
    edir = out / fid
    edir.mkdir(parents=True, exist_ok=True)
    (edir / "contour.geojson").write_text(json.dumps(feature))
    props = dict(feature.get("properties", {}))
    props["window"] = window_of(feature["geometry"], margin_km, min_px)
    props["window_rule"] = f"bbox + {margin_km} km, min {min_px} px @ {SCALE_FIRE} m"
    (edir / "info.geojson").write_text(json.dumps(
        {"id": fid, "geometry": feature["geometry"], "properties": props}))
    for day in days:
        (edir / day).mkdir(exist_ok=True)


def run_events(out: Path, contours: Path, margin_km: float = MARGIN_KM,
               min_px: int = MIN_PX):
    """Create the event skeleton from a GeoJSON of fire contours.

    Each feature needs an id, a geometry, and start_date / end_date in its
    properties -- which is what create-dataset/get_fires.py and the expansion
    census produce. THAT selection (which fires are in the dataset, after the
    cropland, duration and clustering filters) is a research decision and stays
    upstream: this phase turns an agreed list into directories, it does not
    decide the list. The raster window of each event is `window_of`.
    """
    gj = json.loads(contours.read_text())
    feats = gj["features"] if "features" in gj else gj
    made = 0
    for f in feats:
        fid = str(f.get("id") or f["properties"].get("id"))
        if not fid.startswith("id_"):
            fid = f"id_{int(fid):04d}"
        props = f.get("properties", {})
        d0 = datetime.strptime(props["start_date"], "%Y-%m-%d")
        d1 = datetime.strptime(props["end_date"], "%Y-%m-%d")
        days, day = [], d0
        while day <= d1:
            days.append(day.strftime("%Y-%m-%d")); day += timedelta(days=1)
        _make_event(out, fid, f, days, margin_km, min_px)
        made += 1
    log(f"fase events: {made} eventos creados en {out}")
    return {"events": made}


def run_events_from_tree(out: Path, src: Path, margin_km: float = MARGIN_KM,
                         min_px: int = MIN_PX, inherit: bool = True):
    """Same skeleton, but from an existing tree: its ids, contours and the
    days it actually has (not start/end: holes stay holes). This is how the
    1,194 events of dataset_v6 were re-downloaded into dataset_v7."""
    made = 0
    for edir in sorted(p for p in Path(src).iterdir()
                       if p.is_dir() and p.name.startswith("id_")):
        feature = json.loads((edir / "contour.geojson").read_text())
        if "features" in feature:                 # a one-feature collection
            feature = feature["features"][0]
        days = sorted(p.name for p in edir.iterdir()
                      if p.is_dir() and p.name[:4].isdigit())
        _make_event(out, edir.name, feature, days, margin_km, min_px)
        made += 1
    # A curated source stays curated: its exclusion record travels with the
    # ids, so `--phase curate` on the new tree is a no-op instead of taking
    # another 15% of what is left (which is what happened on 2026-08-25).
    src_ex = Path(src) / "excluded.json"
    if inherit and src_ex.exists():
        rec = json.loads(src_ex.read_text())
        rec["inherited_from"] = str(src_ex)
        (out / "excluded.json").write_text(json.dumps(rec, indent=1))
        log(f"fase events: excluded.json heredado de {src} "
            f"({len(rec.get('events', {}))} eventos ya excluidos)")
    log(f"fase events: {made} eventos recreados en {out} desde {src}")
    return {"events": made}

# --------------------------------------------------------------------- driver
def region_of(event_dir: Path):
    """Event footprint, from the day rasters if present, else info.geojson."""
    for day in sorted(event_dir.iterdir()):
        if not day.is_dir():
            continue
        for name in ("active-fire.tif", "climate.tif", "reflectance.tif"):
            p = day / name
            if p.exists():
                with rasterio.open(p) as r:
                    b = r.bounds
                return ee.Geometry.Rectangle([b.left, b.bottom, b.right, b.top]), p
    info = event_dir / "info.geojson"
    if info.exists():
        g = json.loads(info.read_text())
        win = g.get("properties", {}).get("window")
        if win:                                   # written by window_of
            return ee.Geometry.Rectangle(win), None
        geom = g.get("geometry", g)               # legacy trees: the contour
        return ee.Geometry(geom).bounds(), None
    raise NoData("sin geometria")


NEEDS = {
    "fire": ("active-fire.tif", fetch_fire),
    "climate": ("climate.tif", fetch_climate),
    "reflect": ("reflectance.tif", fetch_reflectance),
}


def sick(path: Path, kind: str) -> bool:
    """Is this raster missing or holding a faked constant?"""
    if not path.exists():
        return True
    try:
        with rasterio.open(path) as r:
            a = r.read()
    except Exception:                                      # noqa: BLE001
        return True
    if kind == "reflect":
        return float(np.nanmax(np.abs(np.nan_to_num(a)))) == 0.0
    if kind == "climate":
        # the old 273.15 K fallback lands as exactly 0.0 across band 0
        return a.shape[0] > 0 and np.all(np.abs(np.nan_to_num(a[0])) < 1e-9)
    return False


def run_phase(out: Path, phase: str, workers: int, limit: int, force: bool):
    fname, fetcher = NEEDS[phase]
    jobs = []
    for ev in sorted(os.listdir(out)):
        edir = out / ev
        if not edir.is_dir() or not ev.startswith("id_"):
            continue
        for day in sorted(os.listdir(edir)):
            ddir = edir / day
            if not ddir.is_dir() or not day[:4].isdigit():
                continue
            target = ddir / fname
            if force or sick(target, phase):
                jobs.append((ev, day, target))
            if limit and len(jobs) >= limit:
                break
        if limit and len(jobs) >= limit:
            break
    log(f"fase {phase}: {len(jobs)} dias por (re)construir")
    if not jobs:
        return {}

    prov, stats = {}, {}
    lock = threading.Lock()
    regions = {}
    pfile = out / "provenance.json"

    def flush_provenance():
        """Write provenance incrementally: a crash at hour two must not lose
        the record of which source fed which day."""
        existing = json.loads(pfile.read_text()) if pfile.exists() else {}
        existing.setdefault(phase, {}).update(prov)
        tmp = pfile.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, indent=1))
        tmp.rename(pfile)

    def one(ev, day, target):
        with lock:
            if ev not in regions:
                regions[ev] = region_of(out / ev)
        region, ref = regions[ev]
        if phase == "fire":
            # the grid reference for a fire re-download is any OTHER layer
            # of the event (they were put on the first fire raster's grid);
            # on a fresh build none exists yet and fire defines the grid
            others = [out / ev / day / "climate.tif", out / ev / day / "reflectance.tif",
                      out / ev / "vegetation.tif", out / ev / "dem.tif"]
            ref_fire = next((p for p in others if p.exists()), None)
            return fetcher(region, day, target, ref_fire)
        if phase == "climate":
            return fetcher(region, day, target, ref, ev)
        return fetcher(region, day, target, ref)

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for fu in as_completed(futs):
            ev, day, _ = futs[fu]
            try:
                res = fu.result()
            except NoData as e:
                res = f"hueco: {e}"
            except Exception as e:                         # noqa: BLE001
                res = f"error: {str(e)[:60]}"
            with lock:
                prov[f"{ev}/{day}"] = res
                k = res.split(":")[0]
                stats[k] = stats.get(k, 0) + 1
                done += 1
                if done % 200 == 0 or done == len(jobs):
                    rate = done / max(time.time() - t0, 1e-9)
                    log(f"  [{done}/{len(jobs)}] "
                        + "  ".join(f"{a}={b}" for a, b in sorted(stats.items()))
                        + f"  {rate:.1f}/s  eta "
                          f"{(len(jobs)-done)/max(rate,1e-9)/60:.0f} min")
                    flush_provenance()
    flush_provenance()
    log(f"fase {phase} terminada: {stats}")
    return stats


def audit(out: Path):
    import collections
    c = collections.Counter()
    for ev in sorted(os.listdir(out)):
        edir = out / ev
        if not edir.is_dir() or not ev.startswith("id_"):
            continue
        for day in sorted(os.listdir(edir)):
            ddir = edir / day
            if not ddir.is_dir() or not day[:4].isdigit():
                continue
            c["dias"] += 1
            for phase, (fname, _) in NEEDS.items():
                p = ddir / fname
                if not p.exists():
                    c[f"{phase}: falta"] += 1
                    continue
                with rasterio.open(p) as r:
                    a = r.read()
                if phase == "climate":
                    # band 6 is precipitation: a flat zero is a dry day,
                    # not a hole, so it is counted apart and never flagged
                    otras = [b for b in range(1, min(6, a.shape[0]))
                             if np.all(np.abs(np.nan_to_num(a[b])) < 1e-9)]
                    if otras:
                        c["climate: BANDAS ERA5 EN CERO (sospechoso)"] += 1
                    if a.shape[0] > 6 and np.all(np.abs(np.nan_to_num(a[6])) < 1e-9):
                        c["climate: precipitacion 0 (dia seco, normal)"] += 1
                    if np.isnan(a[0]).all():
                        c["climate: LST hueco declarado"] += 1
                    elif np.all(np.abs(np.nan_to_num(a[0])) < 1e-9):
                        c["climate: LST RELLENO FINGIDO"] += 1
                    else:
                        c["climate: LST real"] += 1
                elif phase == "reflect":
                    if np.isnan(a).all():
                        c["reflect: hueco declarado"] += 1
                    elif float(np.nanmax(np.abs(a))) == 0.0:
                        c["reflect: CERO FINGIDO"] += 1
                    else:
                        c["reflect: real"] += 1
                else:
                    c["fire: ok"] += 1
    d = c.pop("dias")
    log(f"\naudit: {d} dias-fuego")
    for k in sorted(c):
        log("  %-34s %6d  (%4.1f%%)" % (k, c[k], 100 * c[k] / max(d, 1)))
    if any("FINGIDO" in k for k in c):
        log("\n  AVISO: quedan rellenos fingidos, que este pipeline no debe producir")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="arbol del dataset (por defecto, paths.AFSEQ)")
    ap.add_argument("--phase", default="all",
                    choices=["all", "events", "static", "fire", "climate",
                             "reflect", "audit", "curate", "splits", "cache",
                             "stats"])
    ap.add_argument("--drop", nargs="+", metavar="REGION=FRAC", default=None,
                    help="curate: fraccion de eventos mas persistentes a excluir "
                         "por region (por defecto US=0.15, ver dataset/curate.py)")
    ap.add_argument("--contours", type=Path,
                    help="GeoJSON de contornos para --phase events")
    ap.add_argument("--from-tree", type=Path, nargs="+",
                    help="events: recrear los eventos (ids, contornos, dias) de "
                         "uno o mas arboles existentes con la ventana nueva")
    ap.add_argument("--fresh", action="store_true",
                    help="events --from-tree: NO heredar excluded.json, para que "
                         "--phase curate se calcule de nuevo sobre el arbol final")
    ap.add_argument("--margin-km", type=float, default=MARGIN_KM)
    ap.add_argument("--min-px", type=int, default=MIN_PX)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even rasters that look healthy")
    args = ap.parse_args()

    if args.phase == "events" and not (args.contours or args.from_tree):
        raise SystemExit("--phase events necesita --contours <geojson> o --from-tree <arbol>")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "curate":
        return run_curate(args)
    if args.phase in ("splits", "cache", "stats"):
        return run_post(args.out, args.phase)
    ee.Initialize(project=PROJECT)
    if args.phase == "audit":
        audit(args.out)
        return 0
    if args.phase == "events":
        if args.from_tree:
            for src in args.from_tree:
                run_events_from_tree(args.out, src, args.margin_km, args.min_px,
                                     inherit=not args.fresh)
        else:
            run_events(args.out, args.contours, args.margin_km, args.min_px)
        return 0
    # fire first: its rasters define the grid every other layer is put on
    phases = (["fire", "static", "climate", "reflect"] if args.phase == "all"
              else [args.phase])
    for ph in phases:
        if ph == "static":
            run_static(args.out, args.workers, args.force)
        else:
            run_phase(args.out, ph, args.workers, args.limit, args.force)
    if args.phase == "all":
        # ONE COMMAND, START TO FINISH. A second pass over the daily layers
        # rebuilds whatever Earth Engine answered badly the first time (sick()
        # finds it: missing, empty, faked constants), then the audit, the
        # curation, the splits, the decoded cache with its identity check and
        # the channel statistics. Every step is also its own --phase.
        for ph in ("fire", "climate", "reflect"):
            run_phase(args.out, ph, args.workers, args.limit, False)
        audit(args.out)
        run_curate(args)         # what leaves the dataset, and why: curate.py
        for step in ("splits", "cache", "stats"):
            run_post(args.out, step)
    else:
        audit(args.out)
    return 0


def run_post(out: Path, step: str):
    """The steps after the download, on the curated tree, with the release's
    own tools: splits (dataset/make_splits.py), the decoded cache and its
    identity check (dataset/afseq_cache.py), channel statistics
    (dataset/channel_stats.py). Each one reads AFSEQ_DATA, so it is pointed
    at `out` here."""
    os.environ["AFSEQ_DATA"] = str(out)
    os.environ.setdefault("AFSEQ_CACHE", str(out) + "_cache")
    cmd = {"splits": ["dataset/make_splits.py", "--seeds", "5", "7", "23"],
           "cache": ["dataset/afseq_cache.py", "--build", "--procs", "4"],
           "stats": ["dataset/channel_stats.py", "--days", "3000"]}[step]
    log(f"fase {step}: {' '.join(cmd)}")
    rc = subprocess.call([sys.executable, str(ROOT / cmd[0])] + cmd[1:],
                         cwd=str(ROOT))
    if step == "cache" and rc == 0:
        rc = subprocess.call([sys.executable, str(ROOT / "dataset/afseq_cache.py"),
                              "--check", "40"], cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f"fase {step} fallo (rc={rc})")
    return rc


def run_curate(args):
    """Duplicates out, the most persistent US events out; see dataset/curate.py."""
    from dataset.curate import DROP_DEFAULT, curate
    drop = ({k: float(v) for k, v in (d.split("=") for d in args.drop)}
            if args.drop else DROP_DEFAULT)
    curate(args.out, drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
