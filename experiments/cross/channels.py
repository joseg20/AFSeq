#!/usr/bin/env python3
"""The shared channel space the two benchmarks are fused in.

WHY VEG. AFSeq has 34 channels, WSTS+ has 40, and they are not subsets of one
another. To train ONE model on both, the two have to be expressed in the same
space, and the space has to be one the published checkpoints already live in --
otherwise the fused row cannot be compared with the WSTS+-only row of wstsplus.py.
That space is "Veg", the 7 slots their own cfgs use:

    0 m11   1 i2   2 i1   3 ndvi   4 evi2   5 fire hour   6 fire binary

WSTS+ needs no mapping: Veg IS a slice of its 40 processed channels, so its side
is `x[:, [0,1,2,3,4,38,39]]` and nothing else happens to it. That is what keeps
the fused experiment commensurable with wstsplus.py.

AFSeq needs the table below. Since 2026-08-25 all seven slots have a
counterpart: M11 joined the reflectance, and the detection hour travels as
band 2 of active-fire.tif and is appended by the loader when asked
(Seq2SeqActivePadded(with_fire_hour=True)) as channel FIRE_HOUR_CHANNEL. No
slot is filled with a constant any more; the "fill" kind stays for trees
built before the hour existed. This is stated rather than hidden: the
alternative (a smaller intersected space) would not be comparable with the
published checkpoints, which is the whole point of fusing in Veg. When reading
the fused rows, remember the model had that shortcut available.
"""
from __future__ import annotations

import torch

# AFSeq's channels, by name (dataset/afseq_loader.CHANNELS; 34 since M11
# and the one-hot land cover joined on 2026-08-25)
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from dataset.afseq_loader import CH as AF, FIRE_HOUR_CHANNEL  # noqa: E402

# WSTS+'s 40 processed channels
WS = dict(m11=0, i2=1, i1=2, ndvi=3, evi2=4, fire_hour=38, fire=39)

VEG = [0, 1, 2, 3, 4, 38, 39]      # the Veg slice of the 40
N_VEG = len(VEG)
FIRE_SLOT = 6                      # what UTAE-AR feeds back

# One entry per Veg slot, in slot order: (afseq channel | None, kind, note)
#   "z"    z-score the AFSeq channel with AFSeq statistics
#   "raw"  copy through untouched (binary 0/1 on both sides)
#   "fill" no counterpart; write 0 = the WSTS training mean of that channel
VEG_FROM_AFSEQ = [
    (AF["refl_m11"], "z",   "VIIRS M11 <- AFSeq M11 (VNP09GA, same band)"),
    (AF["refl_i2"], "z",    "VIIRS I2 <- AFSeq reflectance I2 (VNP09GA, proxy)"),
    (AF["refl_i1"], "z",    "VIIRS I1 <- AFSeq reflectance I1 (VNP09GA, proxy)"),
    (AF["ndvi"],    "z",    "NDVI <- NDVI, MODIS on both sides (exact)"),
    (AF["evi"],     "z",    "EVI2 <- AFSeq EVI: related indices, same MODIS scale"),
    (FIRE_HOUR_CHANNEL, "z", "detection hour <- band 2 of active-fire.tif (UTC hour, 0 = none)"),
    (AF["fire"],    "raw",  "binary active fire <- binary active fire (exact)"),
]


def afseq_to_veg(x: torch.Tensor, mean, std, drop_refl: bool = False):
    """(..., 35, h, w) AFSeq (with the hour) -> (..., 7, h, w) in the Veg space.

    drop_refl blanks the two reflectance slots instead of mapping them. In the
    v1 release those AFSeq channels are ~76% zeros -- the VNP09GA v001 product
    they came from was retired -- so after z-scoring they are nearly constant
    (train std 0.064 and 0.076 on a 0..1 range) and carry almost no signal,
    while WSTS has real data there. Neither choice is clean: mapping them feeds
    near-constant noise into channels the other benchmark fills with signal,
    and dropping them widens the confound above. Both are reported.
    """
    shape = list(x.shape)
    shape[-3] = N_VEG
    out = torch.zeros(shape, dtype=x.dtype, device=x.device)
    for slot, (src, kind, _) in enumerate(VEG_FROM_AFSEQ):
        if kind == "fill":
            continue                                # 0 = the WSTS training mean
        if drop_refl and slot in (1, 2):
            continue
        ch = x[..., src, :, :]
        out[..., slot, :, :] = ch if kind == "raw" else (ch - mean[src]) / std[src]
    return out


def wstsplus_to_veg(x: torch.Tensor):
    """(..., 40, h, w) WSTS+ -> (..., 7, h, w). A slice; nothing is invented."""
    return x[..., VEG, :, :]


if __name__ == "__main__":
    filled = sum(1 for _, k, _ in VEG_FROM_AFSEQ if k == "fill")
    print(f"AFSeq -> Veg: {N_VEG - filled}/{N_VEG} slots con contraparte, "
          f"{filled} rellenos con la media de WSTS")
    for i, (src, kind, note) in enumerate(VEG_FROM_AFSEQ):
        mark = {"fill": " [RELLENO]", "raw": " [binario]"}.get(kind, "")
        print(f"  slot {i}{mark:11s} {note}")
