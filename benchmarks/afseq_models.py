#!/usr/bin/env python3
"""Their WSTS+ models, with the one change AFSeq's variable-length fires need.

Their repository stays byte-identical; the delta lives here, as a subclass.

--------------------------------------------------------------------------
THE CHANGE -- pad_mask through the temporal attention

    WSTS hands every model exactly `n_leading_observations` days, so a padded
    frame never exists and their `SMPTempModel.forward` has no reason to mask
    one. AFSeq fires have variable-length histories: 80% of training samples
    and 76% of test samples carry fewer than the 16-day maximum, and the AFSeq
    collate pads them with zero frames at the END of the sequence.

    Run their forward unchanged on that, and the LTAE attends over zero frames
    as if they were observations -- a straw-man version of their model, which
    would understate their architecture on our data. Both Garnot classes they
    vendor (`LTAE2d.forward(..., pad_mask=None)`, `Temporal_Aggregator.forward(
    ..., pad_mask=None)`) already accept a pad mask; their SMPTempModel simply
    never passes one.

    So: same architecture, same weights layout, same everything, plus a pad
    mask derived from the batch. This is the charitable port of their model,
    and it is what our earlier Res18 replica (with_res18/smptemp_active.py)
    already did -- the difference now is that the architecture itself comes
    from their repo instead of being re-implemented by us.

    Everything below the `# --- theirs` marker is their forward, verbatim,
    except for the two `pad_mask=` arguments and the fixed decoder call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

THEIRS = Path(__file__).resolve().parent / "wsts+"
if str(THEIRS) not in sys.path:
    sys.path.insert(0, str(THEIRS))

from src.models.SMPTempModel import SMPTempModel   # noqa: E402


class SMPTempModelPadded(SMPTempModel):
    """SMPTempModel that ignores zero-padded frames in its temporal attention."""

    @staticmethod
    def pad_mask_of(x: torch.Tensor) -> torch.Tensor:
        """(B, T) True where the whole frame is zero, i.e. it is padding.

        Same test as with_res18/smptemp_active.py. A genuine observation is
        never all-zero across all channels: AFSeq normalises into [0,1] with
        offsets, and its holes are NaN imputed to the channel mean, never 0.
        """
        return (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None) -> torch.Tensor:
        B, T, C, H, W = x.shape
        pad_mask = self.pad_mask_of(x)
        if not pad_mask.any():
            pad_mask = None          # exactly their path when nothing is padded

        # --- theirs (SMPTempModel.forward), with pad_mask threaded through ---
        doys = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        num_stages = len(self.model.encoder.out_channels)
        encoder_features = [[] for _ in range(num_stages)]
        for t in range(T):
            features = self.model.encoder(x[:, t, :, :, :])
            for i in range(num_stages):
                encoder_features[i].append(features[i])

        last_stage = torch.stack(encoder_features[-1], dim=1)
        aggregated_last, attn = self.ltae(last_stage, batch_positions=doys,
                                          pad_mask=pad_mask)          # <-- delta
        aggregated_skips = []
        for i in range(1, num_stages - 1):
            stage = torch.stack(encoder_features[i], dim=1)
            aggregated_skips.append(
                self.temporal_aggregator(stage, pad_mask=pad_mask,    # <-- delta
                                         attn_mask=attn))
        dummy = encoder_features[0][0]
        decoder_features = [dummy] + aggregated_skips + [aggregated_last]
        decoder_output = self.model.decoder(decoder_features)
        return self.model.segmentation_head(decoder_output)
