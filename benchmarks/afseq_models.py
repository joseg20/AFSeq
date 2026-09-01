"""Their WSTS+ models, with the one change AFSeq's variable-length fires need.

Their repository stays byte-identical; the delta lives here, as a subclass.

--------------------------------------------------------------------------
THE CHANGE: pad_mask through the temporal attention

    WSTS hands every model exactly `n_leading_observations` days, so a padded
    frame never exists and their `SMPTempModel.forward` has no reason to mask
    one. AFSeq fires have variable-length histories: 80% of training samples
    and 76% of test samples carry fewer than the 16-day maximum, and the AFSeq
    collate pads them with zero frames at the END of the sequence.

    Run their forward unchanged on that, and the LTAE attends over zero frames
    as if they were observations, a straw-man version of their model, which
    would understate their architecture on our data. Both Garnot classes they
    vendor (`LTAE2d.forward(..., pad_mask=None)`, `Temporal_Aggregator.forward(
    ..., pad_mask=None)`) already accept a pad mask; their SMPTempModel simply
    never passes one.

    So: same architecture, same weights layout, same everything, plus a pad
    mask derived from the batch. This is the charitable port of their model,
    and it is what our earlier Res18 replica (with_res18/smptemp_active.py)
    already did. The difference now is that the architecture itself comes
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
from src.models.utae_paps_models.ltae import LTAE2d  # noqa: E402


def masked_group_norm(gn, x, valid):
    """GroupNorm over (channels, time) that ignores padded time steps.

    Copied verbatim from models/utae.py (directories do not import across each
    other; see the top-level README). The full story is in that docstring; the
    short one: `LTAE.in_norm` is a GroupNorm over (N, d, T), so it pools its
    statistics over the TEMPORAL axis, padded steps included, and the attention
    mask downstream cannot undo it because the leak happens before the
    attention. Measured here as a padding drift of 1.4e-1 / 2.9e-1 in the
    scored probabilities of utae_res18 before this fix.

    x: (N, C, T) · valid: (N, T), 1.0 where the step is real
    """
    N, C, T = x.shape
    G = gn.num_groups
    xg = x.view(N, G, C // G, T)
    v = valid.view(N, 1, 1, T)
    n = v.sum(-1, keepdim=True).clamp(min=1) * (C // G)
    mean = (xg * v).sum((-2, -1), keepdim=True) / n
    var = (((xg - mean) ** 2) * v).sum((-2, -1), keepdim=True) / n
    xg = (xg - mean) / torch.sqrt(var + gn.eps)
    out = xg.view(N, C, T)
    if gn.affine:
        out = out * gn.weight.view(1, C, 1) + gn.bias.view(1, C, 1)
    return out * valid.unsqueeze(1)      # padded steps carry no value


class _LTAE2dMaskedNorm(LTAE2d):
    """Their LTAE2d, with `in_norm` masked when the batch carries padding.

    Their forward already accepts `pad_mask` but only feeds it to the attention
    softmax; the GroupNorm two lines earlier still pools padded steps into its
    statistics. This subclass is their `forward` verbatim except for the
    `in_norm` call, mirroring the fix models/utae.py applies to our own LTAE.
    It adds no parameters, so checkpoints load unchanged, and with
    `pad_mask=None` it is bit-identical to theirs.
    """

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        if pad_mask is not None and pad_mask.any():                    # <-- delta
            out = masked_group_norm(self.in_norm, out.permute(0, 2, 1),
                                    (~pad_mask).to(out.dtype)).permute(0, 2, 1)
        else:
            out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class SMPTempModelPadded(SMPTempModel):
    """SMPTempModel that ignores zero-padded frames in its temporal attention
    AND in the GroupNorm that precedes it (see _LTAE2dMaskedNorm)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Same module, same weights, masked in_norm: swapping the class (not
        # the module) keeps the checkpoint keys (`ltae.*`) identical.
        self.ltae.__class__ = _LTAE2dMaskedNorm

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

        # theirs (SMPTempModel.forward), with pad_mask threaded through
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
