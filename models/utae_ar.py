"""UTAE-AR: autoregressive wildfire-spread forecasting on a U-TAE backbone.

One file, no package: everything needed to build the model, load a released
checkpoint and roll a forecast forward. Merged from the audited `utae_ar`
package (model.py + utae_parts.py + __init__.py); the merge is verified to
produce bit-identical outputs to that package, see dev/ for the check.

    model = load_pretrained("utae_ar_seed23.pth")   # eval mode, horizon=5
    logits = model(x_hist, fut_covars)              # (B, 5, 1, H, W)
    probs = logits.sigmoid()

INPUT CONTRACT. `fut_covars` supplies the 16 covariate channels of each rollout
step. What you put there is a protocol decision, not a detail: covariates
observed after day t are an unreachable ceiling, worth a large margin at the far
end of the horizon over freezing them at day t. See evaluation/ for the four
supported modes
(observed / persist / gfs_fcst / gfs_anl).

From utae_ar/__init__.py
UTAE-AR: autoregressive wildfire spread forecasting on a U-TAE backbone.

Typical use::

    from utae_ar import UTAE_AR, load_pretrained

    model = load_pretrained("utae_ar_seed23.pth")   # eval mode, horizon=5
    logits = model(x_hist, fut_covars)              # (B, 5, 1, H, W)
    probs = logits.sigmoid()

``fut_covars`` supplies the 16 covariate channels of the rollout inputs; read
"Input contract" in the README before choosing what to put in it.

From utae_ar/model.py
UTAE-AR: autoregressive multi-scale ConvLSTM decoder on a shared U-TAE encoder.

The model takes a variable-length history of daily wildfire observations and
rolls out ``horizon`` days of next-day active-fire probability maps.

    ENCODE THE PAST (once)                 ROLL OUT THE FUTURE (once per day)
    ----------------------                 ---------------------------------
    x_hist (B,T,17,H,W)                    prev_frame (B,17,H,W)
           |                                      |
           v                                      v
     U-TAE spatial encoder                  the SAME encoder
     feats[0] 64ch  H                       enc[0] 64ch  H
     feats[1] 64ch  H/2                     enc[1] 64ch  H/2
     feats[2] 128ch H/4                     enc[2] 128ch H/4
           |                                      |
           v                                      |
     +-------------+  bottleneck ctx --> h        |
     |    L-TAE    |               c = 0          v
     +-------------+                        ConvLSTM (bottleneck)  --> h_bott
           |                                      .
           | attention weights                    .
           v                                      v
     +-------------+  skip ctx ------> h    ConvLSTM (one per skip) --> h_skip
     | TemporalAgg |               c = 0          |
     +-------------+                              v
                                     U-TAE decoder (shared up-blocks)
                                                  |
                                                  v
                                        head --> logit (B,1,H,W)
                                                  |
                          sigmoid(logit) becomes channel 16 of the next
                          prev_frame, alongside 16 covariate channels
                          (see "Input contract" below)

Two ideas do the work:

1. **The temporal context initialises the recurrence, it is not re-read.**
   The L-TAE summary of the history becomes the *hidden* state ``h`` of one
   ConvLSTM per U-Net level (bottleneck + every skip).  ``c`` starts at zero.
   From then on the ConvLSTMs carry the fire state forward on their own.

2. **The decoder is autoregressive over the forecast axis.**  The ConvLSTM input
   at step k+1 is the encoding of the frame for the day just predicted: 16
   covariate channels concatenated with the fire mask the model produced for
   that day.  This is what lets the model be steered (drop an observed mask into
   the loop with ``observed_masks`` and it keeps rolling) and what makes wind and
   terrain act on a fire state the model itself produced, rather than on a frozen
   context.

**Input contract for the covariate channels.**  The model does not know, and
does not care, which day the 16 covariate channels of ``fut_covars[:, k]`` were
observed on: it consumes whatever the caller supplies.  The two contracts that
matter are (a) *observed future covariates*, the real fields of days t+1..t+4,
which is what the research pipeline fed and what the published multi-day numbers
were measured with, and (b) *persisted day-t covariates*, nothing beyond the
observed history, which is the deployable setting.  The first forecast day
(t+1) is produced before ``fut_covars`` is touched at all, so it is identical
under both contracts.  See "Input contract" in the README for the measured
difference.

Output is **logits**, shape ``(B, horizon, 1, H, W)``.  Apply ``sigmoid`` to get
probabilities.

From utae_ar/utae_parts.py
Building blocks borrowed from U-TAE, plus the ConvLSTM cell used by the decoder.

U-TAE (`ConvLayer`, `ConvBlock`, `DownConvBlock`, `UpConvBlock`, `LTAE2d`,
`MultiHeadAttention`, `PositionalEncoder`, `TemporalAggregator`,
`TemporallySharedBlock`) is:

    Vivien Sainte Fare Garnot and Loic Landrieu,
    "Panoptic Segmentation of Satellite Image Time Series with Convolutional
    Temporal Attention Networks", ICCV 2021.
    https://github.com/VSainteuf/utae-paps, MIT licence.

`ConvLSTMCell` comes from https://github.com/TUM-LMF/MTLCC-pytorch (authors:
TUM-LMF).

Everything here is a *trimmed* copy: only the options that UTAE-AR actually uses
survive (see README, "What was removed"). The numerics of the surviving paths
are unchanged, so the released checkpoints load with ``strict=True``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UTAE_AR", "UTAEBackbone", "FIRE_CHANNEL", "load_pretrained"]
__version__ = "1.0.0"


# ============================================================ backbone parts
__all__ = [
    "ConvBlock",
    "ConvLSTMCell",
    "DownConvBlock",
    "LTAE2d",
    "TemporalAggregator",
    "UpConvBlock",
    "UTAEBackbone",
]


# Convolutional blocks (spatial encoder / decoder)
class TemporallySharedBlock(nn.Module):
    """Applies a 2-D block to every frame of a (B, T, C, H, W) tensor.

    Sequences in this dataset have variable length and are right-padded with
    ``pad_value`` frames.  Padded frames are *not* pushed through the convs:
    their output is written back as ``pad_value`` so that the L-TAE pad mask and
    the feature maps stay consistent (a conv over an all-zero frame would emit a
    non-zero bias pattern that the mask could no longer identify).
    """

    def __init__(self, pad_value: float | None = None) -> None:
        super().__init__()
        self.pad_value = pad_value

    def smart_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # (B, C, H, W): a single frame, nothing to mask
            return self.forward(x)

        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)

        if self.pad_value is None:
            out = self.forward(flat)
        else:
            is_pad = (flat == self.pad_value).all(-1).all(-1).all(-1)  # (B*T,)
            if is_pad.any():
                real = self.forward(flat[~is_pad])
                out = flat.new_full((b * t,) + real.shape[1:], self.pad_value)
                out[~is_pad] = real
            else:
                out = self.forward(flat)

        return out.view(b, t, *out.shape[1:])


class ConvLayer(nn.Module):
    """``[Conv2d -> Norm -> ReLU] x (len(nkernels) - 1)``.

    ``norm`` is ``"group"`` in the encoder (batch statistics are unreliable with
    B=4 and heavy padding) and ``"batch"`` in the decoder, exactly as in U-TAE.
    """

    def __init__(
        self,
        nkernels: Sequence[int],
        norm: str = "batch",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        if norm == "batch":
            make_norm = nn.BatchNorm2d
        elif norm == "group":
            make_norm = lambda c: nn.GroupNorm(num_channels=c, num_groups=n_groups)  # noqa: E731
        else:
            raise ValueError(f"unsupported norm {norm!r} (use 'batch' or 'group')")

        layers: list[nn.Module] = []
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    nkernels[i], nkernels[i + 1],
                    kernel_size=k, stride=s, padding=p, padding_mode=padding_mode,
                )
            )
            layers.append(make_norm(nkernels[i + 1]))
            # The final ReLU is dropped when this block emits logits.
            if last_relu or i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(TemporallySharedBlock):
    """Plain conv stack, shared across time via :meth:`smart_forward`."""

    def __init__(
        self,
        nkernels: Sequence[int],
        pad_value: float | None = None,
        norm: str = "batch",
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels, norm=norm, last_relu=last_relu, padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    """Strided conv (/2) followed by a residual pair of 3x3 convs."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        pad_value: float | None = None,
        norm: str = "group",
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer([d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode)
        self.conv1 = ConvLayer([d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.down(x))
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    """Transposed conv (x2), concat with the (1x1-projected) skip, residual convs."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_skip: int,
        k: int,
        s: int,
        p: int,
        norm: str = "batch",
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(d_skip, d_skip, kernel_size=1), nn.BatchNorm2d(d_skip), nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer([d_out + d_skip, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.up(x), self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)


# L-TAE: lightweight temporal attention encoder
class PositionalEncoder(nn.Module):
    """Sinusoidal encoding of the day index, tiled once per attention head."""

    def __init__(self, d: int, T: int = 1000, repeat: int = 1) -> None:
        super().__init__()
        self.repeat = repeat
        denom = torch.pow(T, 2 * (torch.arange(d).float() // 2) / d)  # (d,)
        # persistent=False -> stays out of state_dict, so checkpoints stay clean.
        self.register_buffer("denom", denom, persistent=False)

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        # batch_positions: (N, T) integer day indices -> (N, T, d * repeat)
        table = batch_positions[:, :, None] / self.denom
        table[:, :, 0::2] = torch.sin(table[:, :, 0::2])
        table[:, :, 1::2] = torch.cos(table[:, :, 1::2])
        return table.repeat(1, 1, self.repeat)


class MultiHeadAttention(nn.Module):
    """L-TAE attention: a *learned* master query per head, keys from the sequence.

    Values are the head-wise channel groups of the input itself (no value
    projection), which is what makes L-TAE lightweight.
    """

    def __init__(self, n_head: int, d_k: int, d_in: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.temperature = math.sqrt(d_k)

        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.Q, mean=0.0, std=math.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0.0, std=math.sqrt(2.0 / d_k))
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, v: torch.Tensor, pad_mask: torch.Tensor | None = None):
        # v: (N, T, d_in) with N = B*h*w pixel positions
        n, seq_len, _ = v.shape
        nh, dk = self.n_head, self.d_k

        q = torch.stack([self.Q for _ in range(n)], dim=1).view(-1, dk)      # (nh*N, dk)
        k = self.fc1_k(v).view(n, seq_len, nh, dk)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, dk)          # (nh*N, T, dk)
        # values = channel groups of v, one group per head
        vh = torch.stack(v.split(v.shape[-1] // nh, dim=-1)).view(nh * n, seq_len, -1)

        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature  # (nh*N,1,T)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.repeat(nh, 1).unsqueeze(1), -1e3)
        attn = self.attn_dropout(torch.softmax(attn, dim=2))
        out = torch.matmul(attn, vh)                                          # (nh*N,1,d_in/nh)

        out = out.view(nh, n, self.d_in // nh)
        attn = attn.view(nh, n, seq_len)
        return out, attn


class LTAE2d(nn.Module):
    """Temporal attention applied independently at every pixel of the bottleneck.

    Maps (B, T, C, h, w) -> a single (B, C, h, w) summary of the history, plus
    the per-head attention weights (n_head, B, T, h, w) that the skip
    connections reuse to collapse their own time axis.
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        d_model: int = 256,
        mlp: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        T: int = 1000,
    ) -> None:
        super().__init__()
        assert mlp[0] == d_model
        self.n_head = n_head
        self.d_model = d_model

        self.inconv = nn.Conv1d(in_channels, d_model, 1)
        self.positional_encoder = PositionalEncoder(d_model // n_head, T=T, repeat=n_head)
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        layers: list[nn.Module] = []
        for i in range(len(mlp) - 1):
            layers += [nn.Linear(mlp[i], mlp[i + 1]), nn.BatchNorm1d(mlp[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: torch.Tensor,
        pad_mask: torch.Tensor,
    ):
        b, t, d, h, w = x.shape
        n = b * h * w  # every pixel is an independent "sequence"

        def to_pixels(m: torch.Tensor) -> torch.Tensor:
            # (B, T) -> (B*h*w, T), matching the pixel flattening below
            m = m[:, :, None, None].expand(b, t, h, w)
            return m.permute(0, 2, 3, 1).contiguous().view(n, t)

        pix_pad = to_pixels(pad_mask)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(n, t, d)        # (N, T, C)
        # in_norm is a GroupNorm over (channels, time), so it pools its
        # statistics over the padded steps too. See utae.masked_group_norm.
        # The fixed-length path (no padding) keeps the original call, so the
        # WSTS+ anchors are bit-identical.
        if pix_pad.any():
            from utae import masked_group_norm
            out = masked_group_norm(self.in_norm, out.permute(0, 2, 1),
                                    (~pix_pad).to(out.dtype)).permute(0, 2, 1)
        else:
            out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)         # (N, T, d_model)
        out = out + self.positional_encoder(to_pixels(batch_positions))

        out, attn = self.attention_heads(out, pad_mask=pix_pad)           # (nh,N,d/nh), (nh,N,T)

        out = out.permute(1, 0, 2).contiguous().view(n, -1)               # concat heads -> (N, d)
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out)
        out = out.view(b, h, w, -1).permute(0, 3, 1, 2)                   # (B, C, h, w)

        attn = attn.view(self.n_head, b, h, w, t).permute(0, 1, 4, 2, 3)  # (nh, B, T, h, w)
        return out, attn


class TemporalAggregator(nn.Module):
    """Collapse a skip connection's time axis with the L-TAE attention ("att_group").

    The attention map lives at bottleneck resolution, so it is bilinearly
    upsampled to the skip's resolution; head *g* weights channel group *g*.
    """

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, T, C, H, W)   attn_mask: (n_head, B, T, h, w)
        n_head, b, t, h, w = attn_mask.shape
        attn = attn_mask.view(n_head * b, t, h, w)
        if (h, w) != x.shape[-2:]:
            attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = attn.view(n_head, b, t, *x.shape[-2:])
        attn = attn * (~pad_mask).float()[None, :, :, None, None]  # zero-out padded days

        groups = torch.stack(x.chunk(n_head, dim=2))               # (nh, B, T, C/nh, H, W)
        out = (attn[:, :, :, None] * groups).sum(dim=2)            # weighted sum over time
        return torch.cat([g for g in out], dim=1)                  # (B, C, H, W)


# ConvLSTM
class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell: one 3x3 conv produces the four gates."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_cur, c_cur = state
        gates = self.conv(torch.cat([x, h_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# U-TAE backbone (encoder + L-TAE + aggregator + decoder), no output head
class UTAEBackbone(nn.Module):
    """The parts of U-TAE that UTAE-AR reuses.

    Kept as a single submodule (attribute name ``utae`` on :class:`UTAE_AR`) both
    because it mirrors the original architecture and because the released
    checkpoints are keyed as ``utae.*``.

    Note there is **no** ``out_conv``: UTAE-AR owns its own logit head, which
    must not end in a ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_widths: Sequence[int],
        decoder_widths: Sequence[int],
        pad_value: float = 0.0,
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        assert len(encoder_widths) == len(decoder_widths)
        assert encoder_widths[-1] == decoder_widths[-1]
        self.n_stages = len(encoder_widths)
        self.pad_value = pad_value

        self.in_conv = ConvBlock(
            nkernels=[input_dim, encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value, norm="group", padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i], d_out=encoder_widths[i + 1],
                k=str_conv_k, s=str_conv_s, p=str_conv_p,
                pad_value=pad_value, norm="group", padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        # up_blocks[0] consumes the deepest skip, up_blocks[-1] the shallowest.
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i], d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k, s=str_conv_s, p=str_conv_p,
                norm="batch", padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1], d_model=d_model, n_head=n_head, d_k=d_k,
            mlp=[d_model, encoder_widths[-1]],
        )
        self.temporal_aggregator = TemporalAggregator()

    # spatial encoder
    def encode(self, x: torch.Tensor):
        """Encode a history or a single frame with the same weights.

        Args:
            x: ``(B, T, C, H, W)`` sequence, or ``(B, C, H, W)`` single frame.

        Returns:
            ``(feats, pad_mask)``.  ``feats[i]`` is the level-*i* feature map,
            with the same leading dimensions as ``x``.  ``pad_mask`` is the
            ``(B, T)`` boolean mask of padded frames for a sequence, and ``None``
            for a single frame (a frame is never padding).
        """
        pad_mask = None
        if x.dim() == 5:
            pad_mask = (x == self.pad_value).all(-1).all(-1).all(-1)  # (B,T) True where padded
        feats = [self.in_conv.smart_forward(x)]
        for block in self.down_blocks:
            feats.append(block.smart_forward(feats[-1]))
        return feats, pad_mask

    # decoder
    def decode(self, bottleneck: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        """Walk the U-Net back up: (B, C, h, w) + skips -> (B, decoder_widths[0], H, W)."""
        out = bottleneck
        for block, skip in zip(self.up_blocks, skips):
            out = block(out, skip)
        return out


# ================================================== the model
__all__ = ["UTAE_AR"]

FIRE_CHANNEL = 16  # channel index of the active-fire mask inside the 17-channel frame

# U-Net widths per depth. 3 is the released model. 4 is the U-TAE exactly as
# WildfireSpreadTS configures it (UTAELightning.py; Garnot's own defaults) and
# exists for the depth ablation: UTAEBackbone already matches their remaining
# hyper-parameters (n_head=16, d_model=256, d_k=4, strided conv 4/2/1, reflect
# padding, group norm in the encoder), so depth is the only thing this varies.
WIDTHS = {
    3: ((64, 64, 128), (32, 64, 128)),
    4: ((64, 64, 64, 128), (32, 32, 64, 128)),
}


class UTAE_AR(nn.Module):
    """UTAE-AR, the model reported in the paper (registered as ``AR_SS0`` in the
    research code).

    Args:
        input_dim: channels per daily frame (16 covariates + 1 fire mask).
        levels: U-Net depth, a key of ``WIDTHS``.  3 is the released model;
            4 is the original WildfireSpreadTS configuration (see ``WIDTHS``).
            Ignored when ``encoder_widths`` is passed explicitly.
        encoder_widths: channel width per U-Net level, shallow -> deep.
            ``None`` (the default) takes them from ``WIDTHS[levels]``.
        decoder_widths: same, for the decoder; the last entry must match
            ``encoder_widths[-1]`` (the bottleneck is shared).
        out_conv: widths of the logit head after the decoder.
        horizon: number of forecast days produced per call.
        pad_value: value used by the collate function to right-pad short
            histories.  Frames equal to it everywhere are masked out of the
            encoder and of the temporal attention.

    The model is fully convolutional: it was trained on 64x64 patches but runs
    on any spatial size divisible by ``2 ** (len(encoder_widths) - 1)``.
    """

    def __init__(
        self,
        input_dim: int = 34,
        encoder_widths: Sequence[int] | None = None,
        decoder_widths: Sequence[int] | None = None,
        out_conv: Sequence[int] = (32, 1),
        horizon: int = 5,
        pad_value: float = 0.0,
        levels: int = 3,
    ) -> None:
        super().__init__()
        if (encoder_widths is None) != (decoder_widths is None):
            raise ValueError("pass encoder_widths and decoder_widths together, or neither")
        if encoder_widths is None:
            encoder_widths, decoder_widths = WIDTHS[levels]
        encoder_widths = list(encoder_widths)
        decoder_widths = list(decoder_widths)
        self.horizon = horizon
        self.n_stages = len(encoder_widths)

        self.utae = UTAEBackbone(
            input_dim=input_dim,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            pad_value=pad_value,
        )

        # One ConvLSTM per U-Net level.  Its hidden state has the same width as
        # the aggregated context it is initialised with, and its input is the
        # encoding of the previous frame at that same level, hence
        # input_dim == hidden_dim == encoder_widths[level].
        self.lstm_bott = ConvLSTMCell(
            input_dim=encoder_widths[-1], hidden_dim=encoder_widths[-1]
        )
        # up_blocks[i] consumes feats[-(i + 2)], i.e. level n_stages - 2 - i.
        # skip_levels lists those levels in decoder order (deepest skip first).
        self.skip_levels = [self.n_stages - 2 - i for i in range(self.n_stages - 1)]
        self.lstm_sk = nn.ModuleList(
            ConvLSTMCell(input_dim=encoder_widths[lvl], hidden_dim=encoder_widths[lvl])
            for lvl in self.skip_levels
        )

        # Logit head: U-TAE's own out_conv ends in a ReLU, which would clamp
        # every negative logit to zero, so UTAE-AR uses its own head instead.
        self.head = ConvBlock(
            nkernels=[decoder_widths[0]] + list(out_conv), norm="batch", last_relu=False
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        fut_covars: torch.Tensor,
        y_true: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        observed_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Roll out ``horizon`` days of fire-spread logits.

        Args:
            x_hist: ``(B, T, 17, H, W)`` history, right-padded with ``pad_value``.
            fut_covars: ``(B, horizon - 1, 16, H, W)`` covariate frames used as
                channels 0..15 of the rollout inputs.  ``fut_covars[:, k]`` is
                paired with the mask predicted for day t+1+k, so it is the
                covariate field *of the day just predicted*.  Which day's data
                you put there is the caller's choice and defines the input
                contract; see "Input contract" in the README. They are always
                *inputs*: the model forecasts fire, not weather.
            y_true: ``(B, horizon, 1, H, W)`` ground-truth masks, used only for
                scheduled sampling during training.  Pass ``None`` for inference.
                It has no effect at ``teacher_forcing_ratio=0``.
            teacher_forcing_ratio: probability, drawn independently per sample
                and per step, of feeding ``y_true`` instead of the model's own
                prediction at the next step.  ``0.0`` is a free-running rollout,
                which is how the paper's numbers were produced.
            observed_masks: optional per-day override of the feedback mask,
                length ``horizon``, entry *k* either ``None`` or a
                ``(B, 1, H, W)`` mask for forecast day t+1+k.  Where a mask is
                given, it is fed back *verbatim and deterministically* (no
                Bernoulli draw) instead of ``sigmoid(logit)``, which is how you
                steer a rollout with observations or with a hand-drawn mask.
                Entry ``horizon - 1`` is never read (nothing follows the last
                predicted day).  ``None`` everywhere -> ordinary rollout.

        Returns:
            ``(B, horizon, 1, H, W)`` logits.
        """
        u = self.utae
        b, t = x_hist.shape[:2]
        device = x_hist.device

        # 1. encode the history and summarise it in time
        # feats[i]: (B, T, encoder_widths[i], H/2**i, W/2**i)
        feats, pad_mask = u.encode(x_hist)
        day_index = torch.arange(t, device=device).unsqueeze(0).repeat(b, 1)  # (B, T)

        # L-TAE collapses the time axis at the bottleneck and hands back the
        # attention weights, which the aggregator reuses for the skips so that
        # every level is summarised with a consistent notion of "which days
        # mattered".
        bott, attn = u.temporal_encoder(feats[-1], day_index, pad_mask)  # (B,128,H/4,W/4)
        skip_ctx = [
            u.temporal_aggregator(feats[-(i + 2)], pad_mask=pad_mask, attn_mask=attn)
            for i in range(self.n_stages - 1)
        ]

        # 2. seed the ConvLSTMs with that context
        # h = context (what we know), c = 0 (nothing accumulated yet).
        h_b, c_b = bott, torch.zeros_like(bott)
        h_s = list(skip_ctx)
        c_s = [torch.zeros_like(s) for s in skip_ctx]

        # 3. pick the frame the rollout starts from
        # Histories are right-padded to the longest one in the batch, so
        # x_hist[:, -1] is an all-zero frame for most samples: measured on the
        # test set, and the share grows with the evaluation batch size and
        # vanishes at batch size one: it is a property of the batch, not of the
        # sample).  The rollout must start from the last REAL observed day.
        last_real = ((~pad_mask).long().sum(dim=1) - 1).clamp(min=0)          # (B,)
        prev_frame = x_hist[torch.arange(b, device=device), last_real]        # (B,17,H,W)

        # 4. autoregressive rollout
        logits = []
        for k in range(self.horizon):
            enc, _ = u.encode(prev_frame)  # per-level encoding of yesterday

            h_b, c_b = self.lstm_bott(enc[-1], (h_b, c_b))
            for i, lvl in enumerate(self.skip_levels):
                h_s[i], c_s[i] = self.lstm_sk[i](enc[lvl], (h_s[i], c_s[i]))

            logit = self.head(u.decode(h_b, h_s))  # (B, 1, H, W)
            logits.append(logit)

            if k < self.horizon - 1:
                observed = None if observed_masks is None else observed_masks[k]
                mask = self._next_mask(logit, y_true, k, teacher_forcing_ratio, observed)
                prev_frame = torch.cat([fut_covars[:, k], mask], dim=1)
                # (B, 17, H, W): the 16 covariate channels for step k + a fire mask

        return torch.stack(logits, dim=1)  # (B, horizon, 1, H, W)

    def _next_mask(
        self,
        logit: torch.Tensor,
        y_true: torch.Tensor | None,
        k: int,
        teacher_forcing_ratio: float,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fire mask fed back into the encoder for the next rollout step.

        The feedback is the *continuous* probability ``sigmoid(logit)``, not a
        thresholded 0/1 mask: keeping it soft lets low-confidence rim pixels
        decay instead of latching, and it is what the released checkpoints were
        trained with.  It is detached, so gradients travel through the ConvLSTM
        states only and the rollout does not backpropagate through its own
        predictions.

        An ``observed`` mask, when given, replaces the feedback outright and
        deterministically: this is the steering path, and it takes precedence
        over scheduled sampling.

        During training, scheduled sampling replaces the feedback by the ground
        truth with probability ``teacher_forcing_ratio``, drawn independently
        **per sample** (a whole-batch coin flip made the gradient much noisier at
        batch size 4).
        """
        if observed is not None:
            return observed
        feedback = torch.sigmoid(logit.detach())
        if y_true is None or teacher_forcing_ratio <= 0.0:
            return feedback
        b = logit.shape[0]
        use_gt = (torch.rand(b, 1, 1, 1, device=logit.device) < teacher_forcing_ratio).float()
        return use_gt * y_true[:, k] + (1.0 - use_gt) * feedback


# ================================================== loading
__all__ = ["UTAE_AR", "UTAEBackbone", "FIRE_CHANNEL", "load_pretrained"]

__version__ = "1.0.0"


def load_pretrained(
    checkpoint: str | Path,
    horizon: int = 5,
    map_location: str | torch.device = "cpu",
    **kwargs,
) -> UTAE_AR:
    """Build UTAE-AR with the paper's hyper-parameters and load a checkpoint.

    The released checkpoints are plain ``state_dict``s and load with
    ``strict=True``; no key renaming is involved.  The model is returned in
    ``eval()`` mode.
    """
    model = UTAE_AR(horizon=horizon, **kwargs)
    state = torch.load(checkpoint, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
