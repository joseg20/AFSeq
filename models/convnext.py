"""ConvNeXtUNet -- baseline of the architecture table.

ConvNeXt-style U-Net on the last observed frame. The second static baseline,
with a modern convolutional backbone.

Self-contained: extracted from the research code with the transitive closure of
what it uses, verified to produce bit-identical output to the original class.
Part of the architecture table of the paper.
"""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import timm

__all__ = ["ConvNeXtUNet"]


def make_cbr(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class ConvNeXtUNet(nn.Module):
    def __init__(self, backbone_name='convnext_small', in_chans=17, out_channels=1, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_chans)
        feat_channels = self.backbone.feature_info.channels()

        self.num_stages = len(feat_channels)
        dec_channels = [256, 128, 64, 32][:self.num_stages - 1]

        self.decoder_blocks = nn.ModuleList()
        prev_ch = feat_channels[-1]
        for i, out_ch in enumerate(dec_channels):
            skip_idx = -2 - i
            if abs(skip_idx) > len(feat_channels):
                break  # no más niveles de skip disponibles
            skip_ch = feat_channels[skip_idx]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    make_cbr(prev_ch + skip_ch, out_ch),
                    make_cbr(out_ch, out_ch)
                )
            )
            prev_ch = out_ch

        self.final_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]

        for i, block in enumerate(self.decoder_blocks):
            skip = features[-2 - i]
            up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([up, skip], dim=1))

        x = torch.sigmoid(self.final_conv(x))
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)

        return x


def make_cbr(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

