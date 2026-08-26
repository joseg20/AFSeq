"""UNet -- baseline of the architecture table.

U-Net on the last observed frame. Static baseline: no temporal modelling at
all, which is the point -- it measures how much of the task is solved by
today's fire mask plus context.

Self-contained: extracted from the research code with the transitive closure of
what it uses, verified to produce bit-identical output to the original class.
Part of the architecture table of the paper.
"""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ["UNet"]


class UNet(nn.Module):
    def __init__(self, in_channels=34, out_channels=1):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(in_channels, 32), CBR(32, 32))
        self.enc2 = nn.Sequential(CBR(32, 64), CBR(64, 64))
        self.enc3 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.dec2 = CBR(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d1 = self.up1(e3)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return torch.sigmoid(self.final(d2))

