"""CBAM channel-then-spatial attention for one terminal feature map.

Reference: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class CBAMBlock(nn.Module):
    def __init__(self, channels=2048, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        if (type(channels) is not int or type(reduction_ratio) is not int
                or channels < 1 or reduction_ratio < 1 or channels // reduction_ratio < 1):
            raise ValueError("CBAM channels and reduction must yield a positive hidden size")
        if spatial_kernel_size != 7:
            raise ValueError("This confirmed CBAM configuration uses a 7x7 spatial kernel")
        self.channels = channels
        # Average- and max-pooled descriptors share one MLP; the outputs are summed before the sigmoid.
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=True),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels, bias=True),
        )
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel_size,
                                      padding=spatial_kernel_size // 2, bias=False)
        # BN settings follow the authors' standalone CBAM module; no zero initialisation from their per-block ResNet.
        self.spatial_bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("CBAM expects NCHW features with the configured channel count")
        average = F.avg_pool2d(features, kernel_size=features.shape[2:]).flatten(1)
        maximum = features.flatten(2).max(dim=2).values
        channel_gate = self.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        refined = features * channel_gate[:, :, None, None]
        spatial_descriptor = torch.cat((refined.max(dim=1, keepdim=True).values,
                                        refined.mean(dim=1, keepdim=True)), dim=1)
        spatial_gate = self.sigmoid(self.spatial_bn(self.spatial_conv(spatial_descriptor)))
        return refined * spatial_gate
