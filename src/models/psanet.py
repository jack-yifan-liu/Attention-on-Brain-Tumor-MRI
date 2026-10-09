"""Terminal classification adaptation of Point-wise Spatial Attention.

Reference: hszhao/semseg commit 4f274c3f276778228bc14a4565822d46359f0cc8,
model/psanet.py and lib/psa/src/cpu/psamask.cpp. This is not the full PSANet.
"""

import torch
from torch import nn


class RelativePositionMask(nn.Module):
    """Map relative-offset logits to [batch, source, target] matrices."""

    def __init__(self, height, width):
        super().__init__()
        if height < 1 or width < 1:
            raise ValueError("Feature dimensions must be positive")
        self.height, self.width = height, width
        self.mask_channels = (2 * height - 1) * (2 * width - 1)
        positions = torch.arange(height * width)
        rows, cols = positions // width, positions % width
        offsets = ((rows[:, None] - rows[None, :] + height - 1) * (2 * width - 1)
                   + cols[:, None] - cols[None, :] + width - 1)
        # Collect weights are predicted at the target position; the offset is the source relative to the target.
        indices = offsets * (height * width) + positions[None, :]
        self.register_buffer("collect_indices", indices.flatten())

    def forward(self, logits, mode):
        if (logits.ndim != 4
                or tuple(logits.shape[1:]) != (self.mask_channels, self.height, self.width)):
            raise ValueError("Relative attention logits do not match the configured mask/grid")
        if mode not in ("collect", "distribute"):
            raise ValueError("Unknown PSA propagation direction")
        size = self.height * self.width
        matrix = logits.flatten(1).index_select(1, self.collect_indices).reshape(-1, size, size)
        # Distribute swaps sender and receiver, matching the transpose in the authors' operator.
        return matrix if mode == "collect" else matrix.transpose(1, 2)


def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU())


class PSANetAttention(nn.Module):
    def __init__(self, channels=2048, mid_channels=512, feature_size=(7, 7)):
        super().__init__()
        if channels < 1 or mid_channels < 1:
            raise ValueError("PSA channel counts must be positive")
        self.channels = channels
        self.feature_size = tuple(feature_size)
        self.mask = RelativePositionMask(*self.feature_size)
        self.collect_reduce = conv_bn_relu(channels, mid_channels)
        self.distribute_reduce = conv_bn_relu(channels, mid_channels)
        self.collect_attention = nn.Sequential(
            conv_bn_relu(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, self.mask.mask_channels, 1, bias=False))
        self.distribute_attention = nn.Sequential(
            conv_bn_relu(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, self.mask.mask_channels, 1, bias=False))
        self.project = conv_bn_relu(2 * mid_channels, channels)
        # Keep the original [x, context] concatenation; this projection lets the shared pooling and head be reused.
        self.fuse = conv_bn_relu(2 * channels, channels)

    def aggregate(self, features, logits, mode):
        # Map the direction first, then softmax over sources; for distribute, softmax-then-transpose would be wrong.
        weights = self.mask(logits, mode).softmax(dim=1)
        return torch.bmm(features.flatten(2), weights).reshape_as(features)

    def forward(self, features):
        if (features.ndim != 4 or features.shape[1] != self.channels
                or tuple(features.shape[2:]) != self.feature_size):
            raise ValueError("PSA expects the configured NCHW terminal features")
        collect = self.collect_reduce(features)
        distribute = self.distribute_reduce(features)
        collect = self.aggregate(collect, self.collect_attention(collect), "collect")
        distribute = self.aggregate(distribute, self.distribute_attention(distribute), "distribute")
        context = self.project(torch.cat((collect, distribute), dim=1))
        return self.fuse(torch.cat((features, context), dim=1))
