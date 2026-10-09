"""Squeeze-and-Excitation channel attention.

Mechanism: https://arxiv.org/pdf/1709.01507 (Section 3).
The terminal insertion used here is a project adaptation, not SE-ResNet-50.
"""

from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        if (type(channels) is not int or type(reduction_ratio) is not int
                or channels < 1 or reduction_ratio < 1 or channels // reduction_ratio < 1):
            raise ValueError("SE channels and reduction must yield a positive hidden size")
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Both Linear layers use bias and PyTorch default initialisation; no residual or identity-gate initialisation.
        self.fc1 = nn.Linear(channels, channels // reduction_ratio, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction_ratio, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("SE expects NCHW features with the configured channel count")
        descriptor = self.pool(features).flatten(1)
        gates = self.sigmoid(self.fc2(self.relu(self.fc1(descriptor))))
        return features * gates[:, :, None, None]
