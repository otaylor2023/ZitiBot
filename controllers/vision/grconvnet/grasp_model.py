"""Base class + residual block for GR-ConvNet.

Vendored from https://github.com/skumra/robotic-grasping (inference/models/grasp_model.py).
License: BSD-3-Clause (upstream).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
    """Abstract grasp-prediction network with shared loss/predict helpers."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_in):
        raise NotImplementedError

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            "pos": pos_pred,
            "cos": cos_pred,
            "sin": sin_pred,
            "width": width_pred,
        }


class ResidualBlock(nn.Module):
    """Residual block used in GR-ConvNet's bottleneck stack."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
