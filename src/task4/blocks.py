"""
src/task4/blocks.py
-------------------
Architectural building blocks for conditional U-Net generators:
1. DownBlock (strided conv downsampling)
2. UpBlock (transposed conv upsampling with skip concat)
3. ResBlock (residual convolutional block with identity shortcut)
4. AttentionGate (attention-weighted skip connection filter)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """Downsampling convolutional block (stride 2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block with transposed conv and skip concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.deconv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        if skip is not None:
            out = torch.cat([out, skip], dim=1)
        return out


class ResBlock(nn.Module):
    """Residual convolutional block with identity shortcut."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections (Oktay et al., 2018).
    Weights encoder skip features using gating signal from the decoder.
    """

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(f_int, affine=False),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(f_int, affine=False),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions if needed
        w_g = self.w_g(g)
        w_x = self.w_x(x)
        if w_g.shape[2:] != w_x.shape[2:]:
            w_g = F.interpolate(w_g, size=w_x.shape[2:], mode="bilinear", align_corners=False)
        f = self.relu(w_g + w_x)
        alpha = self.psi(f)
        return x * alpha
