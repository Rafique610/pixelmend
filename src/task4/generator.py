"""
src/task4/generator.py
----------------------
Production Conditional U-Net Generator for Face-to-Sketch synthesis.
Implements 4-stage encoder-decoder with FiLM style conditioning at the bottleneck,
instance normalization, skip connections, and Gaussian weight initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.task4.blocks import DownBlock, UpBlock
from src.task4.conditioning import FiLMConditioning


def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    """Initialize network weights using Gaussian N(0, 0.02)."""
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class UNetGenerator(nn.Module):
    """
    Conditional U-Net Generator with FiLM style conditioning.
    G(x, s) -> synthesized sketch in [-1, 1].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_styles: int = 3,
        embed_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        c = base_channels
        # Downsampling: 128 -> 64 -> 32 -> 16 -> 8
        self.d1 = DownBlock(in_channels, c, normalize=False)
        self.d2 = DownBlock(c, c * 2)
        self.d3 = DownBlock(c * 2, c * 4)
        self.d4 = DownBlock(c * 4, c * 8, dropout=dropout)

        # FiLM conditioning at bottleneck
        self.cond = FiLMConditioning(c * 8, embed_dim)

        # Upsampling: 8 -> 16 -> 32 -> 64 -> 128
        self.u1 = UpBlock(c * 8, c * 4, dropout=dropout)
        self.u2 = UpBlock(c * 8, c * 2)
        self.u3 = UpBlock(c * 4, c)
        self.u4 = UpBlock(c * 2, c)

        self.final = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        init_weights(self)

    def forward(self, photo: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        photo: [B, 3, 128, 128] in [-1, 1]
        style_idx: [B] in {0, 1, 2}
        Returns: [B, 3, 128, 128] in [-1, 1]
        """
        emb = self.style_embed(style_idx)

        # Encoder
        e1 = self.d1(photo)     # [B, C, 64, 64]
        e2 = self.d2(e1)        # [B, 2C, 32, 32]
        e3 = self.d3(e2)        # [B, 4C, 16, 16]
        e4 = self.d4(e3)        # [B, 8C, 8, 8]

        # Bottleneck modulation
        b = self.cond(e4, emb)

        # Decoder with skip connections
        d1 = self.u1(b, e3)     # [B, 8C, 16, 16]
        d2 = self.u2(d1, e2)    # [B, 4C, 32, 32]
        d3 = self.u3(d2, e1)    # [B, 2C, 64, 64]
        d4 = self.u4(d3, None)  # [B, C, 128, 128]

        return self.final(d4)
