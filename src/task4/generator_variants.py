"""
src/task4/generator_variants.py
-------------------------------
Candidate U-Net generator architectures for conditional Face-to-Sketch synthesis:
1. VanillaUNetGenerator (pix2pix standard baseline)
2. ResNetUNetGenerator (Residual blocks in bottleneck & stages)
3. AttentionUNetGenerator (Attention-gated skip connections)

All variants support pluggable style conditioning (FiLM, AdaIN, SpatialConcat).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.task4.blocks import AttentionGate, DownBlock, ResBlock, UpBlock
from src.task4.conditioning import AdaINConditioning, FiLMConditioning, SpatialConcatConditioning


class VanillaUNetGenerator(nn.Module):
    """
    Standard U-Net Generator with 4 downsampling stages, bottleneck,
    and 4 upsampling stages with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_styles: int = 3,
        embed_dim: int = 16,
        conditioning: str = "film",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning.lower()
        self.embed_dim = embed_dim
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        c = base_channels
        # Downsampling: 128 -> 64 -> 32 -> 16 -> 8
        self.d1 = DownBlock(in_channels, c, normalize=False)
        self.d2 = DownBlock(c, c * 2)
        self.d3 = DownBlock(c * 2, c * 4)
        self.d4 = DownBlock(c * 4, c * 8, dropout=dropout)

        # Bottleneck conditioning
        if self.conditioning_type == "film":
            self.cond = FiLMConditioning(c * 8, embed_dim)
        elif self.conditioning_type == "adain":
            self.cond = AdaINConditioning(c * 8, embed_dim)
        elif self.conditioning_type == "spatial":
            self.cond = SpatialConcatConditioning(embed_dim)
            self.spatial_proj = nn.Conv2d(c * 8 + embed_dim, c * 8, kernel_size=1)
        else:
            raise ValueError(f"Unsupported conditioning: {conditioning}")

        # Upsampling: 8 -> 16 -> 32 -> 64 -> 128
        self.u1 = UpBlock(c * 8, c * 4, dropout=dropout)
        self.u2 = UpBlock(c * 8, c * 2)
        self.u3 = UpBlock(c * 4, c)
        self.u4 = UpBlock(c * 2, c)

        self.final = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        emb = self.style_embed(style_idx)

        e1 = self.d1(x)        # [B, C, 64, 64]
        e2 = self.d2(e1)       # [B, 2C, 32, 32]
        e3 = self.d3(e2)       # [B, 4C, 16, 16]
        e4 = self.d4(e3)       # [B, 8C, 8, 8]

        # Condition at bottleneck
        if self.conditioning_type in ("film", "adain"):
            b = self.cond(e4, emb)
        else:
            b = self.spatial_proj(self.cond(e4, emb))

        d1 = self.u1(b, e3)    # [B, 4C + 4C = 8C, 16, 16]
        d2 = self.u2(d1, e2)   # [B, 2C + 2C = 4C, 32, 32]
        d3 = self.u3(d2, e1)   # [B, C + C = 2C, 64, 64]
        d4 = self.u4(d3, None) # [B, C, 128, 128]

        return self.final(d4)


class ResNetUNetGenerator(nn.Module):
    """
    ResNet-based U-Net Generator.
    Augments bottleneck and intermediate stages with Residual Blocks
    for enhanced gradient flow and stroke texture synthesis.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_styles: int = 3,
        embed_dim: int = 16,
        conditioning: str = "film",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning.lower()
        self.embed_dim = embed_dim
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        c = base_channels
        self.d1 = DownBlock(in_channels, c, normalize=False)
        self.d2 = DownBlock(c, c * 2)
        self.d3 = DownBlock(c * 2, c * 4)
        self.d4 = DownBlock(c * 4, c * 8, dropout=dropout)

        # Residual bottleneck
        self.res1 = ResBlock(c * 8)
        self.res2 = ResBlock(c * 8)

        if self.conditioning_type == "film":
            self.cond = FiLMConditioning(c * 8, embed_dim)
        elif self.conditioning_type == "adain":
            self.cond = AdaINConditioning(c * 8, embed_dim)
        elif self.conditioning_type == "spatial":
            self.cond = SpatialConcatConditioning(embed_dim)
            self.spatial_proj = nn.Conv2d(c * 8 + embed_dim, c * 8, kernel_size=1)
        else:
            raise ValueError(f"Unsupported conditioning: {conditioning}")

        self.u1 = UpBlock(c * 8, c * 4, dropout=dropout)
        self.u2 = UpBlock(c * 8, c * 2)
        self.u3 = UpBlock(c * 4, c)
        self.u4 = UpBlock(c * 2, c)

        self.final = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        emb = self.style_embed(style_idx)

        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)

        # Bottleneck processing with residual flow
        b = self.res1(e4)
        if self.conditioning_type in ("film", "adain"):
            b = self.cond(b, emb)
        else:
            b = self.spatial_proj(self.cond(b, emb))
        b = self.res2(b)

        d1 = self.u1(b, e3)
        d2 = self.u2(d1, e2)
        d3 = self.u3(d2, e1)
        d4 = self.u4(d3, None)

        return self.final(d4)


class AttentionUNetGenerator(nn.Module):
    """
    Attention U-Net Generator (Oktay et al., 2018).
    Gated attention mechanisms filter skip connections to focus on facial landmarks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_styles: int = 3,
        embed_dim: int = 16,
        conditioning: str = "film",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning.lower()
        self.embed_dim = embed_dim
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        c = base_channels
        self.d1 = DownBlock(in_channels, c, normalize=False)
        self.d2 = DownBlock(c, c * 2)
        self.d3 = DownBlock(c * 2, c * 4)
        self.d4 = DownBlock(c * 4, c * 8, dropout=dropout)

        if self.conditioning_type == "film":
            self.cond = FiLMConditioning(c * 8, embed_dim)
        elif self.conditioning_type == "adain":
            self.cond = AdaINConditioning(c * 8, embed_dim)
        elif self.conditioning_type == "spatial":
            self.cond = SpatialConcatConditioning(embed_dim)
            self.spatial_proj = nn.Conv2d(c * 8 + embed_dim, c * 8, kernel_size=1)
        else:
            raise ValueError(f"Unsupported conditioning: {conditioning}")

        # Attention gates for skip connections
        self.ag1 = AttentionGate(f_g=c * 4, f_l=c * 4, f_int=c * 2)
        self.ag2 = AttentionGate(f_g=c * 2, f_l=c * 2, f_int=c)
        self.ag3 = AttentionGate(f_g=c, f_l=c, f_int=c // 2)

        self.u1 = UpBlock(c * 8, c * 4, dropout=dropout)
        self.u2 = UpBlock(c * 8, c * 2)
        self.u3 = UpBlock(c * 4, c)
        self.u4 = UpBlock(c * 2, c)

        self.final = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        emb = self.style_embed(style_idx)

        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)

        if self.conditioning_type in ("film", "adain"):
            b = self.cond(e4, emb)
        else:
            b = self.spatial_proj(self.cond(e4, emb))

        # Upsample with attention-gated skips
        g1 = self.u1.deconv(b)
        g1 = self.u1.norm(g1)
        g1 = self.u1.relu(g1)
        s3 = self.ag1(g=g1, x=e3)
        d1 = torch.cat([g1, s3], dim=1)

        g2 = self.u2.deconv(d1)
        g2 = self.u2.norm(g2)
        g2 = self.u2.relu(g2)
        s2 = self.ag2(g=g2, x=e2)
        d2 = torch.cat([g2, s2], dim=1)

        g3 = self.u3.deconv(d2)
        g3 = self.u3.norm(g3)
        g3 = self.u3.relu(g3)
        s1 = self.ag3(g=g3, x=e1)
        d3 = torch.cat([g3, s1], dim=1)

        d4 = self.u4(d3, None)
        return self.final(d4)
