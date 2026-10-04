"""
src/task4/discriminator.py
--------------------------
PatchGAN Discriminators for conditional Face-to-Sketch synthesis:
1. PatchGANDiscriminator: standard 70x70 or 16x16 local receptive field PatchGAN.
2. MultiScaleDiscriminator: dual-scale PatchGAN operating on full and downsampled resolutions.

Conditioned on concatenated (photo, candidate sketch, spatially replicated style embedding).
Emits 2D patch logits for stable BCEWithLogitsLoss adversarial optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task4.generator import init_weights


class PatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator.
    Evaluates whether overlapping N x N patches of (photo, candidate sketch, style)
    are authentic FS2K pairs or synthetic.
    """

    def __init__(
        self,
        in_channels: int = 3,
        sketch_channels: int = 3,
        num_styles: int = 3,
        embed_dim: int = 16,
        base_channels: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        # Input channels = photo (3) + sketch (3) + style_embed (embed_dim)
        total_in_channels = in_channels + sketch_channels + embed_dim

        layers: list[nn.Module] = [
            nn.Conv2d(total_in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = base_channels
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.extend([
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Stride-1 penultimate layer
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.extend([
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # 1-channel patch logit projection (no sigmoid)
        layers.append(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        init_weights(self)

    def forward(
        self, photo: torch.Tensor, sketch: torch.Tensor, style_idx: torch.Tensor
    ) -> torch.Tensor:
        b, _, h, w = photo.shape
        emb = self.style_embed(style_idx)
        emb_spatial = emb.view(b, self.embed_dim, 1, 1).expand(b, self.embed_dim, h, w)

        x_cond = torch.cat([photo, sketch, emb_spatial], dim=1)
        return self.model(x_cond)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (Wang et al., pix2pixHD).
    Evaluates pairs across two scales: D1 (full resolution) and D2 (downsampled 2x).
    """

    def __init__(
        self,
        in_channels: int = 3,
        sketch_channels: int = 3,
        num_styles: int = 3,
        embed_dim: int = 16,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.d1 = PatchGANDiscriminator(
            in_channels=in_channels,
            sketch_channels=sketch_channels,
            num_styles=num_styles,
            embed_dim=embed_dim,
            base_channels=base_channels,
            n_layers=3,
        )
        self.d2 = PatchGANDiscriminator(
            in_channels=in_channels,
            sketch_channels=sketch_channels,
            num_styles=num_styles,
            embed_dim=embed_dim,
            base_channels=base_channels // 2,
            n_layers=2,
        )

    def forward(
        self, photo: torch.Tensor, sketch: torch.Tensor, style_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Scale 1: Full resolution 128x128
        out1 = self.d1(photo, sketch, style_idx)

        # Scale 2: Half resolution 64x64 via average pooling
        photo_down = F.avg_pool2d(photo, kernel_size=2, stride=2)
        sketch_down = F.avg_pool2d(sketch, kernel_size=2, stride=2)
        out2 = self.d2(photo_down, sketch_down, style_idx)

        return out1, out2
