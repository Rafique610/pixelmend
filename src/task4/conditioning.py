"""
src/task4/conditioning.py
-------------------------
Style conditioning mechanisms for conditional GAN generator:
1. Spatial Concatenation (Isola et al., 2017)
2. FiLM (Feature-wise Linear Modulation - Perez et al., 2018)
3. AdaIN (Adaptive Instance Normalization - Huang & Belongie, 2017)
4. Conditional Instance Normalization (Dumoulin et al., 2017)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConcatConditioning(nn.Module):
    """
    Spatially tiles dense style embedding e_s across (H, W) and concatenates
    with input or intermediate feature maps.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        # emb: [B, D] -> [B, D, 1, 1] -> [B, D, H, W]
        emb_spatial = emb.view(b, self.embed_dim, 1, 1).expand(b, self.embed_dim, h, w)
        return torch.cat([x, emb_spatial], dim=1)


class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Generates per-channel affine transformation parameters (gamma, beta)
    from the style embedding: y = (1 + gamma) * x + beta
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels, in_channels * 2),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.mlp(emb)  # [B, 2*C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.view(-1, self.in_channels, 1, 1)
        beta = beta.view(-1, self.in_channels, 1, 1)
        return (1.0 + gamma) * x + beta


class AdaINConditioning(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN).
    Normalizes feature map x to zero-mean and unit-variance, then scales and shifts
    using style-predicted target standard deviation and mean.
    """

    def __init__(self, in_channels: int, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels, in_channels * 2),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        params = self.mlp(emb)  # [B, 2*C]
        scale, shift = torch.chunk(params, 2, dim=1)
        scale = (scale + 1.0).view(b, c, 1, 1)
        shift = shift.view(b, c, 1, 1)

        # Instance statistics
        var = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = torch.sqrt(var + self.eps)

        normed = (x - mean) / std
        return scale * normed + shift


class ConditionalInstanceNorm(nn.Module):
    """
    Conditional Instance Normalization (CIN).
    Uses discrete style class indices to look up learned affine weights and biases.
    """

    def __init__(self, in_channels: int, num_styles: int = 3, eps: float = 1e-5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.gamma_embed = nn.Embedding(num_styles, in_channels)
        self.beta_embed = nn.Embedding(num_styles, in_channels)

        # Initialize gamma to 1 and beta to 0
        nn.init.ones_(self.gamma_embed.weight)
        nn.init.zeros_(self.beta_embed.weight)

    def forward(self, x: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gamma = self.gamma_embed(style_idx).view(b, c, 1, 1)
        beta = self.beta_embed(style_idx).view(b, c, 1, 1)

        # Instance normalization without learnable affine params
        normed = F.instance_norm(x, eps=self.eps)
        return gamma * normed + beta
