"""Encoder module for Task 1 Universal Autoencoder.

Progressive spatial downsampling network compressing 128x128 RGB images
into a compact latent bottleneck representation.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Dual-convolution block with BatchNorm, LeakyReLU, and optional residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.use_residual else None
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual and residual is not None:
            out = out + residual
        return self.act2(out)


class Encoder(nn.Module):
    """Convolutional encoder with progressive spatial reduction and channel expansion.

    Downsamples 128x128 inputs across N stages (default: 4 stages to 8x8) into
    a compressed latent bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_dim: int = 256,
        use_residual: bool = False,
        use_skips: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.bottleneck_dim = bottleneck_dim
        self.use_skips = use_skips

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Progressive downsampling stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.skip_reducers = nn.ModuleList() if use_skips else None

        for i in range(len(channels)):
            c_in = channels[0] if i == 0 else channels[i - 1]
            c_out = channels[i]

            stage = ConvBlock(c_in, c_out, use_residual=use_residual)
            self.stages.append(stage)

            if use_skips:
                # 1x1 bottleneck on skip connection to prevent uncompressed bypass
                skip_ch = max(4, c_out // 4)
                self.skip_reducers.append(
                    nn.Sequential(
                        nn.Conv2d(c_out, skip_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(skip_ch),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
                )

            # Downsample to next spatial resolution (stride 2 conv)
            down = nn.Sequential(
                nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.downsamplers.append(down)

        # Bottleneck projection
        bottleneck_layers: List[nn.Module] = [
            nn.Conv2d(channels[-1], bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0.0:
            bottleneck_layers.append(nn.Dropout2d(p=dropout))

        self.bottleneck = nn.Sequential(*bottleneck_layers)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass through encoder.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Tuple of (bottleneck_features, list_of_skip_tensors_or_None).
        """
        feats = self.stem(x)
        skips: Optional[List[torch.Tensor]] = [] if self.use_skips else None

        for i, (stage, down) in enumerate(zip(self.stages, self.downsamplers)):
            feats = stage(feats)
            if self.use_skips and skips is not None:
                skip_feat = self.skip_reducers[i](feats)
                skips.append(skip_feat)
            feats = down(feats)

        latent = self.bottleneck(feats)
        return latent, skips
