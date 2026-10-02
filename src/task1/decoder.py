"""Decoder module for Task 1 Universal Autoencoder.

Progressive spatial upsampling network reconstructing clean 128x128 RGB images
from a compressed latent bottleneck.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from src.task1.encoder import ConvBlock


class Decoder(nn.Module):
    """Convolutional decoder with mirrored spatial expansion and channel reduction.

    Upsamples latent features from bottleneck resolution (e.g. 8x8) back to
    128x128 RGB with Sigmoid output activation constrained to [0.0, 1.0].
    """

    def __init__(
        self,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_dim: int = 256,
        use_residual: bool = False,
        use_skips: bool = False,
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.bottleneck_dim = bottleneck_dim
        self.use_skips = use_skips
        self.upsample_mode = upsample_mode

        # Project bottleneck features back to top encoder feature depth
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(bottleneck_dim, channels[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Mirrored upsampling stages (e.g. 256 -> 128 -> 64 -> 32)
        reversed_channels = list(reversed(channels))
        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(len(reversed_channels)):
            c_in = reversed_channels[i]
            c_out = reversed_channels[i + 1] if i + 1 < len(reversed_channels) else reversed_channels[-1]

            # Upsample block: 2x spatial increase
            if upsample_mode == "bilinear":
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                up = nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            self.upsamplers.append(up)

            # ConvBlock after upsampling (+ skip channels if enabled)
            skip_idx = len(channels) - 1 - i
            enc_stage_ch = channels[skip_idx] if skip_idx >= 0 else 0
            skip_ch = max(4, enc_stage_ch // 4) if (use_skips and skip_idx >= 0) else 0

            stage = ConvBlock(c_out + skip_ch, c_out, use_residual=use_residual)
            self.stages.append(stage)

        # Final projection to RGB image space with Sigmoid activation
        self.output_head = nn.Sequential(
            nn.Conv2d(reversed_channels[-1], out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            latent: Latent tensor of shape (B, bottleneck_dim, H_b, W_b).
            skips: Optional list of skip tensors from encoder stages.

        Returns:
            Reconstructed image tensor of shape (B, out_channels, H, W) in [0, 1].
        """
        feats = self.bottleneck_proj(latent)

        for i, (up, stage) in enumerate(zip(self.upsamplers, self.stages)):
            feats = up(feats)
            skip_idx = len(self.channels) - 1 - i
            if self.use_skips and skips is not None and skip_idx >= 0:
                skip_feat = skips[skip_idx]
                feats = torch.cat([feats, skip_feat], dim=1)
            feats = stage(feats)

        return self.output_head(feats)
