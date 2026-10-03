"""Specialist Autoencoder Architectures for Task 2 Hard-Routing.

Dedicated single-corruption restoration models for Salt-and-Pepper noise,
Gaussian Blur, and Rectangular Occlusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.task1.decoder import Decoder
from src.task1.encoder import Encoder

VALID_SPECIALISTS = ("salt_and_pepper", "gaussian_blur", "occlusion")


class SpecialistAutoencoder(nn.Module):
    """Specialist autoencoder tailored for targeted restoration of a single corruption type."""

    def __init__(
        self,
        corruption_type: Optional[str] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_dim: int = 256,
        use_residual: bool = True,
        use_skips: bool = False,
        dropout: float = 0.0,
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()
        if corruption_type is not None and corruption_type not in VALID_SPECIALISTS:
            raise ValueError(
                f"Invalid corruption_type '{corruption_type}'. Must be one of {VALID_SPECIALISTS}."
            )

        self.corruption_type = corruption_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.bottleneck_dim = bottleneck_dim
        self.use_residual = use_residual
        self.use_skips = use_skips
        self.dropout = dropout
        self.upsample_mode = upsample_mode

        self.encoder = Encoder(
            in_channels=in_channels,
            channels=channels,
            bottleneck_dim=bottleneck_dim,
            use_residual=use_residual,
            use_skips=use_skips,
            dropout=dropout,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            channels=channels,
            bottleneck_dim=bottleneck_dim,
            use_residual=use_residual,
            use_skips=use_skips,
            upsample_mode=upsample_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run full forward restoration pass from corrupted input to reconstructed output."""
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck latent representation."""
        latent, _ = self.encoder(x)
        return latent

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compression_ratio(
        self, input_shape: Tuple[int, int, int] = (3, 128, 128)
    ) -> float:
        """Calculate spatial compression ratio between input and bottleneck."""
        c, h, w = input_shape
        input_elements = c * h * w
        downsample_factor = 2 ** len(self.channels)
        h_bottleneck = h // downsample_factor
        w_bottleneck = w // downsample_factor
        bottleneck_elements = self.bottleneck_dim * h_bottleneck * w_bottleneck
        return float(input_elements / max(1, bottleneck_elements))

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int = 0,
        val_loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model weights, architecture configuration, and metrics to checkpoint."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "metrics": metrics or {},
            "config": {
                "corruption_type": self.corruption_type,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "channels": self.channels,
                "bottleneck_dim": self.bottleneck_dim,
                "use_residual": self.use_residual,
                "use_skips": self.use_skips,
                "dropout": self.dropout,
                "upsample_mode": self.upsample_mode,
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, checkpoint_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple["SpecialistAutoencoder", Dict[str, Any]]:
        """Instantiate model and load weights from checkpoint file."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint


def build_specialist(
    corruption_type: str,
    variant: str = "homogeneous",
    dropout: float = 0.0,
    upsample_mode: str = "transpose",
) -> SpecialistAutoencoder:
    """Factory helper to build a specialist autoencoder for a specified corruption type.

    Variants:
    - 'homogeneous': Standard 4-stage ResBlock architecture across all specialists.
    - 'lightweight': 3-stage compact architecture (32, 64, 128) with bottleneck 128.
    - 'tailored': Corruption-specific structure:
        - salt_and_pepper: 3-stage with residual shortcuts (440K params).
        - gaussian_blur: 4-stage with residual shortcuts (1.54M params).
        - occlusion: 4-stage deep bottleneck with residual shortcuts (1.54M params).
    """
    if corruption_type not in VALID_SPECIALISTS:
        raise ValueError(
            f"Invalid corruption_type '{corruption_type}'. Must be one of {VALID_SPECIALISTS}."
        )

    if variant == "homogeneous":
        return SpecialistAutoencoder(
            corruption_type=corruption_type,
            channels=(32, 64, 128, 256),
            bottleneck_dim=256,
            use_residual=True,
            use_skips=False,
            dropout=dropout,
            upsample_mode=upsample_mode,
        )

    if variant == "lightweight":
        return SpecialistAutoencoder(
            corruption_type=corruption_type,
            channels=(32, 64, 128),
            bottleneck_dim=128,
            use_residual=False,
            use_skips=False,
            dropout=dropout,
            upsample_mode=upsample_mode,
        )

    if variant == "tailored":
        if corruption_type == "salt_and_pepper":
            # High-frequency local noise: 3 stages with residual bypass preserves spatial edges
            return SpecialistAutoencoder(
                corruption_type=corruption_type,
                channels=(32, 64, 128),
                bottleneck_dim=128,
                use_residual=True,
                use_skips=False,
                dropout=dropout,
                upsample_mode=upsample_mode,
            )
        if corruption_type == "gaussian_blur":
            # Multi-scale deblurring: 4 stages with residual connections
            return SpecialistAutoencoder(
                corruption_type=corruption_type,
                channels=(32, 64, 128, 256),
                bottleneck_dim=256,
                use_residual=True,
                use_skips=False,
                dropout=dropout,
                upsample_mode=upsample_mode,
            )
        if corruption_type == "occlusion":
            # Deep context inpainting: 4 stages with 256 bottleneck to hallucinate missing context
            return SpecialistAutoencoder(
                corruption_type=corruption_type,
                channels=(32, 64, 128, 256),
                bottleneck_dim=256,
                use_residual=True,
                use_skips=False,
                dropout=dropout,
                upsample_mode=upsample_mode,
            )

    raise ValueError(f"Unknown variant '{variant}'. Expected 'homogeneous', 'lightweight', or 'tailored'.")
