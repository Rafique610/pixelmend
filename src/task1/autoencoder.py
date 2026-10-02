"""Universal Denoising Autoencoder for Task 1.

Encapsulates Encoder and Decoder modules into a unified PyTorch model
capable of blind restoration across Clean, Salt-and-Pepper, Blur, and Occlusion.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.task1.decoder import Decoder
from src.task1.encoder import Encoder


class UniversalAutoencoder(nn.Module):
    """End-to-end convolutional autoencoder for blind multi-corruption image restoration."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_dim: int = 256,
        use_residual: bool = False,
        use_skips: bool = False,
        dropout: float = 0.0,
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()
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
        """Run full forward pass from corrupted image to restored image."""
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
        """Calculate spatial compression ratio between input and bottleneck.

        Args:
            input_shape: (C, H, W) of input image.

        Returns:
            Ratio of input elements to latent bottleneck elements.
        """
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
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model weights and metadata to file."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": {
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
    ) -> Tuple["UniversalAutoencoder", Dict[str, Any]]:
        """Instantiate model and restore weights from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint
