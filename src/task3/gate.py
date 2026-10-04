"""src/task3/gate.py
-----------------
Gating Network architecture for Task 3 Soft Mixture-of-Experts (MoE) Image Restoration.

The GatingNetwork maps an input image x_tilde in R^(B x 3 x 128 x 128) to continuous
routing logits z in R^(B x 4) and temperature-scaled softmax weights w in [0, 1]^(B x 4),
where:
    w = softmax(z / tau, dim=-1)
    sum_{k=1}^4 w_k = 1.0

The 4 routing outputs correspond to:
    Branch 0: Clean / Identity bypass
    Branch 1: Salt-and-Pepper specialist
    Branch 2: Gaussian Blur specialist
    Branch 3: Rectangular Occlusion specialist
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256)
MIN_TEMPERATURE: float = 0.05


class GatingNetwork(nn.Module):
    """4-stage convolutional Gating Network matching Task 2 CustomConvClassifier.

    Computes routing logits and continuous temperature-scaled routing probabilities
    over the identity pass-through branch and the three specialist autoencoders.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = tuple(channels)
        self.num_classes = num_classes
        self.dropout_rate = dropout

        stages: list[nn.Module] = []
        current_c = in_channels
        for out_c in self.channels:
            stages.extend(
                [
                    nn.Conv2d(current_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            current_c = out_c

        self.features = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(current_c, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 256-dimensional feature representations.

        Args:
            x: Input image tensor (B, C, H, W).

        Returns:
            Flattened bottleneck feature tensor (B, channels[-1]).
        """
        feat = self.features(x)
        feat = self.gap(feat)
        return torch.flatten(feat, 1)

    def forward(
        self, x: torch.Tensor, tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing temperature-scaled routing weights and logits.

        Args:
            x: Input image tensor of shape (B, 3, H, W) or (3, H, W).
            tau: Softmax temperature parameter controls routing sharpness.
                Clamped to tau >= 0.05 to prevent division-by-zero or numerical overflow.

        Returns:
            Tuple of (routing_weights, logits):
                - routing_weights: Tensor of shape (B, num_classes) with simplex constraint.
                - logits: Unnormalized routing logits of shape (B, num_classes).
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Safeguard: clamp temperature to avoid division-by-zero / NaN overflow
        tau_clamped = max(float(tau), MIN_TEMPERATURE)

        feat = self.extract_features(x)
        feat = self.dropout(feat)
        logits = self.fc(feat)

        routing_weights = F.softmax(logits / tau_clamped, dim=-1)
        return routing_weights, logits

    def load_from_classifier_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Load pretrained weights from a Task 2 CorruptionClassifier checkpoint.

        Handles checkpoints saved with or without 'model.' or 'module.' prefixes.

        Args:
            checkpoint_path: Path to classifier checkpoint (.pt or .pth).
            map_location: Target device for loaded state dict.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )

        clean_sd: dict[str, torch.Tensor] = {}
        for key, val in state_dict.items():
            clean_k = key
            if clean_k.startswith("module."):
                clean_k = clean_k[len("module.") :]
            if clean_k.startswith("model."):
                clean_k = clean_k[len("model.") :]
            clean_sd[clean_k] = val

        self.load_state_dict(clean_sd, strict=True)

    def count_parameters(self, only_trainable: bool = True) -> int:
        """Return total parameter count for the gating network."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_gating_network(
    in_channels: int = 3,
    channels: Sequence[int] = DEFAULT_CHANNELS,
    num_classes: int = 4,
    dropout: float = 0.2,
    checkpoint_path: Optional[Union[str, Path]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> GatingNetwork:
    """Factory helper to construct and optionally load GatingNetwork weights."""
    model = GatingNetwork(
        in_channels=in_channels,
        channels=channels,
        num_classes=num_classes,
        dropout=dropout,
    )
    if checkpoint_path is not None:
        model.load_from_classifier_checkpoint(checkpoint_path, map_location=map_location)
    return model
