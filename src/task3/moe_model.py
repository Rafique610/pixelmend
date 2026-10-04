"""src/task3/moe_model.py
----------------------
Soft Mixture-of-Experts (MoE) restoration model for Task 3.

Coordinates:
    - Gating Network G(x_tilde): maps input to routing logits z and temperature-scaled
      weights w = softmax(z / tau).
    - Branch 1: Identity bypass b_1 = x_tilde (0 parameters, 0 operations, preserves clean images).
    - Branch 2: Salt-and-Pepper Specialist Autoencoder b_2 = A_salt(x_tilde).
    - Branch 3: Gaussian Blur Specialist Autoencoder b_3 = A_blur(x_tilde).
    - Branch 4: Rectangular Occlusion Specialist Autoencoder b_4 = A_occlusion(x_tilde).

Produces the convex composite restoration:
    x_hat = sum_{k=1}^4 w_{:, k, None, None, None} * b_k
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.task2.specialist import SpecialistAutoencoder
from src.task3.gate import GatingNetwork


class SoftMoE(nn.Module):
    """Soft Mixture-of-Experts (MoE) Image Restoration Model.

    Combines an identity bypass branch with 3 specialist autoencoders
    via a differentiable temperature-scaled gating network.
    """

    def __init__(
        self,
        gate: Optional[GatingNetwork] = None,
        specialist_salt: Optional[SpecialistAutoencoder] = None,
        specialist_blur: Optional[SpecialistAutoencoder] = None,
        specialist_occlusion: Optional[SpecialistAutoencoder] = None,
        in_channels: int = 3,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.gate: GatingNetwork = gate or GatingNetwork(
            in_channels=in_channels, channels=channels, num_classes=4, dropout=dropout
        )

        def _make_spec(c_type: str, spec: Optional[SpecialistAutoencoder]) -> SpecialistAutoencoder:
            return spec or SpecialistAutoencoder(
                corruption_type=c_type, in_channels=in_channels, out_channels=in_channels,
                channels=channels, bottleneck_dim=bottleneck_dim, use_residual=True,
                use_skips=False, dropout=0.0, upsample_mode="transpose",
            )

        self.specialist_salt = _make_spec("salt_and_pepper", specialist_salt)
        self.specialist_blur = _make_spec("gaussian_blur", specialist_blur)
        self.specialist_occlusion = _make_spec("occlusion", specialist_occlusion)

    # Aliases for plan compatibility
    @property
    def salt_expert(self) -> SpecialistAutoencoder:
        return self.specialist_salt

    @property
    def blur_expert(self) -> SpecialistAutoencoder:
        return self.specialist_blur

    @property
    def occlusion_expert(self) -> SpecialistAutoencoder:
        return self.specialist_occlusion

    def forward(
        self, x: torch.Tensor, tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with temperature-scaled soft routing and differentiable blending.

        Args:
            x: Corrupted input image tensor (B, 3, H, W) or (3, H, W) in [0, 1].
            tau: Routing temperature controlling sharpness (clamped >= 0.05).

        Returns:
            Tuple of:
                - restored_image: Convex composite restoration (B, 3, H, W).
                - routing_weights: Softmax routing probabilities (B, 4) in [0, 1].
                - logits: Unnormalized routing logits (B, 4).
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # 1. Routing logits and weights
        routing_weights, logits = self.gate(x, tau=tau)

        # 2. Parallel branch evaluation
        b1 = x  # Identity branch (clean preservation, 0 params, 0 FLOPs)
        b2 = self.specialist_salt(x)
        b3 = self.specialist_blur(x)
        b4 = self.specialist_occlusion(x)

        # 3. Differentiable convex blending
        w = routing_weights.view(routing_weights.shape[0], 4, 1, 1, 1)
        restored = (
            w[:, 0] * b1
            + w[:, 1] * b2
            + w[:, 2] * b3
            + w[:, 3] * b4
        )

        return restored, routing_weights, logits

    def set_experts_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze all parameters of the 3 specialist autoencoders.

        Used to stabilize training:
            - Phase 1 Warm-up: frozen=True (train gate only)
            - Phase 2 Joint Fine-tuning: frozen=False (train gate + specialists end-to-end)

        Args:
            frozen: If True, sets requires_grad = False for specialists.
                    If False, restores requires_grad = True.
        """
        for specialist in (
            self.specialist_salt,
            self.specialist_blur,
            self.specialist_occlusion,
        ):
            for param in specialist.parameters():
                param.requires_grad = not frozen

    @property
    def experts_frozen(self) -> bool:
        """Check whether specialist autoencoders are currently frozen."""
        return not any(
            p.requires_grad
            for specialist in (
                self.specialist_salt,
                self.specialist_blur,
                self.specialist_occlusion,
            )
            for p in specialist.parameters()
        )

    def load_pretrained_components(
        self,
        classifier_path: Union[str, Path] = "checkpoints/task2/classifier_best.pt",
        salt_path: Union[str, Path] = "checkpoints/task2/specialist_salt_best.pt",
        blur_path: Union[str, Path] = "checkpoints/task2/specialist_blur_best.pt",
        occlusion_path: Union[str, Path] = "checkpoints/task2/specialist_occlusion_best.pt",
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Load pretrained Task 2 checkpoints into gate and all 3 specialists.

        Supports both .pt and .pth files, handling optional 'model_state_dict' wrappers.

        Args:
            classifier_path: Path to Task 2 corruption classifier checkpoint.
            salt_path: Path to salt-and-pepper specialist checkpoint.
            blur_path: Path to Gaussian blur specialist checkpoint.
            occlusion_path: Path to rectangular occlusion specialist checkpoint.
            map_location: Target device for loaded weights.
        """
        # Load Gate
        self.gate.load_from_classifier_checkpoint(
            classifier_path, map_location=map_location
        )

        # Load Specialists
        for model, path_val, name in [
            (self.specialist_salt, salt_path, "salt specialist"),
            (self.specialist_blur, blur_path, "blur specialist"),
            (self.specialist_occlusion, occlusion_path, "occlusion specialist"),
        ]:
            p = Path(path_val)
            if not p.is_file():
                raise FileNotFoundError(f"{name} checkpoint not found at: {p}")

            ckpt = torch.load(p, map_location=map_location, weights_only=False)
            state_dict = (
                ckpt["model_state_dict"]
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                else ckpt
            )

            # Strip module. prefix if present
            clean_sd = {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(clean_sd, strict=True)

    def count_parameters(self, only_trainable: bool = False) -> int:
        """Return total parameter count across all components."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int = 0,
        val_loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint containing state dict, optimizer state, epoch, and metadata."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        opt_sd = None
        if optimizer_state_dict is not None:
            opt_sd = optimizer_state_dict
        elif optimizer is not None:
            opt_sd = optimizer.state_dict()
        elif extra and "optimizer_state_dict" in extra:
            opt_sd = extra["optimizer_state_dict"]

        payload = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": opt_sd,
            "epoch": epoch,
            "val_loss": val_loss,
            "metrics": metrics or {},
            "config": config if config is not None else (extra.get("config") if extra else {}),
            "extra": extra or {},
        }
        torch.save(payload, out_path)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load model state dict and return full checkpoint payload."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        sd = (
            payload["model_state_dict"]
            if isinstance(payload, dict) and "model_state_dict" in payload
            else payload
        )
        clean_sd = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        self.load_state_dict(clean_sd, strict=strict)
        return payload if isinstance(payload, dict) else {"model_state_dict": clean_sd}


def build_soft_moe(
    classifier_path: Optional[Union[str, Path]] = None,
    salt_path: Optional[Union[str, Path]] = None,
    blur_path: Optional[Union[str, Path]] = None,
    occlusion_path: Optional[Union[str, Path]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> SoftMoE:
    """Factory helper to construct and optionally load pretrained SoftMoE."""
    model = SoftMoE(**kwargs)
    if (
        classifier_path is not None
        and salt_path is not None
        and blur_path is not None
        and occlusion_path is not None
    ):
        model.load_pretrained_components(
            classifier_path=classifier_path,
            salt_path=salt_path,
            blur_path=blur_path,
            occlusion_path=occlusion_path,
            map_location=map_location,
        )
    return model


class ExportWrapper(nn.Module):
    """Encapsulates SoftMoE fixing temperature tau for atomic ONNX graph export."""

    def __init__(self, moe_model: SoftMoE, tau: float = 1.0) -> None:
        super().__init__()
        self.moe_model = moe_model
        self.tau = float(tau)

    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        restored, routing_weights, _ = self.moe_model(input_image, tau=self.tau)
        return restored, routing_weights
