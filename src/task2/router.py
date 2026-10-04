"""
src/task2/router.py
-------------------
Hard-Routing Inference Engine for Task 2.

Integrates:
1. Corruption Classifier C(x) predicting class in {0: clean, 1: salt_pepper, 2: blur, 3: occlusion}
2. Identity Bypass: routes clean inputs directly with bit-exact preservation and 0 FLOPS
3. 3 Specialist Autoencoders (S_salt, S_blur, S_occlusion) trained on single-corruption distributions

Supports both predicted routing and oracle-guided routing, batched dispatching,
and latency profiling.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from src.task2.classifier import CorruptionClassifier, build_classifier
from src.task2.specialist import SpecialistAutoencoder, build_specialist

LABEL_TO_CLASS: Dict[int, str] = {
    0: "clean",
    1: "salt_pepper",
    2: "blur",
    3: "occlusion",
}

CLASS_TO_LABEL: Dict[str, int] = {v: k for k, v in LABEL_TO_CLASS.items()}

LABEL_TO_EXPERT: Dict[int, str] = {
    0: "identity_bypass",
    1: "specialist_salt",
    2: "specialist_blur",
    3: "specialist_occlusion",
}


class HardRouter(nn.Module):
    """Unified Hard-Routing Restoration Model.

    Dispatches inputs to either identity bypass or the specialized autoencoder
    based on the classifier's argmax decision or an oracle label.
    """

    def __init__(
        self,
        classifier: nn.Module,
        specialist_salt: nn.Module,
        specialist_blur: nn.Module,
        specialist_occlusion: nn.Module,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.specialist_salt = specialist_salt
        self.specialist_blur = specialist_blur
        self.specialist_occlusion = specialist_occlusion

        self.specialists: Dict[int, nn.Module] = {
            1: self.specialist_salt,
            2: self.specialist_blur,
            3: self.specialist_occlusion,
        }

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        oracle_labels: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Run hard-routing inference pipeline on single tensor or mini-batch.

        Args:
            x: Input tensor of shape (3, H, W) or (B, 3, H, W) in [0.0, 1.0].
            oracle_labels: Optional ground-truth corruption label(s). If provided,
                bypasses classifier decision and routes directly to the designated branch.

        Returns:
            Dictionary containing restored tensor, predicted classes, probabilities,
            expert choices, routing decisions, and latency breakdowns.
        """
        if x.ndim == 3:
            is_single = True
            x_batch = x.unsqueeze(0)
        elif x.ndim == 4:
            is_single = False
            x_batch = x
        else:
            raise ValueError(f"Expected 3D (C, H, W) or 4D (B, C, H, W) tensor, got {x.ndim}D tensor with shape {x.shape}")

        b_size, _, _, _ = x_batch.shape
        device = x_batch.device
        t0 = time.perf_counter()

        # 1. Routing classification or oracle bypass
        if oracle_labels is not None:
            if isinstance(oracle_labels, int):
                routing_decisions = torch.tensor([oracle_labels] * b_size, dtype=torch.long, device=device)
            elif isinstance(oracle_labels, (list, tuple)):
                routing_decisions = torch.tensor(oracle_labels, dtype=torch.long, device=device)
            else:
                routing_decisions = oracle_labels.to(device=device, dtype=torch.long)
            classifier_latency_ms = 0.0
            probs_matrix = torch.zeros((b_size, 4), device=device)
            for idx, r in enumerate(routing_decisions):
                probs_matrix[idx, r] = 1.0
        else:
            tc0 = time.perf_counter()
            logits = self.classifier(x_batch)
            probs_matrix = torch.softmax(logits, dim=1)
            routing_decisions = torch.argmax(probs_matrix, dim=1)
            tc1 = time.perf_counter()
            classifier_latency_ms = (tc1 - tc0) * 1000.0

        # 2. Specialist restoration & Identity bypass
        tr0 = time.perf_counter()
        reconstructed = torch.empty_like(x_batch)

        for class_idx in torch.unique(routing_decisions).tolist():
            mask = routing_decisions == class_idx
            sub_inputs = x_batch[mask]

            if class_idx == 0:
                # Identity bypass (0 FLOPS, bit-exact reproduction)
                reconstructed[mask] = sub_inputs
            elif class_idx in self.specialists:
                reconstructed[mask] = self.specialists[class_idx](sub_inputs)
            else:
                # Fallback to identity if unrecognized
                reconstructed[mask] = sub_inputs

        tr1 = time.perf_counter()
        restoration_latency_ms = (tr1 - tr0) * 1000.0
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Clamp output to valid normalized image range
        reconstructed = torch.clamp(reconstructed, 0.0, 1.0)

        # 3. Format response dictionary
        probs_np = probs_matrix.detach().cpu().numpy()
        decisions_list = routing_decisions.detach().cpu().tolist()

        pred_classes = [LABEL_TO_CLASS.get(d, "unknown") for d in decisions_list]
        experts_list = [LABEL_TO_EXPERT.get(d, "unknown") for d in decisions_list]
        probs_dicts = [
            {
                LABEL_TO_CLASS[k]: float(probs_np[i, k])
                for k in range(4)
            }
            for i in range(b_size)
        ]

        if is_single:
            return {
                "reconstructed": reconstructed.squeeze(0),
                "predicted_class": pred_classes[0],
                "probabilities": probs_dicts[0],
                "selected_expert": experts_list[0],
                "routing_decision": decisions_list[0],
                "classifier_latency_ms": round(classifier_latency_ms, 3),
                "restoration_latency_ms": round(restoration_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "oracle_routing": oracle_labels is not None,
            }

        return {
            "reconstructed": reconstructed,
            "predicted_class": pred_classes,
            "probabilities": probs_dicts,
            "selected_expert": experts_list,
            "routing_decision": decisions_list,
            "classifier_latency_ms": round(classifier_latency_ms, 3),
            "restoration_latency_ms": round(restoration_latency_ms, 3),
            "total_latency_ms": round(total_latency_ms, 3),
            "oracle_routing": oracle_labels is not None,
        }


def _load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_type: str,
    device: torch.device,
) -> nn.Module:
    """Load model weights and structure from checkpoint file."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if model_type == "classifier":
        cfg = ckpt.get("config", {})
        backbone = cfg.get("backbone", "custom_conv")
        dropout = cfg.get("dropout", 0.2)
        model = build_classifier(backbone=backbone, dropout=dropout)
    else:
        cfg = ckpt.get("config", {})
        corruption_type = cfg.get("corruption_type", model_type)
        channels = cfg.get("channels", (32, 64, 128, 256))
        bottleneck_dim = cfg.get("bottleneck_dim", 256)
        use_residual = cfg.get("use_residual", True)
        model = SpecialistAutoencoder(
            corruption_type=corruption_type,
            channels=channels,
            bottleneck_dim=bottleneck_dim,
            use_residual=use_residual,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_hard_router(
    classifier_path: Union[str, Path] = "checkpoints/task2/classifier_best.pt",
    salt_path: Union[str, Path] = "checkpoints/task2/specialist_salt_best.pt",
    blur_path: Union[str, Path] = "checkpoints/task2/specialist_blur_best.pt",
    occlusion_path: Union[str, Path] = "checkpoints/task2/specialist_occlusion_best.pt",
    device: Optional[Union[str, torch.device]] = None,
) -> HardRouter:
    """Instantiate and load pre-trained weights for the complete HardRouter.

    Args:
        classifier_path: Path to corruption classifier checkpoint.
        salt_path: Path to salt-and-pepper specialist checkpoint.
        blur_path: Path to Gaussian blur specialist checkpoint.
        occlusion_path: Path to rectangular occlusion specialist checkpoint.
        device: Target torch device ('cpu', 'cuda'). Defaults to autodetect.

    Returns:
        HardRouter module initialized in evaluation mode.
    """
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = _load_model_from_checkpoint(classifier_path, "classifier", dev)
    specialist_salt = _load_model_from_checkpoint(salt_path, "salt_and_pepper", dev)
    specialist_blur = _load_model_from_checkpoint(blur_path, "gaussian_blur", dev)
    specialist_occlusion = _load_model_from_checkpoint(occlusion_path, "occlusion", dev)

    router = HardRouter(
        classifier=classifier,
        specialist_salt=specialist_salt,
        specialist_blur=specialist_blur,
        specialist_occlusion=specialist_occlusion,
    )
    router.to(dev)
    router.eval()
    return router
