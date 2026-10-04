"""Synthetic image corruption service for on-the-fly degradation in API workspaces."""

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.shared.corruptions import (
    apply_clean,
    apply_gaussian_blur,
    apply_occlusion,
    apply_salt_and_pepper,
    compute_occlusion_coverage,
    sample_occlusion_boxes,
)


def apply_synthetic_corruption(
    image_nchw: np.ndarray,
    corruption_type: str,
    severity: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply synthetic corruption matching benchmark parameters to an NCHW float32 image in [0, 1].

    Parameters:
    -----------
    image_nchw: np.ndarray of shape [1, 3, 128, 128] float32 in [0.0, 1.0]
    corruption_type: str ('clean', 'salt_and_pepper', 'gaussian_blur', 'rectangular_occlusion')
    severity: float severity level (1, 2, 3 or continuous value)
    seed: optional int seed for reproducibility

    Returns:
    --------
    corrupted_nchw: np.ndarray of shape [1, 3, 128, 128] float32
    metadata: Dict detailing applied degradation parameters
    """
    generator = None
    rng = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        rng = random.Random(seed)

    tensor = torch.from_numpy(image_nchw).float()
    norm_type = corruption_type.lower().strip().replace("-", "_")

    if norm_type in ("clean", "identity"):
        corrupted = apply_clean(tensor)
        meta = {"type": "clean", "severity": 0, "description": "Clean identity (unmodified)"}

    elif norm_type in ("salt_and_pepper", "salt_pepper", "noise"):
        # Map severity 1, 2, 3 to standard benchmark levels [0.03, 0.08, 0.15]
        if severity == 1:
            p = 0.03
        elif severity == 2:
            p = 0.08
        elif severity == 3:
            p = 0.15
        else:
            p = max(0.01, min(0.5, float(severity)))

        corrupted = apply_salt_and_pepper(tensor, p=p, generator=generator)
        meta = {
            "type": "salt_and_pepper",
            "p": round(p, 4),
            "severity_level": severity,
            "description": f"Salt-and-pepper noise (p={p:.2f})",
        }

    elif norm_type in ("gaussian_blur", "blur"):
        # Map severity 1, 2, 3 to kernel / sigma [(3, 0.7), (5, 1.5), (7, 2.5)]
        if severity == 1:
            kernel_size, sigma = 3, 0.7
        elif severity == 2:
            kernel_size, sigma = 5, 1.5
        elif severity == 3:
            kernel_size, sigma = 7, 2.5
        else:
            sigma = max(0.2, min(5.0, float(severity)))
            kernel_size = 5 if sigma <= 2.0 else 7

        corrupted = apply_gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)
        meta = {
            "type": "gaussian_blur",
            "kernel_size": kernel_size,
            "sigma": round(sigma, 2),
            "severity_level": severity,
            "description": f"Gaussian blur (kernel={kernel_size}, sigma={sigma:.2f})",
        }

    elif norm_type in ("rectangular_occlusion", "occlusion"):
        # Map severity 1, 2, 3 to boxes / coverage [1: 10%, 2: 20%, 3: 35%]
        if severity == 1:
            num_boxes, target_ratio = 1, 0.10
        elif severity == 2:
            num_boxes, target_ratio = 2, 0.20
        elif severity == 3:
            num_boxes, target_ratio = 3, 0.35
        else:
            target_ratio = max(
                0.05, min(0.5, float(severity) / 100.0 if severity > 1.0 else float(severity))
            )
            num_boxes = 2

        _, _, h, w = tensor.shape
        boxes = sample_occlusion_boxes(
            height=h, width=w, num_boxes=num_boxes, target_ratio=target_ratio, rng=rng
        )
        corrupted = apply_occlusion(tensor, boxes=boxes, fill_value=0.0)
        coverage = compute_occlusion_coverage(boxes, height=h, width=w)
        meta = {
            "type": "rectangular_occlusion",
            "num_boxes": num_boxes,
            "target_ratio": round(target_ratio, 2),
            "actual_coverage": round(coverage, 4),
            "boxes": boxes,
            "severity_level": severity,
            "description": f"Occlusion ({num_boxes} masks, ~{round(coverage * 100, 1)}% area)",
        }

    else:
        raise ValueError(
            f"Unsupported corruption type '{corruption_type}'. Supported: 'clean', "
            "'salt_and_pepper', 'gaussian_blur', 'rectangular_occlusion'."
        )

    return corrupted.cpu().numpy().astype(np.float32), meta
