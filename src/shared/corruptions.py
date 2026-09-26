"""
src/shared/corruptions.py
-------------------------
Deterministic and stochastic image corruption functions for the Oxford-IIIT Pet
restoration tasks (Tasks 1, 2, and 3).

Corruptions supported:
1. Clean / Identity (Corruption label 0)
2. Salt-and-Pepper noise (Corruption label 1)
3. Gaussian blur (Corruption label 2)
4. Rectangular occlusion (Corruption label 3)

All functions support single images (C, H, W) and batched tensors (B, C, H, W)
with pixel values in [0.0, 1.0].
"""

from __future__ import annotations

import math
import random
from enum import Enum, IntEnum
from typing import Any, Optional, Union

import torch
import torchvision.transforms.functional as TF


class CorruptionType(str, Enum):
    """Named identifiers for corruption types."""

    CLEAN = "clean"
    SALT_AND_PEPPER = "salt_and_pepper"
    GAUSSIAN_BLUR = "gaussian_blur"
    OCCLUSION = "occlusion"


class CorruptionLabel(IntEnum):
    """Zero-indexed numerical labels for Task 2 routing classifier."""

    CLEAN = 0
    SALT_AND_PEPPER = 1
    GAUSSIAN_BLUR = 2
    OCCLUSION = 3


TYPE_TO_LABEL: dict[CorruptionType, CorruptionLabel] = {
    CorruptionType.CLEAN: CorruptionLabel.CLEAN,
    CorruptionType.SALT_AND_PEPPER: CorruptionLabel.SALT_AND_PEPPER,
    CorruptionType.GAUSSIAN_BLUR: CorruptionLabel.GAUSSIAN_BLUR,
    CorruptionType.OCCLUSION: CorruptionLabel.OCCLUSION,
}

LABEL_TO_TYPE: dict[int, CorruptionType] = {
    0: CorruptionType.CLEAN,
    1: CorruptionType.SALT_AND_PEPPER,
    2: CorruptionType.GAUSSIAN_BLUR,
    3: CorruptionType.OCCLUSION,
}

# Fixed benchmark severity definitions for the official test set (10 variations total)
TEST_SEVERITIES: dict[str, list[dict[str, Any]]] = {
    CorruptionType.CLEAN.value: [
        {"severity": 0, "name": "clean"},
    ],
    CorruptionType.SALT_AND_PEPPER.value: [
        {"severity": 1, "p": 0.03, "name": "sp_mild"},
        {"severity": 2, "p": 0.08, "name": "sp_medium"},
        {"severity": 3, "p": 0.15, "name": "sp_severe"},
    ],
    CorruptionType.GAUSSIAN_BLUR.value: [
        {"severity": 1, "kernel_size": 3, "sigma": 0.7, "name": "blur_mild"},
        {"severity": 2, "kernel_size": 5, "sigma": 1.5, "name": "blur_medium"},
        {"severity": 3, "kernel_size": 7, "sigma": 2.5, "name": "blur_severe"},
    ],
    CorruptionType.OCCLUSION.value: [
        {"severity": 1, "num_boxes": 1, "target_ratio": 0.10, "name": "occl_mild"},
        {"severity": 2, "num_boxes": 2, "target_ratio": 0.20, "name": "occl_medium"},
        {"severity": 3, "num_boxes": 3, "target_ratio": 0.35, "name": "occl_severe"},
    ],
}


def apply_clean(img: torch.Tensor) -> torch.Tensor:
    """Identity corruption: returns input tensor unmodified."""
    return img.clone()


def apply_salt_and_pepper(
    img: torch.Tensor,
    p: float = 0.05,
    channel_independent: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply salt-and-pepper noise to an image tensor in [0.0, 1.0].

    Parameters:
    -----------
    img: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    p: float in [0.0, 1.0], fraction of corrupted pixels.
       Half of corrupted pixels become pepper (0.0), half become salt (1.0).
    channel_independent: bool
       If False, salt/pepper values are synchronized across all channels.
       If True, noise is sampled independently per channel.
    generator: optional torch.Generator for reproducible sampling.
    """
    if p <= 0.0:
        return img.clone()

    out = img.clone()
    is_batched = out.ndim == 4
    if not is_batched:
        out = out.unsqueeze(0)

    b, c, h, w = out.shape

    if channel_independent:
        rand = torch.rand((b, c, h, w), generator=generator, device=out.device, dtype=out.dtype)
        out[rand < (p / 2.0)] = 0.0
        out[(rand >= (p / 2.0)) & (rand < p)] = 1.0
    else:
        rand = torch.rand((b, 1, h, w), generator=generator, device=out.device, dtype=out.dtype)
        pepper_mask = (rand < (p / 2.0)).expand(b, c, h, w)
        salt_mask = ((rand >= (p / 2.0)) & (rand < p)).expand(b, c, h, w)
        out[pepper_mask] = 0.0
        out[salt_mask] = 1.0

    if not is_batched:
        out = out.squeeze(0)

    return torch.clamp(out, 0.0, 1.0)


def apply_gaussian_blur(
    img: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Apply 2D Gaussian blur convolution to an image tensor in [0.0, 1.0].

    Parameters:
    -----------
    img: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    kernel_size: int, odd kernel size (e.g. 3, 5, 7)
    sigma: float, standard deviation of the Gaussian filter kernel
    """
    if kernel_size <= 1 or sigma <= 0.0:
        return img.clone()

    if kernel_size % 2 == 0:
        kernel_size += 1

    out = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return torch.clamp(out, 0.0, 1.0)


def sample_occlusion_boxes(
    height: int = 128,
    width: int = 128,
    num_boxes: int = 1,
    target_ratio: float = 0.10,
    rng: Optional[random.Random] = None,
) -> list[list[int]]:
    """
    Sample non-degenerate rectangular bounding boxes [y1, x1, y2, x2]
    such that their combined union area approximately matches target_ratio * (H * W).
    """
    r = rng or random.Random()
    total_area = height * width
    target_area = total_area * target_ratio
    area_per_box = target_area / max(1, num_boxes)

    boxes: list[list[int]] = []
    for _ in range(num_boxes):
        aspect_ratio = r.uniform(0.5, 2.0)
        box_h = int(round(math.sqrt(area_per_box / aspect_ratio)))
        box_w = int(round(box_h * aspect_ratio))

        box_h = max(4, min(height, box_h))
        box_w = max(4, min(width, box_w))

        max_y = max(0, height - box_h)
        max_x = max(0, width - box_w)

        y1 = r.randint(0, max_y)
        x1 = r.randint(0, max_x)
        y2 = min(height, y1 + box_h)
        x2 = min(width, x1 + box_w)
        boxes.append([y1, x1, y2, x2])

    return boxes


def compute_occlusion_coverage(
    boxes: list[list[int]],
    height: int = 128,
    width: int = 128,
) -> float:
    """Compute the exact fraction of image area covered by the union of bounding boxes."""
    mask = torch.zeros((height, width), dtype=torch.bool)
    for y1, x1, y2, x2 in boxes:
        mask[y1:y2, x1:x2] = True
    return float(mask.sum().item()) / float(height * width)


def apply_occlusion(
    img: torch.Tensor,
    boxes: list[list[int]],
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    Superimpose rectangular occlusion masks onto an image tensor in [0.0, 1.0].

    Parameters:
    -----------
    img: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    boxes: list of [y1, x1, y2, x2] pixel coordinate rectangles
    fill_value: float fill intensity (default 0.0 = black)
    """
    out = img.clone()
    is_batched = out.ndim == 4
    if not is_batched:
        out = out.unsqueeze(0)

    for y1, x1, y2, x2 in boxes:
        out[:, :, y1:y2, x1:x2] = fill_value

    if not is_batched:
        out = out.squeeze(0)

    return torch.clamp(out, 0.0, 1.0)


def sample_random_corruption(
    rng: Optional[random.Random] = None,
    corruption_type: Optional[Union[str, CorruptionType]] = None,
    image_size: int = 128,
) -> tuple[CorruptionType, int, dict[str, Any]]:
    """
    Sample a random corruption configuration suitable for training data augmentation.

    Returns:
    --------
    tuple of (corruption_type, label_id, parameter_dict)
    """
    r = rng or random.Random()

    if corruption_type is None:
        c_type = r.choice(list(CorruptionType))
    elif isinstance(corruption_type, str):
        c_type = CorruptionType(corruption_type)
    else:
        c_type = corruption_type

    label = TYPE_TO_LABEL[c_type].value

    if c_type == CorruptionType.CLEAN:
        params: dict[str, Any] = {}

    elif c_type == CorruptionType.SALT_AND_PEPPER:
        # Training distribution: p ~ U(0.02, 0.15)
        p = round(r.uniform(0.02, 0.15), 4)
        params = {"p": p, "channel_independent": False}

    elif c_type == CorruptionType.GAUSSIAN_BLUR:
        # Training distribution: k in {3, 5, 7}, sigma ~ U(0.5, 2.5)
        k = r.choice([3, 5, 7])
        sigma = round(r.uniform(0.5, 2.5), 3)
        params = {"kernel_size": k, "sigma": sigma}

    elif c_type == CorruptionType.OCCLUSION:
        # Training distribution: N in {1, 2, 3}, ratio ~ U(0.10, 0.35)
        num_boxes = r.choice([1, 2, 3])
        target_ratio = round(r.uniform(0.10, 0.35), 3)
        boxes = sample_occlusion_boxes(
            height=image_size,
            width=image_size,
            num_boxes=num_boxes,
            target_ratio=target_ratio,
            rng=r,
        )
        fill = 0.0
        params = {"boxes": boxes, "fill_value": fill, "target_ratio": target_ratio}

    else:
        raise ValueError(f"Unsupported corruption type: {c_type}")

    return c_type, label, params


def apply_corruption(
    img: torch.Tensor,
    corruption_type: Union[str, CorruptionType],
    params: Optional[dict[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Unified entrypoint to apply any specified corruption to an image tensor.

    Parameters:
    -----------
    img: torch.Tensor of shape (C, H, W) or (B, C, H, W) in [0.0, 1.0]
    corruption_type: CorruptionType or str
    params: dictionary of parameters for the corruption
    generator: optional torch.Generator for salt-and-pepper noise reproducibility
    """
    if isinstance(corruption_type, str):
        c_type = CorruptionType(corruption_type)
    else:
        c_type = corruption_type

    p_dict = params or {}

    if c_type == CorruptionType.CLEAN:
        return apply_clean(img)

    if c_type == CorruptionType.SALT_AND_PEPPER:
        p = float(p_dict.get("p", 0.05))
        ch_indep = bool(p_dict.get("channel_independent", False))
        return apply_salt_and_pepper(
            img,
            p=p,
            channel_independent=ch_indep,
            generator=generator,
        )

    if c_type == CorruptionType.GAUSSIAN_BLUR:
        k = int(p_dict.get("kernel_size", 5))
        sigma = float(p_dict.get("sigma", 1.5))
        return apply_gaussian_blur(img, kernel_size=k, sigma=sigma)

    if c_type == CorruptionType.OCCLUSION:
        boxes = p_dict.get("boxes", [])
        fill = float(p_dict.get("fill_value", 0.0))
        return apply_occlusion(img, boxes=boxes, fill_value=fill)

    raise ValueError(f"Unknown corruption type: {corruption_type}")
