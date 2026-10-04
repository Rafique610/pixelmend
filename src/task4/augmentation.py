"""
src/task4/augmentation.py
-------------------------
Paired spatial transformations and augmentation pipelines for FS2K photo-sketch synthesis.
Guarantees identical geometric distortion across photo and sketch pairs while allowing
subtle photo-only photometric adjustments.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.task4.dataset import FS2KDataset, get_fs2k_dataloaders

__all__ = ["CoupledTransform", "FS2KDataset", "get_fs2k_dataloaders"]


class CoupledTransform:
    """
    Applies identical geometric transformations to paired (photo, sketch) images
    and normalizes them to [-1.0, 1.0] PyTorch tensors.
    """

    def __init__(
        self,
        image_size: int = 128,
        augment: bool = False,
        hflip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        photometric_jitter: bool = True,
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.rotation_degrees = rotation_degrees
        self.photometric_jitter = photometric_jitter

    def __call__(
        self, photo: Image.Image, sketch: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Resize both to target resolution
        photo = photo.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        sketch = sketch.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # 2. Coupled geometric augmentations
        if self.augment:
            # Horizontal flip
            if random.random() < self.hflip_prob:
                photo = TF.hflip(photo)
                sketch = TF.hflip(sketch)

            # Rotation (same angle)
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            photo = TF.rotate(photo, angle)
            sketch = TF.rotate(sketch, angle)

            # 3. Photo-only subtle photometric jitter (does not change sketch strokes)
            if self.photometric_jitter:
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.9, 1.1)
                photo = TF.adjust_brightness(photo, brightness)
                photo = TF.adjust_contrast(photo, contrast)

        # 4. Convert to tensor and normalize to [-1, 1]
        photo_t = TF.to_tensor(photo) * 2.0 - 1.0
        sketch_t = TF.to_tensor(sketch) * 2.0 - 1.0

        return photo_t, sketch_t
