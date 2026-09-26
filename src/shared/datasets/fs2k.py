"""
src/shared/datasets/fs2k.py
---------------------------
PyTorch Dataset and DataLoader factory for the FS2K (Facial Sketch Synthesis 2K)
paired face-to-sketch dataset used in Task 4 (Conditional GAN).

Features:
- Enforces strict 1:1 correspondence between photographic inputs and artist sketches.
- Resizes both paired images to 128x128 resolution.
- Normalizes tensors to [0.0, 1.0] or [-1.0, 1.0].
- Synchronized paired data augmentation (simultaneous random horizontal flip).
- Returns (photo_tensor, sketch_tensor, style_id) where style_id in {0, 1, 2}.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.shared.config import Settings, get_settings


class FS2KDataset(Dataset):
    """
    PyTorch Dataset for paired FS2K Face-to-Sketch generation.

    Yields:
    -------
    (photo_tensor, sketch_tensor, style_id)
    where:
      photo_tensor: (3, 128, 128) float32 in [0.0, 1.0] or [-1.0, 1.0]
      sketch_tensor: (3, 128, 128) float32 in [0.0, 1.0] or [-1.0, 1.0]
      style_id: int in {0, 1, 2}
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 128,
        root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        normalize_range: str = "zero_one",
        augment: bool = False,
        return_meta: bool = False,
        settings: Optional[Settings] = None,
    ) -> None:
        self.split = split.lower().strip()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        self.image_size = image_size
        self.normalize_range = normalize_range.lower()
        if self.normalize_range not in {"zero_one", "neg_one_to_one"}:
            raise ValueError(
                f"Invalid normalize_range '{normalize_range}'. "
                "Expected 'zero_one' or 'neg_one_to_one'."
            )

        self.augment = augment and (self.split == "train")
        self.return_meta = return_meta

        cfg = settings or get_settings()
        self.root_dir = Path(root or cfg.data_dir) / "fs2k"
        m_path = Path(manifest_path or cfg.manifests_dir / "fs2k_split.json")

        if not m_path.is_file():
            raise FileNotFoundError(
                f"FS2K manifest not found at {m_path}. Run scripts/download_fs2k.py first."
            )

        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.items: list[dict[str, Any]] = data.get(self.split, [])
        if not self.items:
            raise ValueError(f"No samples found for split '{self.split}' in {m_path}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, idx: int
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor, int],
        tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]],
    ]:
        item = self.items[idx]

        p_file = self.root_dir / item["photo_path"]
        s_file = self.root_dir / item["sketch_path"]

        if not p_file.is_file():
            raise FileNotFoundError(f"Missing photo image at {p_file}")
        if not s_file.is_file():
            raise FileNotFoundError(f"Missing sketch image at {s_file}")

        # Open and ensure 3-channel RGB representation
        with Image.open(p_file) as raw_p:
            photo_img = raw_p.convert("RGB")
        with Image.open(s_file) as raw_s:
            sketch_img = raw_s.convert("RGB")

        # Resize to target resolution (128x128)
        resample = Image.Resampling.BILINEAR
        photo_img = photo_img.resize((self.image_size, self.image_size), resample)
        sketch_img = sketch_img.resize((self.image_size, self.image_size), resample)

        # Synchronized horizontal flip augmentation
        if self.augment and (random.random() < 0.5):
            photo_img = photo_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            sketch_img = sketch_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Convert to PyTorch float32 tensors [0.0, 1.0]
        photo_tensor = TF.to_tensor(photo_img)
        sketch_tensor = TF.to_tensor(sketch_img)

        # Scale to [-1.0, 1.0] if requested (common for tanh generators in GANs)
        if self.normalize_range == "neg_one_to_one":
            photo_tensor = photo_tensor * 2.0 - 1.0
            sketch_tensor = sketch_tensor * 2.0 - 1.0

        style_id = int(item.get("style_id", 0))

        if self.return_meta:
            meta = {
                "photo_path": item["photo_path"],
                "sketch_path": item["sketch_path"],
                "style_id": style_id,
                "split": self.split,
                "gender": item.get("gender"),
                "smile": item.get("smile"),
                "hair": item.get("hair"),
            }
            return photo_tensor, sketch_tensor, style_id, meta

        return photo_tensor, sketch_tensor, style_id


def get_fs2k_dataloader(
    split: str = "train",
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
    normalize_range: str = "zero_one",
    augment: Optional[bool] = None,
    return_meta: bool = False,
    settings: Optional[Settings] = None,
) -> DataLoader:
    """
    DataLoader factory for FS2K paired dataset.

    Returns batches of:
    - photo_batch:  (B, 3, 128, 128) float32
    - sketch_batch: (B, 3, 128, 128) float32
    - style_ids:    (B,) int64 in {0, 1, 2}
    """
    cfg = settings or get_settings()

    if shuffle is None:
        shuffle = split == "train"

    if augment is None:
        augment = split == "train"

    workers = num_workers if num_workers is not None else cfg.num_workers
    pin_memory = cfg.device == "cuda"

    dataset = FS2KDataset(
        split=split,
        normalize_range=normalize_range,
        augment=augment,
        return_meta=return_meta,
        settings=cfg,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
    )
