"""
src/task4/dataset.py
--------------------
FS2K (Facial Sketch Synthesis 2K) paired dataset and dataloaders.
Handles photo-sketch alignment, dual file extensions (.jpg/.png),
coupled spatial augmentations, and stratified train/val/test partitions.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


class FS2KDataset(Dataset):
    """
    FS2K paired facial photo-to-sketch dataset.
    
    Loads paired (photo, sketch) images and integer style index in {0, 1, 2}.
    Ensures identical geometric transformations across photo and sketch pairs.
    """

    def __init__(
        self,
        root_dir: str | Path = "data/fs2k",
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.val_ratio = val_ratio
        self.seed = seed
        self.image_size = image_size
        self.augment = augment

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {self.split}. Expected train, val, or test.")

        self.samples = self._load_samples()

    def _resolve_paths(self, item: dict) -> Tuple[Path, Path]:
        photo_rel = item["image_name"] + ".jpg"
        photo_path = self.root_dir / "photo" / photo_rel

        parts = item["image_name"].split("/")
        folder = parts[0].replace("photo", "sketch")
        base = parts[1].replace("image", "sketch")
        p_jpg = self.root_dir / "sketch" / folder / f"{base}.jpg"
        p_png = self.root_dir / "sketch" / folder / f"{base}.png"

        sketch_path = p_jpg if p_jpg.exists() else p_png
        if not photo_path.exists():
            raise FileNotFoundError(f"Photo not found: {photo_path}")
        if not sketch_path.exists():
            raise FileNotFoundError(f"Sketch not found: {sketch_path}")

        return photo_path, sketch_path

    def _load_samples(self) -> List[dict]:
        if self.split in ("train", "val"):
            anno_path = self.root_dir / "anno_train.json"
            with open(anno_path, "r", encoding="utf-8") as f:
                all_annos = json.load(f)

            # Stratify by style to maintain balanced distribution
            by_style: Dict[int, List[dict]] = {}
            for item in all_annos:
                by_style.setdefault(item["style"], []).append(item)

            rng = random.Random(self.seed)
            train_items: List[dict] = []
            val_items: List[dict] = []

            for style_id in sorted(by_style.keys()):
                items = list(by_style[style_id])
                rng.shuffle(items)
                n_val = int(len(items) * self.val_ratio)
                val_items.extend(items[:n_val])
                train_items.extend(items[n_val:])

            selected = train_items if self.split == "train" else val_items
        else:
            anno_path = self.root_dir / "anno_test.json"
            with open(anno_path, "r", encoding="utf-8") as f:
                selected = json.load(f)

        return selected

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_transforms(
        self, photo: Image.Image, sketch: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize to target resolution
        photo = photo.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        sketch = sketch.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # Coupled geometric augmentations
        if self.augment:
            if random.random() > 0.5:
                photo = TF.hflip(photo)
                sketch = TF.hflip(sketch)

            # Coupled subtle rotation within +-10 degrees
            angle = random.uniform(-10.0, 10.0)
            photo = TF.rotate(photo, angle)
            sketch = TF.rotate(sketch, angle)

        # Convert to Tensor and normalize to [-1, 1]
        photo_t = TF.to_tensor(photo) * 2.0 - 1.0
        sketch_t = TF.to_tensor(sketch) * 2.0 - 1.0

        return photo_t, sketch_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        photo_path, sketch_path = self._resolve_paths(item)

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")

        photo_t, sketch_t = self._apply_transforms(photo, sketch)
        style_idx = int(item["style"])

        return {
            "photo": photo_t,
            "sketch": sketch_t,
            "style": style_idx,
            "image_name": item["image_name"],
        }


def get_fs2k_dataloaders(
    root_dir: str | Path = "data/fs2k",
    batch_size: int = 8,
    image_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders for FS2K."""
    train_ds = FS2KDataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size,
        augment=True,
        seed=seed,
    )
    val_ds = FS2KDataset(
        root_dir=root_dir,
        split="val",
        image_size=image_size,
        augment=False,
        seed=seed,
    )
    test_ds = FS2KDataset(
        root_dir=root_dir,
        split="test",
        image_size=image_size,
        augment=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
