"""
src/shared/datasets/corrupted.py
--------------------------------
PyTorch Dataset and DataLoader wrappers that pair clean Oxford-IIIT Pet images
with their corrupted counterparts for Tasks 1, 2, and 3.

Modes:
- 'train': Dynamic, stochastic on-the-fly corruption sampling across:
    0: Clean / Identity
    1: Salt-and-Pepper (p in [0.02, 0.15])
    2: Gaussian Blur (k in {3, 5, 7}, sigma in [0.5, 2.5])
    3: Rectangular Occlusion (1-3 boxes, 10%-35% coverage)
- 'val': Deterministic validation manifest (736 images, 25% balanced across classes).
- 'test': Deterministic 10-variation test manifest (36,690 benchmark evaluations).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.shared.config import Settings, get_settings
from src.shared.corruptions import (
    CorruptionType,
    apply_corruption,
    sample_random_corruption,
)
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_test_manifest, load_val_manifest


class CorruptedPetDataset(Dataset):
    """
    Dataset yielding (corrupted_image, clean_image, corruption_label).

    Attributes:
    -----------
    split: 'train', 'val', or 'test'
    base_dataset: underlying PetDataset providing clean 128x128 RGB tensors
    mode: 'dynamic' (random corruption on-the-fly) or 'manifest' (deterministic)
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 128,
        root: Optional[Union[str, Path]] = None,
        val_manifest_path: Optional[Union[str, Path]] = None,
        test_manifest_path: Optional[Union[str, Path]] = None,
        corruption_filter: Optional[Union[str, CorruptionType]] = None,
        severity_filter: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_meta: bool = False,
        seed: int = 42,
        settings: Optional[Settings] = None,
    ) -> None:
        self.split = split.lower().strip()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        self.image_size = image_size
        self.return_meta = return_meta
        self.corruption_filter = (
            CorruptionType(corruption_filter).value if corruption_filter is not None else None
        )
        self.severity_filter = severity_filter
        self.rng = random.Random(seed)
        self.settings = settings or get_settings()

        # In train mode, we use dynamic on-the-fly random corruptions.
        # In val mode, we read manifests/val_manifest.json (736 entries).
        # In test mode, we read manifests/test_manifest.json (36,690 entries).
        if self.split == "train":
            self.mode = "dynamic"
            self.base_dataset = PetDataset(
                split="train",
                image_size=image_size,
                root=root,
                transform=transform,
                return_labels=True,
                settings=self.settings,
            )
            self.manifest_entries: list[dict[str, Any]] = []

        elif self.split == "val":
            self.mode = "manifest"
            self.base_dataset = PetDataset(
                split="val",
                image_size=image_size,
                root=root,
                transform=transform,
                return_labels=True,
                settings=self.settings,
            )
            self.manifest_entries = load_val_manifest(val_manifest_path, settings=self.settings)

            # Apply filters if requested
            if self.corruption_filter is not None:
                self.manifest_entries = [
                    e
                    for e in self.manifest_entries
                    if e["corruption_type"] == self.corruption_filter
                ]

        elif self.split == "test":
            self.mode = "manifest"
            # For test mode, underlying base dataset provides all test images indexed by filename
            self.base_dataset = PetDataset(
                split="test",
                image_size=image_size,
                root=root,
                transform=transform,
                return_labels=True,
                settings=self.settings,
            )
            full_test_manifest = load_test_manifest(test_manifest_path, settings=self.settings)
            raw_entries = full_test_manifest.get("entries", [])

            # Optional test filtering by corruption type and/or severity
            if self.corruption_filter is not None:
                raw_entries = [
                    e for e in raw_entries if e["corruption_type"] == self.corruption_filter
                ]
            if self.severity_filter is not None:
                raw_entries = [e for e in raw_entries if e.get("severity") == self.severity_filter]

            self.manifest_entries = raw_entries

            # Build image filename to base_dataset index mapping for fast O(1) lookups
            self._img_to_idx = {item["image"]: i for i, item in enumerate(self.base_dataset.items)}

    def __len__(self) -> int:
        if self.mode == "dynamic":
            return len(self.base_dataset)
        return len(self.manifest_entries)

    def __getitem__(
        self, idx: int
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor, int],
        tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]],
    ]:
        if self.mode == "dynamic":
            # Train: load clean image and breed label
            clean_tensor, breed_label = self.base_dataset[idx]

            # Sample random corruption dynamically
            c_type, label, params = sample_random_corruption(
                rng=None,  # System random for non-repeating epochs
                image_size=self.image_size,
            )
            corrupted_tensor = apply_corruption(clean_tensor, c_type, params)

            if self.return_meta:
                meta = {
                    "corruption_type": c_type.value,
                    "params": params,
                    "breed_label": breed_label,
                    "split": self.split,
                }
                return corrupted_tensor, clean_tensor, label, meta
            return corrupted_tensor, clean_tensor, label

        elif self.split == "val":
            entry = self.manifest_entries[idx]
            clean_tensor, breed_label = self.base_dataset[idx]

            c_type = entry["corruption_type"]
            label = int(entry["corruption_label"])
            params = entry["params"]

            corrupted_tensor = apply_corruption(clean_tensor, c_type, params)

            if self.return_meta:
                meta = {
                    "val_id": entry["val_id"],
                    "image": entry["image"],
                    "corruption_type": c_type,
                    "params": params,
                    "breed_label": breed_label,
                    "split": self.split,
                }
                return corrupted_tensor, clean_tensor, label, meta
            return corrupted_tensor, clean_tensor, label

        else:  # test
            entry = self.manifest_entries[idx]
            img_file = entry["image"]
            base_idx = self._img_to_idx.get(img_file, 0)
            clean_tensor, breed_label = self.base_dataset[base_idx]

            c_type = entry["corruption_type"]
            label = int(entry["corruption_label"])
            params = entry.get("params", {})

            corrupted_tensor = apply_corruption(clean_tensor, c_type, params)

            if self.return_meta:
                meta = {
                    "eval_id": entry.get("eval_id", idx),
                    "image": img_file,
                    "corruption_type": c_type,
                    "severity": entry.get("severity", 0),
                    "params": params,
                    "breed_label": breed_label,
                    "split": self.split,
                }
                return corrupted_tensor, clean_tensor, label, meta
            return corrupted_tensor, clean_tensor, label


def get_corrupted_pet_dataloader(
    split: str = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
    corruption_filter: Optional[Union[str, CorruptionType]] = None,
    severity_filter: Optional[int] = None,
    return_meta: bool = False,
    settings: Optional[Settings] = None,
) -> DataLoader:
    """
    DataLoader factory for CorruptedPetDataset.

    Returns batches of:
    - (corrupted_batch, clean_batch, labels) where:
        corrupted_batch: (B, 3, 128, 128) float32 in [0.0, 1.0]
        clean_batch:     (B, 3, 128, 128) float32 in [0.0, 1.0]
        labels:          (B,) int64 in {0, 1, 2, 3}
    """
    cfg = settings or get_settings()

    if shuffle is None:
        shuffle = split == "train"

    workers = num_workers if num_workers is not None else cfg.num_workers
    pin_memory = cfg.device == "cuda"

    dataset = CorruptedPetDataset(
        split=split,
        corruption_filter=corruption_filter,
        severity_filter=severity_filter,
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
