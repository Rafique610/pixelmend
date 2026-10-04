"""Dataset utilities for Task 2 Specialist Autoencoders.

Dedicated single-corruption dynamic training datasets and pre-cached validation
loaders for Salt-and-Pepper, Gaussian Blur, and Rectangular Occlusion.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from src.shared.config import Settings, get_settings
from src.shared.corruptions import (
    CorruptionType,
    apply_corruption,
    sample_random_corruption,
)
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_val_manifest
from src.task2.specialist import VALID_SPECIALISTS

LABEL_MAP = {
    "salt_and_pepper": 1,
    "gaussian_blur": 2,
    "occlusion": 3,
}

CORR_TYPE_MAP = {
    "salt_and_pepper": CorruptionType.SALT_AND_PEPPER,
    "gaussian_blur": CorruptionType.GAUSSIAN_BLUR,
    "occlusion": CorruptionType.OCCLUSION,
}


class SpecialistTrainDataset(Dataset):
    """Dynamic single-corruption dataset for targeted specialist training."""

    def __init__(
        self,
        clean_tensors: List[torch.Tensor],
        corruption_type: str,
        image_size: int = 128,
    ) -> None:
        if corruption_type not in VALID_SPECIALISTS:
            raise ValueError(
                f"Invalid corruption_type '{corruption_type}'. Must be one of {VALID_SPECIALISTS}."
            )
        self.clean_tensors = clean_tensors
        self.corruption_type = corruption_type
        self.c_enum = CORR_TYPE_MAP[corruption_type]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.clean_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_tensors[idx]
        _, _, params = sample_random_corruption(
            corruption_type=self.c_enum, image_size=self.image_size
        )
        corrupted = apply_corruption(clean, self.c_enum, params)
        return corrupted, clean


class SpecialistValDataset(Dataset):
    """Pre-cached single-corruption validation dataset for rapid deterministic evaluation."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


def build_specialist_dataloaders(
    corruption_type: str,
    batch_size: int = 32,
    clean_train_tensors: Optional[List[torch.Tensor]] = None,
    settings: Optional[Settings] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Construct dedicated training and validation DataLoaders for a specific specialist."""
    if corruption_type not in VALID_SPECIALISTS:
        raise ValueError(
            f"Invalid corruption_type '{corruption_type}'. Must be one of {VALID_SPECIALISTS}."
        )

    cfg = settings or get_settings()

    # 1. Base clean training tensors
    if clean_train_tensors is None:
        base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=cfg)
        clean_train_tensors = [base_train[i] for i in range(len(base_train))]

    train_dataset = SpecialistTrainDataset(
        clean_tensors=clean_train_tensors,
        corruption_type=corruption_type,
        image_size=128,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # 2. Validation dataset filtered from val_manifest
    base_val = PetDataset(split="val", image_size=128, return_labels=False, settings=cfg)
    val_manifest = load_val_manifest(settings=cfg)
    target_label = LABEL_MAP[corruption_type]

    cached_val_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for item in val_manifest:
        if item["corruption_label"] == target_label:
            clean = base_val[item["val_id"]]
            corrupted = apply_corruption(clean, item["corruption_type"], item["params"])
            cached_val_pairs.append((corrupted, clean))

    val_dataset = SpecialistValDataset(cached_val_pairs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader
