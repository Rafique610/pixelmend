"""
src/task2/dataset.py
--------------------
Dataset and batch sampling utilities for Task 2 Corruption Classifier.

Implements:
- BalancedDynamicTrainDataset: On-the-fly multi-class corruption augmentation.
- BalancedBatchSampler: Enforces exactly B/4 samples per corruption class in every mini-batch.
- CachedValDataset: In-memory tensor caching for fast evaluation.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.shared.config import Settings, get_settings
from src.shared.corruptions import (
    LABEL_TO_TYPE,
    apply_corruption,
    sample_random_corruption,
)
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_val_manifest


class BalancedDynamicTrainDataset(Dataset):
    """Dynamic corruption dataset guaranteeing equal class counts across clean base images."""

    def __init__(self, clean_tensors: List[torch.Tensor], image_size: int = 128) -> None:
        self.clean_tensors = clean_tensors
        self.image_size = image_size
        self.num_clean = len(clean_tensors)
        self.labels = [i % 4 for i in range(self.num_clean)]

    def __len__(self) -> int:
        return self.num_clean

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clean = self.clean_tensors[idx]
        target_label = self.labels[idx]
        c_type = LABEL_TO_TYPE[target_label]
        _, _, params = sample_random_corruption(corruption_type=c_type, image_size=self.image_size)
        corrupted = apply_corruption(clean, c_type, params)
        return corrupted, target_label


class BalancedBatchSampler(Sampler[List[int]]):
    """Guarantees exactly B/4 samples from each of the 4 classes in every mini-batch."""

    def __init__(self, labels: Sequence[int], batch_size: int = 32, shuffle: bool = True, seed: int = 42) -> None:
        if batch_size % 4 != 0:
            raise ValueError(f"batch_size {batch_size} must be divisible by 4.")
        self.batch_size = batch_size
        self.k = batch_size // 4
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        self.class_indices: Dict[int, List[int]] = {c: [] for c in range(4)}
        for idx, lbl in enumerate(labels):
            self.class_indices[lbl].append(idx)

        min_count = min(len(self.class_indices[c]) for c in range(4))
        self.num_batches = min_count // self.k

    def __iter__(self):
        pools = {}
        for c in range(4):
            indices = list(self.class_indices[c])
            if self.shuffle:
                self.rng.shuffle(indices)
            pools[c] = indices

        for b in range(self.num_batches):
            batch = []
            for c in range(4):
                batch.extend(pools[c][b * self.k : (b + 1) * self.k])
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class CachedValDataset(Dataset):
    """Pre-cached validation dataset for rapid multi-epoch evaluation."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, int]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.pairs[idx]


def build_classifier_dataloaders(
    batch_size: int = 32,
    seed: int = 42,
    settings: Settings | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Construct balanced training and cached validation DataLoaders for Task 2 classifier."""
    cfg = settings or get_settings()

    # Train dataloader
    base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=cfg)
    clean_train_tensors = [base_train[i] for i in range(len(base_train))]
    train_dataset = BalancedDynamicTrainDataset(clean_train_tensors, image_size=128)
    batch_sampler = BalancedBatchSampler(train_dataset.labels, batch_size=batch_size, shuffle=True, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

    # Val dataloader
    base_val = PetDataset(split="val", image_size=128, return_labels=False, settings=cfg)
    val_manifest = load_val_manifest(settings=cfg)
    cached_val_pairs: List[Tuple[torch.Tensor, int]] = []
    for item in val_manifest:
        clean = base_val[item["val_id"]]
        corr = apply_corruption(clean, item["corruption_type"], item["params"])
        cached_val_pairs.append((corr, item["corruption_label"]))
    val_dataset = CachedValDataset(cached_val_pairs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
