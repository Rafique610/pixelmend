"""
tests/test_fs2k_dataset.py
--------------------------
Unit tests for FS2K paired dataset, stratified splits, dual normalization,
and DataLoader batching.
"""

import json
from pathlib import Path

import torch

from src.shared.config import get_settings
from src.shared.datasets.fs2k import FS2KDataset, get_fs2k_dataloader


def test_fs2k_manifest_and_splits():
    settings = get_settings()
    manifest_path = Path(settings.manifests_dir) / "fs2k_split.json"
    assert manifest_path.is_file(), f"Manifest missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["train_count"] == 899
    assert meta["val_count"] == 159
    assert meta["test_count"] == 1046
    assert meta["total_pairs"] == 2104

    assert len(data["train"]) == 899
    assert len(data["val"]) == 159
    assert len(data["test"]) == 1046


def test_fs2k_dataset_shapes_and_normalization():
    ds = FS2KDataset(split="val", return_meta=True)
    assert len(ds) == 159

    photo, sketch, style_id, meta = ds[0]
    assert photo.shape == (3, 128, 128)
    assert sketch.shape == (3, 128, 128)
    assert photo.dtype == torch.float32
    assert sketch.dtype == torch.float32

    assert 0.0 <= photo.min() and photo.max() <= 1.0
    assert 0.0 <= sketch.min() and sketch.max() <= 1.0
    assert style_id in [0, 1, 2]
    assert meta["split"] == "val"


def test_fs2k_dual_normalization():
    ds_tanh = FS2KDataset(split="val", normalize_range="neg_one_to_one")
    photo, sketch, _ = ds_tanh[0]

    assert -1.0 <= photo.min() and photo.max() <= 1.0
    assert -1.0 <= sketch.min() and sketch.max() <= 1.0


def test_fs2k_dataloader_batch():
    loader = get_fs2k_dataloader(
        split="val",
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    b_photos, b_sketches, b_styles = next(iter(loader))

    assert b_photos.shape == (4, 3, 128, 128)
    assert b_sketches.shape == (4, 3, 128, 128)
    assert b_styles.shape == (4,)
    assert b_styles.dtype == torch.int64


def test_fs2k_synchronized_augmentation():
    # Test augmentation flag initialization
    ds_aug = FS2KDataset(split="train", augment=True)
    assert ds_aug.augment is True

    # Val split should not augment even if requested
    ds_val_aug = FS2KDataset(split="val", augment=True)
    assert ds_val_aug.augment is False
