"""
tests/test_pets_dataset.py
--------------------------
Unit tests for PetDataset verifying 128x128 spatial shape, 3-channel RGB conversion,
[0.0, 1.0] normalization, split partitioning, and DataLoader batch generation.
"""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.shared.config import Settings
from src.shared.datasets.pets import PetDataset, get_pet_dataloader


@pytest.fixture
def mock_pet_env(tmp_path: Path):
    """Creates a mock Oxford-IIIT Pet directory with synthetic images of varied channel formats."""
    data_dir = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    pet_dir = data_dir / "oxford-iiit-pet" / "images"
    pet_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create varied test images
    # Image 0: Standard RGB (200x150)
    img_rgb = Image.new("RGB", (200, 150), color=(255, 128, 0))
    img_rgb.save(pet_dir / "dog_0.jpg")

    # Image 1: Grayscale 1-channel (100x100)
    img_gray = Image.new("L", (100, 100), color=120)
    img_gray.save(pet_dir / "cat_1.jpg")

    # Image 2: RGBA 4-channel (80x120) saved as PNG to allow RGBA format
    img_rgba = Image.new("RGBA", (80, 120), color=(10, 20, 30, 200))
    img_rgba.save(pet_dir / "dog_2.png")

    # Image 3: Test split image
    img_test = Image.new("RGB", (128, 128), color=(50, 60, 70))
    img_test.save(pet_dir / "test_0.jpg")

    # 2. Create mock pets_split.json
    manifest_data = {
        "train": [
            {"image": "dog_0.jpg", "class_id": 0},
            {"image": "cat_1.jpg", "class_id": 1},
        ],
        "val": [
            {"image": "dog_2.png", "class_id": 2},
        ],
        "test": [
            {"image": "test_0.jpg", "class_id": 3},
        ],
    }
    manifest_path = manifests_dir / "pets_split.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    settings = Settings(data_dir=data_dir, manifests_dir=manifests_dir, num_workers=0)
    return settings, manifest_path


def test_pet_dataset_shapes_and_normalization(mock_pet_env):
    settings, manifest_path = mock_pet_env

    # 1. Train split
    ds_train = PetDataset(split="train", image_size=128, settings=settings)
    assert len(ds_train) == 2

    # Check first item (was RGB)
    t0 = ds_train[0]
    assert isinstance(t0, torch.Tensor)
    assert t0.shape == (3, 128, 128)
    assert t0.dtype == torch.float32
    assert t0.min() >= 0.0 and t0.max() <= 1.0

    # Check second item (was 1-channel Grayscale -> converted to 3-channel RGB)
    t1 = ds_train[1]
    assert t1.shape == (3, 128, 128)
    assert t1.dtype == torch.float32

    # 2. Val split (was 4-channel RGBA -> converted to 3-channel RGB)
    ds_val = PetDataset(split="val", image_size=128, settings=settings)
    assert len(ds_val) == 1
    t_val = ds_val[0]
    assert t_val.shape == (3, 128, 128)


def test_pet_dataset_labels_and_transform(mock_pet_env):
    settings, _ = mock_pet_env

    # Custom transform: multiply by 0.5
    ds = PetDataset(
        split="train",
        image_size=64,
        return_labels=True,
        transform=lambda x: x * 0.5,
        settings=settings,
    )
    tensor, label = ds[0]
    assert tensor.shape == (3, 64, 64)
    assert label == 0
    assert tensor.max() <= 0.5


def test_pet_dataloader(mock_pet_env):
    settings, _ = mock_pet_env

    loader = get_pet_dataloader(
        split="train",
        batch_size=2,
        num_workers=0,
        settings=settings,
    )
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (2, 3, 128, 128)
    assert batch.dtype == torch.float32


def test_invalid_split(mock_pet_env):
    settings, _ = mock_pet_env
    with pytest.raises(ValueError):
        PetDataset(split="invalid_split_name", settings=settings)
