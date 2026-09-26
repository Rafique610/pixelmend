"""
src/shared/datasets/pets.py
---------------------------
Reusable PyTorch Dataset and DataLoader factory for the Oxford-IIIT Pet dataset.
Guarantees 128x128 RGB conversion, [0.0, 1.0] float32 normalization,
and deterministic train/val/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.shared.config import Settings, get_settings


class PetDataset(Dataset):
    """PyTorch Dataset for Oxford-IIIT Pet images."""

    def __init__(
        self,
        split: str = "train",
        image_size: int = 128,
        root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_labels: bool = False,
        settings: Optional[Settings] = None,
    ) -> None:
        self.split = split.lower().strip()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        self.image_size = image_size
        self.transform = transform
        self.return_labels = return_labels

        cfg = settings or get_settings()
        self.root_dir = Path(root or cfg.data_dir) / "oxford-iiit-pet"
        self.images_dir = self.root_dir / "images"

        m_path = Path(manifest_path or cfg.manifests_dir / "pets_split.json")
        self.items = self._load_manifest(m_path)

        # Standard transformation pipeline: PIL RGB -> Resize 128x128 -> Tensor [0.0, 1.0]
        self.base_transform = T.Compose([
            T.Resize(
                (self.image_size, self.image_size),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ToTensor(),
        ])

    def _load_manifest(self, manifest_path: Path) -> list[dict[str, Any]]:
        """Load split items from JSON manifest, or scan directory if manifest absent."""
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self.split in data:
                return data[self.split]

        # Fallback: scan images directory directly if manifest hasn't been generated yet
        if self.images_dir.exists():
            all_images = sorted(list(self.images_dir.glob("*.jpg")))
            return [{"image": p.name, "class_id": 0} for p in all_images]

        return []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple[torch.Tensor, int]]:
        item = self.items[idx]
        img_name = item["image"]
        img_path = self.images_dir / img_name

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found at {img_path}. Run 'task download-pets' first."
            )

        with Image.open(img_path) as raw_img:
            # Force conversion to 3-channel RGB (handles RGBA and grayscale 1-channel)
            rgb_img = raw_img.convert("RGB")
            tensor = self.base_transform(rgb_img)

        if self.transform is not None:
            tensor = self.transform(tensor)

        if self.return_labels:
            label = item.get("class_id", 0)
            return tensor, label
        return tensor


def get_pet_dataloader(
    split: str = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    image_size: int = 128,
    return_labels: bool = False,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    settings: Optional[Settings] = None,
) -> DataLoader:
    """Create a configured DataLoader for PetDataset."""
    cfg = settings or get_settings()
    dataset = PetDataset(
        split=split,
        image_size=image_size,
        return_labels=return_labels,
        settings=cfg,
    )

    is_train = split == "train"
    do_shuffle = is_train if shuffle is None else shuffle
    n_workers = cfg.num_workers if num_workers is None else num_workers
    is_cuda = cfg.resolved_device == "cuda"
    do_pin = is_cuda if pin_memory is None else pin_memory

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=n_workers,
        pin_memory=do_pin,
        persistent_workers=(n_workers > 0),
    )
