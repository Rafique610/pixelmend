"""
scripts/download_pets.py
------------------------
Downloads the official Oxford-IIIT Pet dataset, verifies integrity,
extracts images and annotations into data/oxford-iiit-pet/,
and generates deterministic train (80%) / val (20%) / test manifests.
"""

from __future__ import annotations

import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

# Add project root to sys.path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.shared.config import get_settings


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path) -> None:
    """Download a file with a visual progress bar."""
    print(f"Downloading {url} -> {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)


def ensure_dataset_downloaded(target_dir: Path) -> None:
    """Download and extract Oxford-IIIT Pet dataset using torchvision or direct fallback."""
    images_dir = target_dir / "oxford-iiit-pet" / "images"
    annotations_dir = target_dir / "oxford-iiit-pet" / "annotations"

    has_images = images_dir.exists() and len(list(images_dir.glob("*.jpg"))) > 7000
    if has_images and annotations_dir.exists():
        print(f"Oxford-IIIT Pet dataset already present at {target_dir / 'oxford-iiit-pet'}")
        return

    print("Attempting download via torchvision.datasets.OxfordIIITPet...")
    try:
        OxfordIIITPet(root=str(target_dir), split="trainval", download=True)
        OxfordIIITPet(root=str(target_dir), split="test", download=True)
        print("Torchvision download completed successfully.")
        return
    except Exception as e:
        print(f"Torchvision download encountered issue: {e}")
        print("Falling back to direct mirror download from robotics.ox.ac.uk...")

    # Direct fallback URLs
    base_url = "https://thor.robots.ox.ac.uk/datasets/pets"
    urls = [
        (f"{base_url}/images.tar.gz", target_dir / "images.tar.gz"),
        (f"{base_url}/annotations.tar.gz", target_dir / "annotations.tar.gz"),
    ]

    for url, tar_path in urls:
        if not tar_path.exists():
            download_url(url, tar_path)

        print(f"Extracting {tar_path.name}...")
        dest = target_dir / "oxford-iiit-pet"
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(dest)

        # Clean archive to save disk space
        if tar_path.exists():
            tar_path.unlink()


def generate_deterministic_split(
    target_dir: Path,
    manifests_dir: Path,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Partition official trainval (3,680) into 80% train / 20% val stratified by breed label."""
    pet_dir = target_dir / "oxford-iiit-pet"
    annotations_dir = pet_dir / "annotations"
    trainval_txt = annotations_dir / "trainval.txt"
    test_txt = annotations_dir / "test.txt"

    if not trainval_txt.exists() or not test_txt.exists():
        raise FileNotFoundError(f"Missing annotation files in {annotations_dir}")

    # Parse trainval.txt: Image ID, 1-based class ID (1-37), Species (1=Cat, 2=Dog), Breed ID
    trainval_items = []
    with open(trainval_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                img_name = parts[0]
                class_id, species, breed_id = int(parts[1]), int(parts[2]), int(parts[3])
                trainval_items.append(
                    {
                        "image": f"{img_name}.jpg",
                        "class_id": class_id - 1,  # 0-indexed class (0..36)
                        "species": species,
                        "breed_id": breed_id,
                    }
                )

    test_items = []
    with open(test_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                img_name = parts[0]
                class_id, species, breed_id = int(parts[1]), int(parts[2]), int(parts[3])
                test_items.append(
                    {
                        "image": f"{img_name}.jpg",
                        "class_id": class_id - 1,
                        "species": species,
                        "breed_id": breed_id,
                    }
                )

    print(f"Loaded {len(trainval_items)} trainval items and {len(test_items)} test items.")

    # Stratified 80/20 trainval split
    labels = [item["class_id"] for item in trainval_items]
    train_items, val_items = train_test_split(
        trainval_items,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )

    manifest_data = {
        "seed": seed,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "pets_split.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Generated deterministic split manifest at {manifest_path}:")
    print(f"  - Train: {len(train_items)} images (80%)")
    print(f"  - Val:   {len(val_items)} images (20%)")
    print(f"  - Test:  {len(test_items)} images (Held-out)")
    return manifest_data


def main():
    settings = get_settings()
    data_dir = settings.data_dir
    manifests_dir = settings.manifests_dir

    print(f"Target dataset directory: {data_dir.resolve()}")
    ensure_dataset_downloaded(data_dir)
    generate_deterministic_split(data_dir, manifests_dir, seed=settings.seed)
    print("Oxford-IIIT Pet dataset preparation complete!")


if __name__ == "__main__":
    main()
