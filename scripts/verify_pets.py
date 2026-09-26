"""
scripts/verify_pets.py
----------------------
Verifies the Oxford-IIIT Pet dataset splits, shapes, normalization,
and exports an 8-sample visualization grid to results/oxford_pets_sample_grid.png.
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.shared.config import get_settings
from src.shared.datasets.pets import PetDataset, get_pet_dataloader


def main():
    settings = get_settings()
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Verifying Oxford-IIIT Pet dataset splits...")
    ds_train = PetDataset(split="train", return_labels=True)
    ds_val = PetDataset(split="val", return_labels=True)
    ds_test = PetDataset(split="test", return_labels=True)

    print(f"  - Train count: {len(ds_train)} (Expected: 2944)")
    print(f"  - Val count:   {len(ds_val)} (Expected: 736)")
    print(f"  - Test count:  {len(ds_test)} (Expected: 3669)")

    assert len(ds_train) == 2944, f"Expected 2944 train items, got {len(ds_train)}"
    assert len(ds_val) == 736, f"Expected 736 val items, got {len(ds_val)}"
    assert len(ds_test) == 3669, f"Expected 3669 test items, got {len(ds_test)}"

    # Check a sample
    sample_tensor, sample_label = ds_train[0]
    print(f"Sample tensor shape: {sample_tensor.shape}, dtype: {sample_tensor.dtype}")
    print(
        f"Sample min: {sample_tensor.min():.4f}, max: {sample_tensor.max():.4f}, "
        f"label: {sample_label}"
    )
    assert sample_tensor.shape == (3, 128, 128)
    assert 0.0 <= sample_tensor.min() and sample_tensor.max() <= 1.0

    # Test DataLoader
    loader = get_pet_dataloader(split="val", batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"DataLoader val batch shape: {batch.shape}")
    assert batch.shape == (8, 3, 128, 128)

    # Export 8-sample inspection grid
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    random.seed(42)
    sample_indices = random.sample(range(len(ds_train)), 8)

    for ax, idx in zip(axes.flatten(), sample_indices):
        img_tensor, label = ds_train[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(f"Class: {label} (#{idx})")
        ax.axis("off")

    fig.tight_layout()
    out_path = results_dir / "oxford_pets_sample_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved 8-sample visualization to {out_path}")
    print("All Oxford-IIIT Pet verification checks PASSED successfully!")


if __name__ == "__main__":
    main()
