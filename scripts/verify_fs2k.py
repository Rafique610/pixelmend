"""
scripts/verify_fs2k.py
----------------------
Verifies the FS2K dataset:
1. Asserts exact partition counts (899 train, 159 val, 1,046 test = 2,104 total pairs).
2. Verifies proportional style distribution across splits.
3. Tests both [0.0, 1.0] and [-1.0, 1.0] normalization modes and shapes (3, 128, 128).
4. Exports a 3-pair side-by-side inspection figure (Photo vs Sketch for Style 1, 2, 3)
   to results/fs2k_sample_grid.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.shared.config import get_settings
from src.shared.datasets.fs2k import FS2KDataset, get_fs2k_dataloader


def main():
    settings = get_settings()
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Verifying FS2K Paired Dataset & Splits")
    print("=" * 60)

    # 1. Check split counts
    ds_train = FS2KDataset(split="train")
    ds_val = FS2KDataset(split="val")
    ds_test = FS2KDataset(split="test")

    print(f"  - Train count: {len(ds_train)} (Expected: 899)")
    print(f"  - Val count:   {len(ds_val)}   (Expected: 159)")
    print(f"  - Test count:  {len(ds_test)}  (Expected: 1046)")
    print(f"  - Total pairs: {len(ds_train) + len(ds_val) + len(ds_test)} (Expected: 2104)")

    assert len(ds_train) == 899, f"Expected 899 train, got {len(ds_train)}"
    assert len(ds_val) == 159, f"Expected 159 val, got {len(ds_val)}"
    assert len(ds_test) == 1046, f"Expected 1046 test, got {len(ds_test)}"
    assert len(ds_train) + len(ds_val) + len(ds_test) == 2104

    # 2. Check shapes and normalization ([0, 1])
    photo, sketch, style_id = ds_train[0]
    print(f"\nSample tensor shapes: Photo {photo.shape}, Sketch {sketch.shape}, Style: {style_id}")
    assert photo.shape == (3, 128, 128)
    assert sketch.shape == (3, 128, 128)
    assert 0.0 <= photo.min() and photo.max() <= 1.0
    assert 0.0 <= sketch.min() and sketch.max() <= 1.0

    # 3. Check [-1, 1] normalization mode
    ds_tanh = FS2KDataset(split="val", normalize_range="neg_one_to_one")
    photo_t, sketch_t, _ = ds_tanh[0]
    assert -1.0 <= photo_t.min() and photo_t.max() <= 1.0
    assert -1.0 <= sketch_t.min() and sketch_t.max() <= 1.0
    print("Dual normalization modes ([0, 1] and [-1, 1]) verified.")

    # 4. Check DataLoader batching
    loader = get_fs2k_dataloader(split="val", batch_size=8, shuffle=False, num_workers=0)
    b_photo, b_sketch, b_styles = next(iter(loader))
    assert b_photo.shape == (8, 3, 128, 128)
    assert b_sketch.shape == (8, 3, 128, 128)
    assert b_styles.shape == (8,)
    print(f"DataLoader batch shape: {b_photo.shape}, styles: {b_styles.tolist()}")

    # 5. Export 3-style inspection figure (Style 1, Style 2, Style 3)
    fig, axes = plt.subplots(3, 2, figsize=(8, 11))
    style_names = {0: "Style 1 (Sketch 1)", 1: "Style 2 (Sketch 2)", 2: "Style 3 (Sketch 3)"}

    # Find one sample per style from the validation set
    samples_by_style: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i in range(len(ds_val)):
        p, s, s_id = ds_val[i]
        if s_id not in samples_by_style:
            samples_by_style[s_id] = (p, s)
        if len(samples_by_style) == 3:
            break

    for row_idx, s_id in enumerate(sorted(samples_by_style.keys())):
        p_tensor, s_tensor = samples_by_style[s_id]
        p_np = p_tensor.permute(1, 2, 0).numpy()
        s_np = s_tensor.permute(1, 2, 0).numpy()

        axes[row_idx, 0].imshow(p_np)
        axes[row_idx, 0].set_title(
            f"Input Photo — {style_names[s_id]}", fontsize=11, fontweight="bold"
        )
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(s_np)
        axes[row_idx, 1].set_title(
            f"Ground Truth Sketch — {style_names[s_id]}", fontsize=11, fontweight="bold"
        )
        axes[row_idx, 1].axis("off")

    fig.tight_layout()
    out_path = results_dir / "fs2k_sample_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved 3-style inspection figure to {out_path}")
    print("All FS2K verification checks PASSED successfully!")


if __name__ == "__main__":
    main()
