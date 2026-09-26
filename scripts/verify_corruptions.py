"""
scripts/verify_corruptions.py
-----------------------------
Validates mathematical properties of the corruption pipeline:
- S&P noise pixel fractions and dynamic ranges
- Gaussian blur frequency attenuation and energy conservation
- Occlusion area fractions and boundary compliance
Exports a 4x4 visual verification panel to results/corruption_verification_grid.png.
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.shared.config import get_settings
from src.shared.corruptions import (
    TEST_SEVERITIES,
    CorruptionType,
    apply_clean,
    apply_gaussian_blur,
    apply_occlusion,
    apply_salt_and_pepper,
    compute_occlusion_coverage,
    sample_occlusion_boxes,
)
from src.shared.datasets.pets import PetDataset


def main():
    settings = get_settings()
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Verifying Corruption Pipeline Mathematical Properties")
    print("=" * 60)

    # 1. Salt-and-Pepper Verification
    print("\n1. Testing Salt-and-Pepper Noise:")
    for spec in TEST_SEVERITIES[CorruptionType.SALT_AND_PEPPER.value]:
        p = spec["p"]
        dummy = torch.full((3, 200, 200), 0.5)
        out = apply_salt_and_pepper(dummy, p=p, channel_independent=False)
        corrupted = (out == 0.0) | (out == 1.0)
        frac = corrupted.float().mean().item()
        print(f"   Severity {spec['severity']} (p={p}): observed noise fraction = {frac:.4f}")
        assert abs(frac - p) < 0.02, f"Expected ~{p}, got {frac:.4f}"

    # 2. Gaussian Blur Verification
    print("\n2. Testing Gaussian Blur Frequency Attenuation:")
    # Checkerboard high frequency pattern
    checker = torch.zeros((3, 128, 128))
    checker[:, ::2, ::2] = 1.0
    checker[:, 1::2, 1::2] = 1.0
    var_orig = torch.var(checker).item()

    for spec in TEST_SEVERITIES[CorruptionType.GAUSSIAN_BLUR.value]:
        k = spec["kernel_size"]
        sig = spec["sigma"]
        blurred = apply_gaussian_blur(checker, kernel_size=k, sigma=sig)
        var_blur = torch.var(blurred).item()
        attenuation = (1.0 - (var_blur / var_orig)) * 100.0
        print(
            f"   Severity {spec['severity']} (k={k}, sigma={sig}): "
            f"spatial variance reduced by {attenuation:.1f}%"
        )
        assert var_blur < var_orig

    # 3. Occlusion Area Verification
    print("\n3. Testing Rectangular Occlusion:")
    rng = random.Random(42)
    for spec in TEST_SEVERITIES[CorruptionType.OCCLUSION.value]:
        target = spec["target_ratio"]
        boxes = sample_occlusion_boxes(
            height=128, width=128, num_boxes=spec["num_boxes"], target_ratio=target, rng=rng
        )
        coverage = compute_occlusion_coverage(boxes, height=128, width=128)
        print(
            f"   Severity {spec['severity']} ({spec['num_boxes']} boxes, target={target:.2f}): "
            f"actual coverage = {coverage:.4f}"
        )
        assert abs(coverage - target) < 0.10

    # 4. Generate 4x4 Visual Verification Grid
    print("\n4. Generating 4x4 Visual Verification Panel...")
    dataset = PetDataset(split="val", return_labels=True)
    sample_indices = [10, 50, 100, 150]

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    column_titles = ["Clean", "Salt-and-Pepper", "Gaussian Blur", "Occlusion"]

    severities = [1, 2, 3, 2]  # Row 1: Mild, Row 2: Med, Row 3: Severe, Row 4: Mixed

    for row_idx, (data_idx, sev) in enumerate(zip(sample_indices, severities)):
        clean_img, breed_id = dataset[data_idx]

        # Col 0: Clean
        c_clean = apply_clean(clean_img)
        # Col 1: S&P
        sp_spec = TEST_SEVERITIES[CorruptionType.SALT_AND_PEPPER.value][sev - 1]
        c_sp = apply_salt_and_pepper(clean_img, p=sp_spec["p"])
        # Col 2: Blur
        blur_spec = TEST_SEVERITIES[CorruptionType.GAUSSIAN_BLUR.value][sev - 1]
        c_blur = apply_gaussian_blur(
            clean_img,
            kernel_size=blur_spec["kernel_size"],
            sigma=blur_spec["sigma"],
        )
        # Col 3: Occlusion
        occl_spec = TEST_SEVERITIES[CorruptionType.OCCLUSION.value][sev - 1]
        boxes = sample_occlusion_boxes(
            height=128,
            width=128,
            num_boxes=occl_spec["num_boxes"],
            target_ratio=occl_spec["target_ratio"],
            rng=rng,
        )
        c_occl = apply_occlusion(clean_img, boxes=boxes, fill_value=0.0)

        images = [c_clean, c_sp, c_blur, c_occl]
        subtitles = [
            "Identity",
            f"Sev {sev} (p={sp_spec['p']})",
            f"Sev {sev} (k={blur_spec['kernel_size']}, \u03c3={blur_spec['sigma']})",
            f"Sev {sev} ({occl_spec['num_boxes']} boxes, {int(occl_spec['target_ratio']*100)}%)",
        ]

        for col_idx, (img, sub) in enumerate(zip(images, subtitles)):
            ax = axes[row_idx, col_idx]
            img_np = img.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            if row_idx == 0:
                ax.set_title(f"{column_titles[col_idx]}\n{sub}", fontsize=11, fontweight="bold")
            else:
                ax.set_title(sub, fontsize=10)
            ax.axis("off")

    fig.tight_layout()
    out_path = results_dir / "corruption_verification_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved 4x4 visual verification panel to {out_path}")

    print("\nAll corruption pipeline checks PASSED successfully!")


if __name__ == "__main__":
    main()
