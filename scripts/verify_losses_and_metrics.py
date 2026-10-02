"""
scripts/verify_losses_and_metrics.py
------------------------------------
Verifies loss calculation, metrics (PSNR, SSIM, MAE), and visual helpers.
Exports a verification figure (original, corrupted, restored, and heatmap)
to results/losses_metrics_verification.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.shared.config import get_settings
from src.shared.corruptions import apply_salt_and_pepper
from src.shared.losses import CombinedReconstructionLoss
from src.shared.metrics import evaluate_metrics
from src.shared.visualization import (
    make_error_heatmap,
    make_reconstruction_grid,
    plot_training_curves,
)


def main():
    settings = get_settings()
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Verifying Shared Losses, Metrics & Visualization Helpers")
    print("=" * 60)

    # 1. Test loss functions with synthetic gradients
    print("\n1. Testing CombinedReconstructionLoss backward pass...")
    loss_fn = CombinedReconstructionLoss(alpha=0.84, data_range=1.0)
    pred = torch.rand((4, 3, 128, 128), requires_grad=True)
    target = torch.rand((4, 3, 128, 128))

    loss, comps = loss_fn(pred, target, return_components=True)
    loss.backward()
    print(
        f"   Total Loss: {comps['loss_total']:.4f} "
        f"(L1: {comps['loss_l1']:.4f}, SSIM: {comps['loss_ssim']:.4f})"
    )
    assert pred.grad is not None and not torch.isnan(pred.grad).any()
    print("   Gradient backpropagation passed without NaNs.")

    # 2. Test metrics evaluation
    print("\n2. Evaluating Metrics on Synthetic Corrupted Data:")
    clean = torch.full((1, 3, 128, 128), 0.5)
    corrupted = apply_salt_and_pepper(clean, p=0.08)
    metrics = evaluate_metrics(corrupted, clean)
    print(
        f"   Corrupted metrics: PSNR = {metrics['psnr']:.2f} dB, "
        f"SSIM = {metrics['ssim']:.4f}, MAE = {metrics['mae']:.4f}"
    )
    assert metrics["psnr"] < 30.0
    assert metrics["ssim"] < 0.95

    # 3. Test Visual Helpers
    print("\n3. Testing Visual Grid and Error Heatmap...")
    grid = make_reconstruction_grid(corrupted, clean, clean, n_samples=1)
    assert grid.shape[0] == 3, f"Expected 3 channels in grid, got {grid.shape}"

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(clean[0].permute(1, 2, 0).numpy())
    axes[0].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(corrupted[0].permute(1, 2, 0).numpy())
    axes[1].set_title("Corrupted (S&P)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Mock restoration
    restored = torch.clamp(corrupted + 0.1 * (clean - corrupted), 0.0, 1.0)
    axes[2].imshow(restored[0].permute(1, 2, 0).numpy())
    axes[2].set_title("Model Output", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    heatmap_r = make_error_heatmap(restored, clean, cmap="inferno")
    axes[3].imshow(heatmap_r.permute(1, 2, 0).numpy())
    axes[3].set_title("Error Heatmap (Residual)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    fig.tight_layout()
    out_path = results_dir / "losses_metrics_verification.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"   Saved verification figure to {out_path}")

    # 4. Test training curves plotting
    history = {
        "train_loss": [0.45, 0.35, 0.28, 0.22, 0.18],
        "val_loss": [0.48, 0.38, 0.30, 0.25, 0.20],
        "val_psnr": [20.5, 23.1, 25.4, 27.2, 28.8],
        "val_ssim": [0.72, 0.79, 0.84, 0.88, 0.91],
    }
    curve_fig = plot_training_curves(history, title="Verification Training Curve")
    curve_out = results_dir / "sample_training_curves.png"
    curve_fig.savefig(curve_out, dpi=150)
    plt.close(curve_fig)
    print(f"   Saved sample training curves to {curve_out}")

    print("\nAll shared losses, metrics, and visual helper checks PASSED successfully!")


if __name__ == "__main__":
    main()
