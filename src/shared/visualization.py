"""
src/shared/visualization.py
---------------------------
Visual logging helpers, reconstruction comparison grids, error heatmaps,
and training curve plotters for Tasks 1 through 4.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from src.shared.metrics import evaluate_metrics


def make_reconstruction_grid(
    corrupted: torch.Tensor,
    restored: torch.Tensor,
    target: torch.Tensor,
    n_samples: int = 8,
) -> torch.Tensor:
    """
    Construct a 3-row comparison grid:
    - Row 1: Corrupted Inputs
    - Row 2: Model Restorations
    - Row 3: Ground Truth Targets

    Parameters:
    -----------
    corrupted: (B, 3, H, W) in [0.0, 1.0]
    restored:  (B, 3, H, W) in [0.0, 1.0]
    target:    (B, 3, H, W) in [0.0, 1.0]
    n_samples: number of image columns to display

    Returns:
    --------
    torch.Tensor of shape (3, Grid_H, Grid_W) in [0.0, 1.0]
    """
    c = corrupted if corrupted.ndim == 4 else corrupted.unsqueeze(0)
    r = restored if restored.ndim == 4 else restored.unsqueeze(0)
    t = target if target.ndim == 4 else target.unsqueeze(0)

    num = min(n_samples, c.size(0), r.size(0), t.size(0))
    c_sub = c[:num].detach().cpu()
    r_sub = r[:num].detach().cpu()
    t_sub = t[:num].detach().cpu()

    # Stack vertically: all corrupted, then all restored, then all targets
    stacked = torch.cat([c_sub, r_sub, t_sub], dim=0)
    grid = vutils.make_grid(stacked, nrow=num, padding=2, normalize=False)
    return torch.clamp(grid, 0.0, 1.0)


def make_error_heatmap(
    restored: torch.Tensor,
    target: torch.Tensor,
    cmap: str = "inferno",
) -> torch.Tensor:
    """
    Generate an absolute error heatmap |target - restored| colormapped as an RGB tensor.

    Parameters:
    -----------
    restored: (C, H, W) or (B, C, H, W) in [0.0, 1.0]
    target:   (C, H, W) or (B, C, H, W) in [0.0, 1.0]
    cmap: Matplotlib colormap identifier (e.g. 'inferno', 'magma', 'jet')

    Returns:
    --------
    torch.Tensor of shape (3, H, W) in [0.0, 1.0]
    """
    r = restored if restored.ndim == 3 else restored[0]
    t = target if target.ndim == 3 else target[0]

    # Compute per-pixel absolute error averaged across channels: (H, W)
    diff = torch.abs(t.detach().cpu() - r.detach().cpu()).mean(dim=0).numpy()

    # Normalize to [0, 1] relative to max possible error
    diff_norm = np.clip(diff / 0.5, 0.0, 1.0)  # Scale so 0.5 error reaches peak heatmap intensity

    colormap = matplotlib.colormaps[cmap]
    rgba = colormap(diff_norm)  # (H, W, 4) in [0, 1]
    rgb = rgba[:, :, :3]  # (H, W, 3)

    # Convert to torch (3, H, W)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return torch.clamp(tensor, 0.0, 1.0)


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training & Validation Metrics",
) -> plt.Figure:
    """
    Generate a 3-panel figure displaying:
    1. Training & Validation Loss
    2. Validation PSNR (dB)
    3. Validation SSIM

    Parameters:
    -----------
    history: dict containing keys like 'train_loss', 'val_loss', 'val_psnr', 'val_ssim'

    Returns:
    --------
    matplotlib.figure.Figure ready for report export or tracker logging
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # 1. Loss Panel
    ax0 = axes[0]
    if "train_loss" in history and history["train_loss"]:
        ax0.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history and history["val_loss"]:
        val_epochs = range(1, len(history["val_loss"]) + 1)
        ax0.plot(
            val_epochs,
            history["val_loss"],
            label="Val Loss",
            color="#d62728",
            lw=2,
            linestyle="--",
        )
    ax0.set_title("Loss", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(frameon=True)

    # 2. PSNR Panel
    ax1 = axes[1]
    if "val_psnr" in history and history["val_psnr"]:
        val_epochs = range(1, len(history["val_psnr"]) + 1)
        ax1.plot(val_epochs, history["val_psnr"], label="Val PSNR", color="#2ca02c", lw=2)
        ax1.set_title("Peak Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("PSNR (dB)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True)

    # 3. SSIM Panel
    ax2 = axes[2]
    if "val_ssim" in history and history["val_ssim"]:
        val_epochs = range(1, len(history["val_ssim"]) + 1)
        ax2.plot(val_epochs, history["val_ssim"], label="Val SSIM", color="#9467bd", lw=2)
        ax2.set_title("Structural Similarity Index", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("SSIM")
        ax2.set_ylim([0.0, 1.05])
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(frameon=True)

    fig.tight_layout()
    return fig


def log_epoch_visuals(
    tracker: Any,
    corrupted: torch.Tensor,
    restored: torch.Tensor,
    target: torch.Tensor,
    step: int,
    prefix: str = "val",
    n_samples: int = 8,
) -> dict[str, float]:
    """
    Log reconstruction grid, error heatmap, and quantitative metrics to the ExperimentTracker.

    Returns:
    --------
    dict of evaluated metrics: {'psnr': float, 'ssim': float, 'mae': float}
    """
    grid = make_reconstruction_grid(corrupted, restored, target, n_samples=n_samples)
    heatmap = make_error_heatmap(restored, target)

    tracker.log_image(f"{prefix}_reconstruction_grid", grid, step=step)
    tracker.log_image(f"{prefix}_error_heatmap", heatmap, step=step)

    metrics = evaluate_metrics(restored, target)
    tracker.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()}, step=step)
    return metrics
