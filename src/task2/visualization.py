"""
src/task2/visualization.py
--------------------------
Visualization utilities for Task 2 Corruption Classifier and Specialist Restoration:
- Training curves for classification loss, accuracy, and macro-F1.
- Publication-quality normalized confusion matrix heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["Clean", "Salt & Pepper", "Gaussian Blur", "Occlusion"]


def plot_classifier_metrics(
    history: Dict[str, List[float]],
    confusion_matrix_data: np.ndarray,
    vis_dir: Path,
    run_name: str,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Tuple[Path, Path]:
    """Generate and save publication-grade training curves and confusion matrix heatmap."""
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Curves (Loss and Accuracy / Macro-F1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#2b5c8f", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#d95f02", lw=2)
    ax1.set_title("Cross-Entropy Loss Progression", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], label="Val Accuracy (%)", marker="o", color="#1b9e77", lw=2)
    ax2.plot(
        epochs,
        [f * 100 for f in history["val_macro_f1"]],
        label="Val Macro-F1 (x100)",
        marker="^",
        color="#7570b3",
        lw=2,
    )
    ax2.set_title("Validation Accuracy & Macro-F1", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Percentage / Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    curves_path = vis_dir / f"{run_name}_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)

    # 2. Confusion Matrix Heatmap
    cm = np.array(confusion_matrix_data)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    n_classes = len(class_names)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Classifier Confusion Matrix (Normalized)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2%}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    cm_path = vis_dir / f"{run_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    return curves_path, cm_path


def plot_specialists_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Path,
) -> Path:
    """Generate multi-panel comparison curves showing Loss, PSNR, and SSIM across all 3 specialists."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = {
        "salt_and_pepper": "#2b5c8f",
        "gaussian_blur": "#d95f02",
        "occlusion": "#1b9e77",
    }
    labels = {
        "salt_and_pepper": "Salt & Pepper",
        "gaussian_blur": "Gaussian Blur",
        "occlusion": "Occlusion",
    }

    for corr, hist in histories.items():
        epochs = range(1, len(hist["val_loss"]) + 1)
        c = colors.get(corr, "#333333")
        lbl = labels.get(corr, corr)

        ax1.plot(epochs, hist["val_loss"], marker="o", label=lbl, color=c, lw=2)
        ax2.plot(epochs, hist["val_psnr"], marker="s", label=lbl, color=c, lw=2)
        ax3.plot(epochs, hist["val_ssim"], marker="^", label=lbl, color=c, lw=2)

    ax1.set_title("Validation Loss Progression", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Combined L1 + SSIM)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation PSNR (dB)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Validation SSIM", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("SSIM")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_specialists_reconstructions(
    samples: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    save_path: Path,
) -> Path:
    """Generate visual triplet inspection grid (Corrupted, Restored, Ground Truth) per specialist."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_specialists = len(samples)
    fig, axes = plt.subplots(num_specialists, 3, figsize=(9, 3 * num_specialists))
    if num_specialists == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Corrupted Input", "Specialist Restored", "Ground Truth Clean"]
    labels = {
        "salt_and_pepper": "Salt & Pepper",
        "gaussian_blur": "Gaussian Blur",
        "occlusion": "Occlusion",
    }

    for row_idx, (corr, (corrupted, restored, clean)) in enumerate(samples.items()):
        # Tensors: (3, H, W) in [0, 1]
        imgs = [
            corrupted.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0),
            restored.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0),
            clean.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0),
        ]

        for col_idx, img in enumerate(imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontweight="bold", pad=8)
            if col_idx == 0:
                ax.text(
                    -0.1,
                    0.5,
                    labels.get(corr, corr),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontweight="bold",
                    fontsize=11,
                )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

