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
