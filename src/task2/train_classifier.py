"""
src/task2/train_classifier.py
-----------------------------
Training pipeline for Task 2 Corruption Classifier.

Implements:
- Balanced multi-class batching (25% Clean, 25% S&P, 25% Blur, 25% Occlusion per batch).
- Dynamic stochastic corruption sampling for training data.
- Deterministic validation on manifests/val_manifest.json (736 images).
- AdamW optimizer with CosineAnnealingLR scheduler.
- Comprehensive classification metrics (Accuracy, Macro-F1, Per-class F1, Confusion Matrix).
- Model checkpointing (checkpoints/task2/classifier_best.pt) and MLflow tracking.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.shared.config import Settings, get_settings
from src.shared.tracking import ExperimentTracker
from src.task2.classifier import CorruptionClassifier, build_classifier
from src.task2.dataset import build_classifier_dataloaders
from src.task2.visualization import CLASS_NAMES, plot_classifier_metrics


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Exhaustive validation returning loss, accuracy, precision, recall, F1, and confusion matrix."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            total_samples += x.size(0)

    val_loss = total_loss / max(1, total_samples)
    val_acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, normalize="true")

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(per_p[i]), 4),
            "recall": round(float(per_r[i]), 4),
            "f1": round(float(per_f1[i]), 4),
        }

    return {
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 2),
        "val_macro_precision": round(float(macro_p), 4),
        "val_macro_recall": round(float(macro_r), 4),
        "val_macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def train_classifier(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    backbone: str = "custom_conv",
    run_name: str = "classifier-baseline",
    seed: int = 42,
    settings: Settings | None = None,
) -> Dict[str, Any]:
    """Train the Task 2 Corruption Classifier with balanced batches, cosine LR, and tracking."""
    cfg = settings or get_settings()
    device = cfg.torch_device
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n=======================================================")
    print(f"Task 2 Corruption Classifier Training: {run_name}")
    print(f"Backbone: {backbone} | Device: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"=======================================================", flush=True)

    # 1. Model & Optimization
    model = build_classifier(backbone=backbone, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 2. Data Preparation
    print("Preparing balanced dynamic training and deterministic validation loaders...", flush=True)
    train_loader, val_loader = build_classifier_dataloaders(batch_size=batch_size, seed=seed, settings=cfg)
    print(f"Loaders ready: {len(train_loader)} training batches, {len(val_loader)} validation batches.\n")

    # 3. Training Loop with Tracking
    tracker = ExperimentTracker(settings=cfg)
    tracker.start_run(run_name=run_name, experiment_name="genai-task2-hard-routing")
    tracker.log_params({
        "backbone": backbone, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "weight_decay": weight_decay, "dropout": dropout, "seed": seed,
    })

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_metrics: Dict[str, Any] = {}
    best_cm: np.ndarray = np.zeros((4, 4))
    ckpt_dir = Path("checkpoints/task2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "classifier_best.pt"
    latest_ckpt_path = ckpt_dir / "classifier_latest.pt"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_train_loss, train_samples = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)

        scheduler.step()
        epoch_train_loss = running_train_loss / max(1, train_samples)

        # Validation
        v_res = evaluate_classifier(model, val_loader, criterion, device)
        cur_lr = scheduler.get_last_lr()[0]
        t_epoch = time.time() - t0

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(v_res["val_loss"])
        history["val_acc"].append(v_res["val_acc"])
        history["val_macro_f1"].append(v_res["val_macro_f1"])

        tracker.log_metrics({
            "train_loss": epoch_train_loss,
            "val_loss": v_res["val_loss"],
            "val_acc": v_res["val_acc"],
            "val_macro_f1": v_res["val_macro_f1"],
            "lr": cur_lr,
        }, step=epoch)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({t_epoch:.1f}s) | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {v_res['val_loss']:.4f} | Val Acc: {v_res['val_acc']:.2f}% | "
            f"Macro-F1: {v_res['val_macro_f1']:.4f} | LR: {cur_lr:.2e}",
            flush=True,
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": v_res["val_acc"],
            "val_macro_f1": v_res["val_macro_f1"],
            "val_loss": v_res["val_loss"],
            "metrics": v_res,
            "config": {"backbone": backbone, "dropout": dropout},
        }
        torch.save(state, latest_ckpt_path)

        if v_res["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = v_res["val_macro_f1"]
            best_metrics = v_res
            best_cm = np.array(v_res["confusion_matrix"])
            torch.save(state, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Macro-F1: {best_macro_f1:.4f})", flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Macro-F1: {best_macro_f1:.4f}")

    # 4. Artifact Export
    res_dir = Path("results/task2")
    res_dir.mkdir(parents=True, exist_ok=True)
    curves_path, cm_path = plot_classifier_metrics(history, best_cm, res_dir, run_name)
    tracker.log_artifact(str(curves_path), artifact_path="visualizations")
    tracker.log_artifact(str(cm_path), artifact_path="visualizations")

    summary = {
        "run_name": run_name,
        "backbone": backbone,
        "epochs": epochs,
        "total_training_time_s": round(total_time, 2),
        "best_epoch": best_metrics.get("epoch", epochs),
        "best_val_macro_f1": best_macro_f1,
        "best_val_acc": best_metrics.get("val_acc", 0.0),
        "best_val_loss": best_metrics.get("val_loss", 0.0),
        "per_class": best_metrics.get("per_class", {}),
        "confusion_matrix": best_metrics.get("confusion_matrix", []),
        "history": history,
    }
    metrics_path = res_dir / f"{run_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tracker.log_artifact(str(metrics_path), artifact_path="metrics")
    tracker.end_run()

    print(f"Artifacts exported to:\n  - {curves_path}\n  - {cm_path}\n  - {metrics_path}\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 2 Corruption Classifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--backbone", type=str, default="custom_conv")
    parser.add_argument("--run-name", type=str, default="classifier-baseline")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        backbone=args.backbone,
        run_name=args.run_name,
        seed=args.seed,
    )
