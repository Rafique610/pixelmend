"""
src/task2/optuna_classifier.py
------------------------------
Optuna hyperparameter optimization study for Task 2 Corruption Classifier.

Performs Bayesian optimization with TPE and MedianPruner across:
- learning_rate: [1e-4, 1e-2] (log scale)
- batch_size: [16, 32, 64]
- channel_config: ['small', 'medium', 'large']
- dropout: [0.0, 0.5] (step 0.05)
- weight_decay: [1e-5, 1e-2] (log scale)

Direction: maximize validation macro-F1 score.
Persists study in SQLite (optuna/optuna_studies.db) and logs to MLflow.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as ovm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from src.shared.config import Settings, get_settings
from src.shared.corruptions import (
    LABEL_TO_TYPE,
    apply_corruption,
    sample_random_corruption,
)
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_val_manifest
from src.shared.optuna_utils import create_or_load_study, export_study_summary, make_tracking_callback
from src.shared.tracking import ExperimentTracker
from src.task2.classifier import build_classifier
from src.task2.dataset import (
    BalancedBatchSampler,
    BalancedDynamicTrainDataset,
    CachedValDataset,
)

CHANNEL_CONFIGS: Dict[str, Tuple[int, ...]] = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256),
    "large": (48, 96, 192, 256),
}


def evaluate_trial_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Fast evaluation returning validation loss, accuracy (%), and macro-F1 score."""
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
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    return float(val_loss), float(val_acc), float(macro_f1)


def create_classifier_objective(
    clean_tensors: List[torch.Tensor],
    cached_val_pairs: List[Tuple[torch.Tensor, int]],
    epochs_per_trial: int = 5,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Any:
    """Return an Optuna objective function with pre-cached training and validation data."""
    torch_device = device or torch.device("cpu")
    val_dataset = CachedValDataset(cached_val_pairs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        channel_name = trial.suggest_categorical("channel_config", ["small", "medium", "large"])
        channels = CHANNEL_CONFIGS[channel_name]
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        model = build_classifier(
            backbone="custom_conv",
            channels=channels,
            dropout=dropout,
        ).to(torch_device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_trial, eta_min=1e-5)

        train_dataset = BalancedDynamicTrainDataset(clean_tensors, image_size=128)
        batch_sampler = BalancedBatchSampler(
            train_dataset.labels, batch_size=batch_size, shuffle=True, seed=seed + trial.number
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

        best_val_macro_f1 = 0.0
        print(
            f"\n[Trial {trial.number:03d}] lr={lr:.2e}, bs={batch_size}, channels={channel_name}, "
            f"dropout={dropout:.2f}, wd={weight_decay:.2e}",
            flush=True,
        )

        for epoch in range(1, epochs_per_trial + 1):
            model.train()
            train_loss, train_samples = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(torch_device), y.to(torch_device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                train_samples += x.size(0)

            if train_samples > 0:
                scheduler.step()
            v_loss, v_acc, v_f1 = evaluate_trial_model(model, val_loader, criterion, torch_device)

            if v_f1 > best_val_macro_f1:
                best_val_macro_f1 = v_f1

            trial.report(v_f1, step=epoch)
            print(
                f"  Trial {trial.number:03d} | Epoch {epoch:02d}/{epochs_per_trial:02d} | "
                f"Train Loss: {train_loss/max(1, train_samples):.4f} | Val Loss: {v_loss:.4f} | "
                f"Val Acc: {v_acc:.2f}% | Val Macro-F1: {v_f1:.4f}",
                flush=True,
            )

            if trial.should_prune():
                print(f"  --> Trial {trial.number:03d} PRUNED at epoch {epoch} (F1: {v_f1:.4f}).", flush=True)
                raise optuna.TrialPruned()

        return best_val_macro_f1

    return objective


def run_classifier_optuna_study(
    n_trials: int = 20,
    epochs_per_trial: int = 5,
    study_name: str = "task2-classifier",
    pruner_name: str = "median",
    warmup_steps: int = 3,
    seed: int = 42,
    settings: Optional[Settings] = None,
) -> optuna.Study:
    """Execute Optuna hyperparameter study for Task 2 Corruption Classifier."""
    cfg = settings or get_settings()
    device = cfg.torch_device

    print(f"\n=======================================================")
    print(f"Task 2 Classifier Optuna Study: {study_name}")
    print(f"Device: {device} | Trials: {n_trials} | Epochs/Trial: {epochs_per_trial} | Pruner: {pruner_name}")
    print(f"=======================================================", flush=True)

    # 1. Pre-cache dataset in RAM
    print("Pre-caching dataset in memory (2,944 train clean, 736 val pairs)...", flush=True)
    base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=cfg)
    clean_tensors = [base_train[i] for i in range(len(base_train))]

    base_val = PetDataset(split="val", image_size=128, return_labels=False, settings=cfg)
    val_manifest = load_val_manifest(settings=cfg)
    cached_val_pairs: List[Tuple[torch.Tensor, int]] = []
    for item in val_manifest:
        clean = base_val[item["val_id"]]
        corr = apply_corruption(clean, item["corruption_type"], item["params"])
        cached_val_pairs.append((corr, item["corruption_label"]))

    # 2. Setup SQLite Study & MLflow Tracking
    study = create_or_load_study(
        study_name=study_name,
        direction="maximize",
        pruner_name=pruner_name,
        warmup_steps=warmup_steps,
        seed=seed,
        settings=cfg,
    )

    tracker = ExperimentTracker(settings=cfg)
    mlflow_cb = make_tracking_callback(tracker=tracker, experiment_name=study_name, metric_name="val_macro_f1")

    objective = create_classifier_objective(
        clean_tensors=clean_tensors,
        cached_val_pairs=cached_val_pairs,
        epochs_per_trial=epochs_per_trial,
        device=device,
        seed=seed,
    )

    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

    # 3. Best Trial Summary & Export
    print(f"\nStudy complete! Best Trial #{study.best_trial.number} with Val Macro-F1: {study.best_value:.4f}", flush=True)
    print(f"Winning Parameters: {study.best_params}", flush=True)

    res_dir = Path("results/task2")
    res_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir = Path("optuna")
    optuna_dir.mkdir(parents=True, exist_ok=True)

    best_config = {
        "study_name": study_name,
        "best_trial_number": study.best_trial.number,
        "best_val_macro_f1": round(study.best_value, 4),
        "best_params": study.best_params,
        "total_trials": len(study.trials),
        "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }

    best_json_path = res_dir / "classifier_best_hyperparams.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    export_study_summary(study, optuna_dir / f"{study_name}.json")

    # 4. Generate Optuna Visualizations
    try:
        fig_hist = ovm.plot_optimization_history(study)
        fig_hist.figure.savefig(res_dir / "classifier_optuna_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig_hist.figure)
    except Exception as e:
        print(f"Warning: Failed to plot optimization history: {e}")

    try:
        fig_imp = ovm.plot_param_importances(study)
        fig_imp.figure.savefig(res_dir / "classifier_optuna_param_importances.png", dpi=150, bbox_inches="tight")
        plt.close(fig_imp.figure)
    except Exception as e:
        print(f"Warning: Failed to plot parameter importances: {e}")

    print(f"Saved artifacts to:\n  - {best_json_path}\n  - {optuna_dir / f'{study_name}.json'}\n")
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna study for Task 2 Corruption Classifier")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=int, default=5)
    parser.add_argument("--study-name", type=str, default="task2-classifier")
    parser.add_argument("--pruner", type=str, default="median")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_classifier_optuna_study(
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        study_name=args.study_name,
        pruner_name=args.pruner,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
