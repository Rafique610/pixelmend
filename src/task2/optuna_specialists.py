"""Optuna hyperparameter optimization study for Task 2 Specialist Autoencoders.

Discovers the optimal shared autoencoder topology and training parameters:
- learning_rate: [1e-4, 5e-3] (log scale)
- bottleneck_dim: [64, 128, 256]
- channel_config: ['shallow', 'standard', 'deep']
- batch_size: [16, 32]
- alpha: [0.60, 0.95] (step 0.05)
- use_residual: [True, False]

Optimization Direction: minimize validation reconstruction loss.
Persists study in SQLite (optuna/optuna_studies.db) and logs to MLflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as ovm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.shared.config import Settings, get_settings
from src.shared.corruptions import (
    CorruptionType,
    apply_corruption,
    sample_random_corruption,
)
from src.shared.datasets.pets import PetDataset
from src.shared.losses import CombinedReconstructionLoss
from src.shared.manifests import load_val_manifest
from src.shared.metrics import compute_psnr, compute_ssim
from src.shared.optuna_utils import (
    create_or_load_study,
    export_study_summary,
    make_tracking_callback,
)
from src.shared.tracking import ExperimentTracker
from src.task2.specialist import SpecialistAutoencoder, VALID_SPECIALISTS

SPECIALIST_CHANNEL_CONFIGS: Dict[str, Tuple[int, ...]] = {
    "shallow": (32, 64, 128),
    "standard": (32, 64, 128, 256),
    "deep": (48, 96, 192, 256),
}


class SimplePairDataset(Dataset):
    """In-memory paired tensor dataset for rapid Optuna trial evaluation."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


def build_proxy_subsets(
    train_per_corruption: int = 64,
    val_per_corruption: int = 32,
    seed: int = 42,
    settings: Optional[Settings] = None,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Construct balanced multi-corruption proxy subsets across Salt, Blur, and Occlusion."""
    cfg = settings or get_settings()
    torch.manual_seed(seed)

    base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=cfg)
    base_val = PetDataset(split="val", image_size=128, return_labels=False, settings=cfg)
    val_manifest = load_val_manifest(settings=cfg)

    # 1. Validation proxy from val_manifest (32 per corruption = 96 pairs)
    lbl_map = {"salt_and_pepper": 1, "gaussian_blur": 2, "occlusion": 3}
    val_by_corr = {c: [] for c in VALID_SPECIALISTS}
    for item in val_manifest:
        for c_name, lbl in lbl_map.items():
            if item["corruption_label"] == lbl:
                val_by_corr[c_name].append(item)

    val_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for c_name in VALID_SPECIALISTS:
        for it in val_by_corr[c_name][:val_per_corruption]:
            clean = base_val[it["val_id"]]
            corr = apply_corruption(clean, it["corruption_type"], it["params"])
            val_pairs.append((corr, clean))

    # 2. Training proxy from base_train (64 per corruption = 192 pairs)
    perm = torch.randperm(len(base_train)).tolist()
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for c_idx, c_name in enumerate(VALID_SPECIALISTS):
        for i in range(train_per_corruption):
            img_idx = perm[(i * 3 + c_idx) % len(perm)]
            clean = base_train[img_idx]

            if c_name == "salt_and_pepper":
                corr = apply_corruption(clean, CorruptionType.SALT_AND_PEPPER, {"prob": 0.08})
            elif c_name == "gaussian_blur":
                corr = apply_corruption(clean, CorruptionType.GAUSSIAN_BLUR, {"kernel_size": 5, "sigma": 1.5})
            else:  # occlusion
                corr = apply_corruption(clean, CorruptionType.OCCLUSION, {"boxes": [[25, 25, 75, 75]], "fill_value": 0.0})

            train_pairs.append((corr, clean))

    # Shuffle training pairs
    shuffled_idx = torch.randperm(len(train_pairs)).tolist()
    train_pairs = [train_pairs[i] for i in shuffled_idx]

    return train_pairs, val_pairs


def create_specialist_objective(
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    val_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    epochs_per_trial: int = 5,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Callable[[optuna.Trial], float]:
    """Return Optuna objective minimizing validation reconstruction loss on multi-corruption proxy."""

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(seed + trial.number)

        lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [64, 128, 256])
        config_name = trial.suggest_categorical("channel_config", ["shallow", "standard", "deep"])
        channels = SPECIALIST_CHANNEL_CONFIGS[config_name]
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        alpha = trial.suggest_float("alpha", 0.60, 0.95, step=0.05)
        use_residual = trial.suggest_categorical("use_residual", [True, False])

        model = SpecialistAutoencoder(
            in_channels=3,
            out_channels=3,
            channels=channels,
            bottleneck_dim=bottleneck_dim,
            use_residual=use_residual,
        ).to(device)

        criterion = CombinedReconstructionLoss(alpha=alpha)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        train_loader = DataLoader(SimplePairDataset(train_pairs), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SimplePairDataset(val_pairs), batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")

        for epoch in range(1, epochs_per_trial + 1):
            model.train()
            for corrupted, clean in train_loader:
                corrupted, clean = corrupted.to(device), clean.to(device)
                optimizer.zero_grad()
                pred = model(corrupted)
                loss = criterion(pred, clean)
                loss.backward()
                optimizer.step()

            # Validation pass
            model.eval()
            total_loss, total_count = 0.0, 0
            with torch.no_grad():
                for corrupted, clean in val_loader:
                    corrupted, clean = corrupted.to(device), clean.to(device)
                    pred = model(corrupted)
                    loss = criterion(pred, clean)
                    total_loss += loss.item() * clean.size(0)
                    total_count += clean.size(0)

            val_loss = total_loss / max(1, total_count)
            best_val_loss = min(best_val_loss, val_loss)

            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(best_val_loss)

    return objective


def run_specialist_optuna_study(
    n_trials: int = 16,
    epochs_per_trial: int = 5,
    seed: int = 42,
    study_name: str = "task2-specialists",
    device: Optional[torch.device] = None,
    settings: Optional[Settings] = None,
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """Execute SQLite-persisted Optuna HPO study with MedianPruner and MLflow tracking."""
    cfg = settings or get_settings()
    dev = device or torch.device("cpu")
    print(f"Launching Optuna Study '{study_name}' ({n_trials} trials, {epochs_per_trial} epochs/trial) on {dev}...")

    train_pairs, val_pairs = build_proxy_subsets(
        train_per_corruption=64,
        val_per_corruption=32,
        seed=seed,
        settings=cfg,
    )

    study = create_or_load_study(
        study_name=study_name,
        direction="minimize",
        pruner_name="median",
        seed=seed,
        settings=cfg,
    )

    tracker = ExperimentTracker(settings=cfg)
    callbacks = [make_tracking_callback(tracker=tracker, experiment_name=study_name, metric_name="val_loss")]

    objective = create_specialist_objective(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs_per_trial=epochs_per_trial,
        device=dev,
        seed=seed,
    )

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    # Export Study Summary
    export_dir = Path("optuna") / f"{study_name}.json"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_study_summary(study, output_dir=export_dir)

    best_trial = study.best_trial
    best_params = {
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_val_loss": round(best_trial.value, 4),
        "best_params": best_trial.params,
        "total_trials": len(study.trials),
        "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    # Save best parameters to results/task2/
    res_dir = Path("results/task2")
    res_dir.mkdir(parents=True, exist_ok=True)
    best_json_path = res_dir / "specialists_best_hyperparams.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Plot Optimization History & Parameter Importances
    history_plot_path = res_dir / "specialists_optuna_history.png"
    importance_plot_path = res_dir / "specialists_optuna_param_importances.png"

    try:
        fig_hist = ovm.plot_optimization_history(study)
        fig_hist.figure.savefig(history_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig_hist.figure)
    except Exception as e:
        print(f"Warning: could not save optimization history plot: {e}")

    try:
        fig_imp = ovm.plot_param_importances(study)
        fig_imp.figure.savefig(importance_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig_imp.figure)
    except Exception as e:
        print(f"Warning: could not save parameter importances plot: {e}")

    print(f"OPTUNA STUDY COMPLETE: {study_name} | Best Trial #{best_trial.number} (Val Loss: {best_trial.value:.4f})")
    print(f"Winning Parameters: {best_trial.params}\nSaved artifacts to {best_json_path}\n" + "=" * 80)
    return study, best_params


def main() -> None:
    """CLI entrypoint for running Task 2 Specialist Optuna search."""
    parser = argparse.ArgumentParser(description="Run Task 2 Specialist Optuna HPO.")
    parser.add_argument("--n-trials", type=int, default=16, help="Number of trials.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_specialist_optuna_study(n_trials=args.n_trials, epochs_per_trial=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
