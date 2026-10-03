"""Optuna hyperparameter optimization study for Task 1 Universal Autoencoder.

Performs Bayesian optimization with TPE and MedianPruner across learning rate,
batch size, bottleneck dimension, channel depth, dropout, and alpha weighting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as ovm
import torch
from torch.utils.data import DataLoader

from src.shared.config import Settings, get_settings
from src.shared.losses import CombinedReconstructionLoss
from src.shared.optuna_utils import create_or_load_study, export_study_summary, make_tracking_callback
from src.shared.tracking import ExperimentTracker
from src.task1.autoencoder import UniversalAutoencoder
from src.task1.train import CachedPetDataset, DynamicCachedTrainDataset, get_dataloaders, train_one_epoch, validate

CHANNEL_CONFIGS: Dict[str, Tuple[int, ...]] = {
    "compact": (32, 64, 128),
    "standard": (32, 64, 128, 256),
    "wide": (48, 96, 192, 256),
}


def create_objective(
    train_loader_factory: Any,
    val_loader: DataLoader,
    epochs_per_trial: int = 3,
    device: Optional[torch.device] = None,
) -> Any:
    """Return an Optuna objective function with pre-cached validation data."""
    torch_device = device or torch.device("cpu")

    def objective(trial: optuna.Trial) -> float:
        # 1. Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [128, 256])
        channel_name = trial.suggest_categorical("channel_depth", ["compact", "standard", "wide"])
        channels = CHANNEL_CONFIGS[channel_name]
        dropout = trial.suggest_float("dropout", 0.0, 0.25, step=0.05)
        alpha = trial.suggest_float("alpha", 0.65, 0.90, step=0.05)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # 2. Build model, optimizer, scheduler, criterion
        model = UniversalAutoencoder(
            in_channels=3,
            out_channels=3,
            channels=channels,
            bottleneck_dim=bottleneck_dim,
            use_residual=True,
            use_skips=False,
            dropout=dropout,
        ).to(torch_device)

        criterion = CombinedReconstructionLoss(alpha=alpha).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_trial)

        train_loader = train_loader_factory(batch_size=batch_size)

        best_trial_loss = float("inf")
        print(f"\n[Optuna Trial {trial.number:03d}] alpha={alpha:.2f}, lr={lr:.2e}, bs={batch_size}, bottleneck={bottleneck_dim}, channels={channel_name}", flush=True)

        # 3. Train epochs with pruning check
        for epoch in range(1, epochs_per_trial + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, torch_device, epoch, epochs_per_trial, log_interval=20)
            val_metrics, _, _, _ = validate(model, val_loader, criterion, torch_device)
            scheduler.step()

            cur_vloss = val_metrics["val_loss"]
            if cur_vloss < best_trial_loss:
                best_trial_loss = cur_vloss

            trial.report(cur_vloss, step=epoch)
            print(
                f"  Trial {trial.number:03d} | Epoch {epoch:02d}/{epochs_per_trial:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {cur_vloss:.4f} | Val SSIM: {val_metrics['ssim']:.4f} | PSNR: {val_metrics['psnr']:.2f}dB",
                flush=True,
            )

            if trial.should_prune():
                print(f"  --> Trial {trial.number:03d} PRUNED at epoch {epoch}.", flush=True)
                raise optuna.TrialPruned()

        return best_trial_loss

    return objective


def run_optuna_study(
    n_trials: int = 10,
    epochs_per_trial: int = 3,
    max_train_samples: int = 1000,
    study_name: str = "task1-universal-ae",
    pruner_name: str = "median",
    warmup_steps: int = 2,
    seed: int = 42,
    settings: Optional[Settings] = None,
) -> optuna.Study:
    """Execute Optuna hyperparameter study and persist study artifacts."""
    cfg = settings or get_settings()
    device = cfg.torch_device

    print(f"Starting Optuna Study '{study_name}' on {device} | {n_trials} trials | {epochs_per_trial} epochs/trial", flush=True)

    # Pre-cache datasets once so trials don't reload from disk
    print(f"Pre-caching dataset (max {max_train_samples} train, 736 val pairs)...", flush=True)
    _, val_loader = get_dataloaders(batch_size=32, max_train_samples=max_train_samples, settings=cfg)
    
    # Factory to produce DataLoader for trial-selected batch sizes
    from src.shared.datasets.pets import PetDataset
    base_train = PetDataset("train", settings=cfg)
    n_train = min(max_train_samples, len(base_train))
    clean_tensors = [base_train[i] for i in range(n_train)]

    def train_loader_factory(batch_size: int) -> DataLoader:
        return DataLoader(DynamicCachedTrainDataset(clean_tensors), batch_size=batch_size, shuffle=True)

    # Create / load study
    study = create_or_load_study(
        study_name=study_name,
        direction="minimize",
        pruner_name=pruner_name,
        warmup_steps=warmup_steps,
        seed=seed,
        settings=cfg,
    )

    tracker = ExperimentTracker(settings=cfg)
    mlflow_cb = make_tracking_callback(tracker=tracker, experiment_name=study_name, metric_name="val_loss")

    objective = create_objective(
        train_loader_factory=train_loader_factory,
        val_loader=val_loader,
        epochs_per_trial=epochs_per_trial,
        device=device,
    )

    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

    # Summarize & export
    print(f"\nOptuna study complete! Best trial: #{study.best_trial.number} with Val Loss: {study.best_value:.4f}", flush=True)
    print(f"Best hyperparameters: {study.best_params}", flush=True)

    # Save best hyperparams JSON
    met_dir = Path("results/task1/metrics")
    vis_dir = Path("results/task1/visualizations")
    met_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_config = {
        "study_name": study_name,
        "best_trial_number": study.best_trial.number,
        "best_val_loss": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    best_json_path = met_dir / "best_hyperparams.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    export_study_summary(study, Path("optuna/task1-universal-ae.json"))

    # Generate Optuna plots
    try:
        fig_hist = ovm.plot_optimization_history(study)
        fig_hist.figure.savefig(vis_dir / "optuna_optimization_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig_hist.figure)
    except Exception as e:
        print(f"Could not save optimization history plot: {e}", flush=True)

    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 2:
        try:
            fig_imp = ovm.plot_param_importances(study)
            fig_imp.figure.savefig(vis_dir / "optuna_param_importances.png", dpi=150, bbox_inches="tight")
            plt.close(fig_imp.figure)
        except Exception as e:
            print(f"Could not save parameter importances plot: {e}", flush=True)

    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna study for Task 1 Universal Autoencoder")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs-per-trial", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--study-name", type=str, default="task1-universal-ae")
    parser.add_argument("--pruner", type=str, default="median")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_optuna_study(
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        max_train_samples=args.max_train_samples,
        study_name=args.study_name,
        pruner_name=args.pruner,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
