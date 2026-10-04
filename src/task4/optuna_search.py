"""
src/task4/optuna_search.py
--------------------------
Optuna Bayesian hyperparameter optimization study for conditional GAN (Task 4).
Searches generator/discriminator learning rates, L1 loss weight (lambda_l1),
base channels, and dropout. Minimizes validation L1 reconstruction error.
Includes pruning of unpromising trials via MedianPruner and exports winning parameters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.config import get_settings
from src.shared.optuna_utils import create_or_load_study, make_tracking_callback
from src.task4.dataset import FS2KDataset
from src.task4.discriminator import PatchGANDiscriminator
from src.task4.generator import UNetGenerator
from src.task4.train import evaluate_generator


def sample_task4_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for conditional GAN training."""
    return {
        "g_lr": trial.suggest_float("g_lr", 1e-4, 5e-4, log=True),
        "d_lr": trial.suggest_float("d_lr", 1e-4, 5e-4, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 50.0, 150.0),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "base_channels": trial.suggest_categorical("base_channels", [32, 64]),
    }


def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trial_epochs: int = 15,
) -> float:
    """Optuna objective function: trains cGAN for trial_epochs and returns val_l1."""
    params = sample_task4_params(trial)
    is_cuda = device.type == "cuda"

    # Instantiate generator and discriminator
    net_g = UNetGenerator(
        base_channels=params["base_channels"],
        embed_dim=16,
        dropout=params["dropout"],
    ).to(device)
    net_d = PatchGANDiscriminator(
        base_channels=params["base_channels"],
        embed_dim=16,
        n_layers=3,
    ).to(device)

    opt_g = torch.optim.Adam(net_g.parameters(), lr=params["g_lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=params["d_lr"], betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    scaler_g = torch.amp.GradScaler("cuda", enabled=is_cuda)
    scaler_d = torch.amp.GradScaler("cuda", enabled=is_cuda)

    best_trial_val_l1 = float("inf")

    for epoch in range(1, trial_epochs + 1):
        net_g.train()
        net_d.train()

        for batch in train_loader:
            photo = batch["photo"].to(device)
            real_sketch = batch["sketch"].to(device)
            style = batch["style"].to(device)

            # 1. Update Discriminator D
            opt_d.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                d_real = net_d(photo, real_sketch, style)
                loss_d_real = criterion_gan(d_real, torch.full_like(d_real, 0.9))

                with torch.no_grad():
                    fake_sketch = net_g(photo, style)
                d_fake = net_d(photo, fake_sketch.detach(), style)
                loss_d_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # 2. Update Generator G
            opt_g.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                gen_sketch = net_g(photo, style)
                d_pred = net_d(photo, gen_sketch, style)
                loss_g_adv = criterion_gan(d_pred, torch.ones_like(d_pred))
                loss_g_l1 = criterion_l1(gen_sketch, real_sketch) * params["lambda_l1"]
                loss_g = loss_g_adv + loss_g_l1

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

        # Evaluate validation L1
        metrics = evaluate_generator(net_g, val_loader, device)
        val_l1 = metrics["val_l1"]

        if np.isnan(val_l1) or np.isinf(val_l1):
            return float("inf")

        if val_l1 < best_trial_val_l1:
            best_trial_val_l1 = val_l1

        trial.report(val_l1, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_trial_val_l1


def run_optuna_search(
    n_trials: int = 10,
    trial_epochs: int = 15,
    batch_size: int = 16,
    seed: int = 42,
    study_name: str = "task4-cgan",
) -> Dict[str, Any]:
    """Execute Optuna HPO study."""
    settings = get_settings()
    device = settings.torch_device
    is_cuda = device.type == "cuda"
    print(f"=== Starting Task 4 Optuna Search on {device} ({n_trials} trials, {trial_epochs} epochs/trial) ===")

    train_ds = FS2KDataset(root_dir="data/fs2k", split="train", augment=True, seed=seed)
    val_ds = FS2KDataset(root_dir="data/fs2k", split="val", augment=False, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )

    study = create_or_load_study(
        study_name=study_name,
        direction="minimize",
        pruner_name="median",
        warmup_steps=4,
        seed=seed,
    )

    mlflow_cb = make_tracking_callback(experiment_name="genai-task4-optuna", metric_name="val_l1")

    def _obj_wrapper(t: optuna.Trial) -> float:
        return objective(t, train_loader, val_loader, device, trial_epochs=trial_epochs)

    start_t = time.time()
    study.optimize(
        _obj_wrapper,
        n_trials=n_trials,
        callbacks=[mlflow_cb],
        timeout=840,  # 14 minute safety timeout (<15 min limit)
    )
    search_duration = time.time() - start_t

    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"\nOptuna study complete in {search_duration:.1f}s (~{search_duration/60:.1f} mins).")
    print(f"Best Trial #{best_trial.number}: Val L1 = {best_trial.value:.4f}")
    print(f"Best Parameters: {best_params}")

    out_data = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_val_l1": float(best_trial.value),
        "best_params": best_params,
        "search_duration_sec": search_duration,
    }

    # Save to config and results directories
    cfg_path = Path("config/task4_best_params.json")
    res_path = Path("results/task4/optuna_best_params.json")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna search for Task 4")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_optuna_search(n_trials=args.trials, trial_epochs=args.epochs, batch_size=args.batch_size)
