"""
src/task4/retrain.py
--------------------
Retraining engine for Task 4 conditional GAN using optimal hyperparameters
discovered by Optuna in Step 7. Trains the final production UNetGenerator
and PatchGANDiscriminator on the complete training schedule, logging progress,
saving best & latest checkpoints, and recording to MLflow 'genai-task4-final'.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.config import get_settings
from src.task4.dataset import FS2KDataset
from src.task4.discriminator import PatchGANDiscriminator
from src.task4.generator import UNetGenerator
from src.task4.train import create_visual_grid, evaluate_generator


def run_retrain(
    epochs: int = 80,
    batch_size: int = 16,
    seed: int = 42,
    params_path: str | Path = "config/task4_best_params.json",
    checkpoint_dir: str | Path = "checkpoints/task4",
    results_dir: str | Path = "results/task4",
) -> Dict[str, Any]:
    """Execute final cGAN retrain using best discovered hyperparameters."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    settings = get_settings()
    device = settings.torch_device
    is_cuda = device.type == "cuda"

    # Load best hyperparameters
    p_path = Path(params_path)
    if p_path.exists():
        with open(p_path, "r", encoding="utf-8") as f:
            hparams = json.load(f).get("best_params", {})
    else:
        hparams = {}

    g_lr = float(hparams.get("g_lr", 2e-4))
    d_lr = float(hparams.get("d_lr", 2e-4))
    lambda_l1 = float(hparams.get("lambda_l1", 100.0))
    dropout = float(hparams.get("dropout", 0.0))
    base_channels = int(hparams.get("base_channels", 64))
    embed_dim = 16

    print(f"=== Starting Task 4 Final Retrain on {device} ({epochs} epochs, B={batch_size}) ===")
    print(f"Hyperparameters: g_lr={g_lr:.6f}, d_lr={d_lr:.6f}, lambda_l1={lambda_l1:.2f}, base_channels={base_channels}, dropout={dropout:.3f}")

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

    # Reserve 6 fixed validation face samples (2 for each style) for visual tracking
    fixed_samples: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: [], 2: []}
    for item in val_ds:
        st = item["style"]
        if len(fixed_samples[st]) < 2:
            fixed_samples[st].append((item["photo"], item["sketch"]))
        if all(len(v) == 2 for v in fixed_samples.values()):
            break

    fixed_photos = torch.stack([pair[0] for st in (0, 1, 2) for pair in fixed_samples[st]]).to(device)
    fixed_reals = torch.stack([pair[1] for st in (0, 1, 2) for pair in fixed_samples[st]]).to(device)
    fixed_styles = torch.tensor([st for st in (0, 1, 2) for _ in range(2)], dtype=torch.long, device=device)

    # Construct models
    net_g = UNetGenerator(
        base_channels=base_channels,
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)
    net_d = PatchGANDiscriminator(
        base_channels=base_channels,
        embed_dim=embed_dim,
        n_layers=3,
    ).to(device)

    opt_g = torch.optim.Adam(net_g.parameters(), lr=g_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=d_lr, betas=(0.5, 0.999))

    def lr_lambda(epoch: int) -> float:
        decay_start = epochs // 2
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / float(epochs - decay_start)

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_lambda)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lr_lambda)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    scaler_g = torch.amp.GradScaler("cuda", enabled=is_cuda)
    scaler_d = torch.amp.GradScaler("cuda", enabled=is_cuda)

    cp_path = Path(checkpoint_dir)
    res_path = Path(results_dir)
    cp_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)
    vis_dir = res_path / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_val_l1 = float("inf")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        net_g.train()
        net_d.train()

        d_real_losses, d_fake_losses = [], []
        g_adv_losses, g_l1_losses = [], []

        for batch in train_loader:
            photo = batch["photo"].to(device)
            real_sketch = batch["sketch"].to(device)
            style = batch["style"].to(device)

            # 1. Update Discriminator D
            opt_d.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                d_real = net_d(photo, real_sketch, style)
                real_target = torch.full_like(d_real, 0.9)
                loss_d_real = criterion_gan(d_real, real_target)

                with torch.no_grad():
                    fake_sketch = net_g(photo, style)
                d_fake = net_d(photo, fake_sketch.detach(), style)
                fake_target = torch.zeros_like(d_fake)
                loss_d_fake = criterion_gan(d_fake, fake_target)

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            d_real_losses.append(loss_d_real.item())
            d_fake_losses.append(loss_d_fake.item())

            # 2. Update Generator G
            opt_g.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                gen_sketch = net_g(photo, style)
                d_pred = net_d(photo, gen_sketch, style)
                loss_g_adv = criterion_gan(d_pred, torch.ones_like(d_pred))
                loss_g_l1 = criterion_l1(gen_sketch, real_sketch) * lambda_l1
                loss_g = loss_g_adv + loss_g_l1

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            g_adv_losses.append(loss_g_adv.item())
            g_l1_losses.append(loss_g_l1.item())

        sched_g.step()
        sched_d.step()

        val_metrics = evaluate_generator(net_g, val_loader, device)
        epoch_dur = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "d_loss": float(0.5 * (np.mean(d_real_losses) + np.mean(d_fake_losses))),
            "g_adv_loss": float(np.mean(g_adv_losses)),
            "g_l1_loss": float(np.mean(g_l1_losses)),
            "epoch_sec": round(epoch_dur, 2),
            **val_metrics,
        }
        history.append(epoch_record)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"D Loss: {epoch_record['d_loss']:.4f} | "
                f"G Adv: {epoch_record['g_adv_loss']:.4f} | "
                f"G L1: {epoch_record['g_l1_loss']:.2f} | "
                f"Val L1: {val_metrics['val_l1']:.4f} | "
                f"PSNR: {val_metrics['val_psnr']:.2f} dB | "
                f"SSIM: {val_metrics['val_ssim']:.4f} "
                f"({epoch_dur:.1f}s)"
            )

        # Save visual progression grid
        if epoch in (1, 20, 40, 60, epochs):
            net_g.eval()
            with torch.no_grad():
                fakes_fixed = net_g(fixed_photos, fixed_styles)
            grid_file = vis_dir / f"final_progression_epoch_{epoch:03d}.png"
            create_visual_grid(fixed_photos, fixed_reals, fakes_fixed, grid_file)

        # Checkpoint best generator
        if val_metrics["val_l1"] < best_val_l1:
            best_val_l1 = val_metrics["val_l1"]
            torch.save(net_g.state_dict(), cp_path / "best_generator.pth")
            torch.save(net_d.state_dict(), cp_path / "best_discriminator.pth")

        # Rolling checkpoint
        torch.save(
            {
                "epoch": epoch,
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "best_val_l1": best_val_l1,
                "hparams": hparams,
            },
            cp_path / "best_latest_checkpoint.pth",
        )

    total_time_sec = time.time() - start_total_time
    print(f"\nFinal retraining complete in {total_time_sec:.1f} seconds (~{total_time_sec/60:.1f} mins).")
    print(f"Best Validation L1: {best_val_l1:.4f}")

    # Export metrics JSON
    train_metrics_file = res_path / "final_train_metrics.json"
    with open(train_metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_time_sec": total_time_sec,
                "best_val_l1": best_val_l1,
                "hparams": hparams,
                "history": history,
            },
            f,
            indent=2,
        )

    # Log to MLflow
    try:
        mlflow.set_experiment("genai-task4-final")
        with mlflow.start_run(run_name="final-cgan-fs2k"):
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "g_lr": g_lr,
                "d_lr": d_lr,
                "lambda_l1": lambda_l1,
                "base_channels": base_channels,
                "dropout": dropout,
                "embed_dim": embed_dim,
                "seed": seed,
                "device": str(device),
            })
            for rec in history:
                ep = rec["epoch"]
                for k, v in rec.items():
                    if isinstance(v, (int, float)) and k != "epoch":
                        mlflow.log_metric(k, v, step=ep)
            mlflow.log_artifact(str(train_metrics_file))
            for grid_img in vis_dir.glob("final_progression_epoch_*.png"):
                mlflow.log_artifact(str(grid_img))
            print("Successfully logged final retraining run to MLflow (genai-task4-final).")
    except Exception as e:
        print(f"MLflow warning: {e}")

    return {"total_time_sec": total_time_sec, "best_val_l1": best_val_l1, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain Task 4 cGAN with optimal hyperparameters")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--params", type=str, default="config/task4_best_params.json")
    args = parser.parse_args()

    run_retrain(epochs=args.epochs, batch_size=args.batch_size, params_path=args.params)
