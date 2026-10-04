"""
src/task4/train.py
------------------
Conditional GAN training engine for Face-to-Sketch synthesis on FS2K dataset.
Implements alternating min-max optimization with AMP mixed precision,
one-sided label smoothing, validation tracking (L1, PSNR, SSIM per style),
and fixed-sample visual progression grid generation.
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
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from src.shared.config import get_settings
from src.shared.metrics import compute_mae, compute_psnr, compute_ssim
from src.task4.dataset import FS2KDataset
from src.task4.discriminator import PatchGANDiscriminator
from src.task4.generator import UNetGenerator


def create_visual_grid(
    photos: torch.Tensor,
    reals: torch.Tensor,
    fakes: torch.Tensor,
    save_path: str | Path,
) -> None:
    """Save a side-by-side [Photo | Ground Truth | Synthesized] image grid."""
    # Convert from [-1, 1] to [0, 1]
    p = torch.clamp((photos + 1.0) / 2.0, 0.0, 1.0).cpu()
    r = torch.clamp((reals + 1.0) / 2.0, 0.0, 1.0).cpu()
    f = torch.clamp((fakes + 1.0) / 2.0, 0.0, 1.0).cpu()

    # Interleave rows: Photo, Real, Fake for each instance
    grid_list = []
    for i in range(p.size(0)):
        grid_list.extend([p[i], r[i], f[i]])
    combined = torch.stack(grid_list)

    grid = vutils.make_grid(combined, nrow=3, padding=4, normalize=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(save_path)


def evaluate_generator(
    net_g: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute validation L1, PSNR, SSIM overall and partitioned across styles."""
    net_g.eval()
    l1_vals: List[float] = []
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    by_style: Dict[int, Dict[str, List[float]]] = {0: {"l1": [], "psnr": [], "ssim": []},
                                                  1: {"l1": [], "psnr": [], "ssim": []},
                                                  2: {"l1": [], "psnr": [], "ssim": []}}

    with torch.no_grad():
        for batch in val_loader:
            photos = batch["photo"].to(device)
            reals = batch["sketch"].to(device)
            styles = batch["style"].to(device)

            fakes = net_g(photos, styles)

            # Map to [0, 1] for standardized evaluation
            fakes_01 = torch.clamp((fakes + 1.0) / 2.0, 0.0, 1.0)
            reals_01 = torch.clamp((reals + 1.0) / 2.0, 0.0, 1.0)

            for i in range(photos.size(0)):
                f_i = fakes_01[i]
                r_i = reals_01[i]
                s_i = int(styles[i].item())

                l1 = compute_mae(f_i, r_i)
                psnr = compute_psnr(f_i, r_i, max_val=1.0)
                ssim = compute_ssim(f_i, r_i, data_range=1.0)

                l1_vals.append(l1)
                psnr_vals.append(psnr)
                ssim_vals.append(ssim)

                if s_i in by_style:
                    by_style[s_i]["l1"].append(l1)
                    by_style[s_i]["psnr"].append(psnr)
                    by_style[s_i]["ssim"].append(ssim)

    metrics = {
        "val_l1": float(np.mean(l1_vals)),
        "val_psnr": float(np.mean(psnr_vals)),
        "val_ssim": float(np.mean(ssim_vals)),
    }
    for s_i, vals in by_style.items():
        if vals["l1"]:
            metrics[f"style_{s_i}_l1"] = float(np.mean(vals["l1"]))
            metrics[f"style_{s_i}_psnr"] = float(np.mean(vals["psnr"]))
            metrics[f"style_{s_i}_ssim"] = float(np.mean(vals["ssim"]))

    return metrics


def train_baseline(
    epochs: int = 80,
    batch_size: int = 16,
    g_lr: float = 2e-4,
    d_lr: float = 2e-4,
    lambda_l1: float = 100.0,
    base_channels: int = 64,
    embed_dim: int = 16,
    seed: int = 42,
    checkpoint_dir: str | Path = "checkpoints/task4",
    results_dir: str | Path = "results/task4",
) -> Dict[str, Any]:
    """Execute complete cGAN baseline training on FS2K dataset."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    settings = get_settings()
    device = settings.torch_device
    is_cuda = device.type == "cuda"
    print(f"=== Starting Task 4 cGAN Training on {device} ({epochs} epochs, B={batch_size}) ===")

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

    # Initialize Generator and Discriminator
    net_g = UNetGenerator(base_channels=base_channels, embed_dim=embed_dim).to(device)
    net_d = PatchGANDiscriminator(base_channels=base_channels, embed_dim=embed_dim, n_layers=3).to(device)

    opt_g = torch.optim.Adam(net_g.parameters(), lr=g_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=d_lr, betas=(0.5, 0.999))

    # Linear decay scheduler over second half of training
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

    history: List[Dict[str, Any]] = []
    best_val_l1 = float("inf")
    start_total_time = time.time()

    cp_path = Path(checkpoint_dir)
    res_path = Path(results_dir)
    cp_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)
    vis_dir = res_path / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

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

            # ----------------------------------------------------
            # 1. Update Discriminator D
            # ----------------------------------------------------
            opt_d.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                # Real pair with one-sided label smoothing (0.9 target)
                d_real = net_d(photo, real_sketch, style)
                real_target = torch.full_like(d_real, 0.9)
                loss_d_real = criterion_gan(d_real, real_target)

                # Fake pair
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

            # ----------------------------------------------------
            # 2. Update Generator G
            # ----------------------------------------------------
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

        # Validate
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

        # Print progress every 10 epochs (or first epoch)
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
            grid_file = vis_dir / f"visual_progression_epoch_{epoch:03d}.png"
            create_visual_grid(fixed_photos, fixed_reals, fakes_fixed, grid_file)

        # Save best generator checkpoint
        if val_metrics["val_l1"] < best_val_l1:
            best_val_l1 = val_metrics["val_l1"]
            torch.save(net_g.state_dict(), cp_path / "baseline_generator.pth")
            torch.save(net_d.state_dict(), cp_path / "baseline_discriminator.pth")

        # Save latest rolling checkpoint
        torch.save(
            {
                "epoch": epoch,
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "best_val_l1": best_val_l1,
            },
            cp_path / "latest_checkpoint.pth",
        )

    total_training_sec = time.time() - start_total_time
    print(f"\nTraining complete in {total_training_sec:.1f} seconds (~{total_training_sec/60:.1f} mins).")
    print(f"Best Validation L1 Error: {best_val_l1:.4f}")

    # Export history to JSON
    json_path = res_path / "baseline_train_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"total_time_sec": total_training_sec, "best_val_l1": best_val_l1, "history": history}, f, indent=2)

    # Log to MLflow
    try:
        mlflow.set_experiment("genai-task4-baseline")
        with mlflow.start_run(run_name="baseline-cgan-fs2k"):
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "g_lr": g_lr,
                "d_lr": d_lr,
                "lambda_l1": lambda_l1,
                "base_channels": base_channels,
                "embed_dim": embed_dim,
                "seed": seed,
                "device": str(device),
            })
            for rec in history:
                ep = rec["epoch"]
                for k, v in rec.items():
                    if isinstance(v, (int, float)) and k != "epoch":
                        mlflow.log_metric(k, v, step=ep)
            mlflow.log_artifact(str(json_path))
            for grid_img in vis_dir.glob("visual_progression_epoch_*.png"):
                mlflow.log_artifact(str(grid_img))
            print("Successfully logged baseline run to MLflow (genai-task4-baseline).")
    except Exception as e:
        print(f"MLflow warning: {e}")

    return {"total_time_sec": total_training_sec, "best_val_l1": best_val_l1, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task 4 cGAN baseline on FS2K")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    args = parser.parse_args()

    train_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        lambda_l1=args.lambda_l1,
    )
