"""Training pipeline for Task 1 Universal Denoising Autoencoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.shared.config import Settings, get_settings
from src.shared.corruptions import sample_random_corruption, apply_corruption
from src.shared.datasets.corrupted import CorruptedPetDataset, get_corrupted_pet_dataloader
from src.shared.datasets.pets import PetDataset
from src.shared.losses import CombinedReconstructionLoss
from src.shared.metrics import evaluate_metrics
from src.shared.tracking import ExperimentTracker
from src.shared.visualization import make_reconstruction_grid, plot_training_curves
from src.task1.autoencoder import UniversalAutoencoder


class CachedPetDataset(Dataset):
    """Memory-cached dataset for fast validation."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.pairs[idx]


class DynamicCachedTrainDataset(Dataset):
    """Dynamic corruption dataset using pre-loaded clean tensors for fast training."""

    def __init__(self, clean_tensors: List[torch.Tensor], image_size: int = 128):
        self.clean_tensors = clean_tensors
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.clean_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        clean = self.clean_tensors[idx]
        c_type, label, params = sample_random_corruption(rng=None, image_size=self.image_size)
        corrupted = apply_corruption(clean, c_type, params)
        return corrupted, clean, label


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
    log_interval: int = 15,
) -> float:
    """Run one epoch of training over dynamic corruptions with live batch heartbeat."""
    model.train()
    total_loss, num_batches = 0.0, len(dataloader)
    epoch_start = time.time()

    for batch_idx, (corrupted, clean, _) in enumerate(dataloader, 1):
        b_start = time.time()
        corrupted, clean = corrupted.to(device), clean.to(device)

        optimizer.zero_grad()
        restored = model(corrupted)
        loss = criterion(restored, clean)
        loss.backward()
        optimizer.step()

        b_loss = loss.item()
        total_loss += b_loss

        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            elapsed = time.time() - epoch_start
            speed = corrupted.size(0) / max(0.001, time.time() - b_start)
            eta = (elapsed / batch_idx) * (num_batches - batch_idx)
            print(
                f"  [Epoch {epoch:02d}/{total_epochs:02d} | Batch {batch_idx:02d}/{num_batches:02d} "
                f"({100 * batch_idx / num_batches:2.0f}%)] Loss: {b_loss:.4f} | Speed: {speed:.1f} img/s | ETA: {eta:.0f}s",
                flush=True,
            )

    return total_loss / max(1, num_batches)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run validation across deterministic manifest and collect aggregate + per-type metrics."""
    model.eval()
    val_loss, num_batches = 0.0, 0
    metrics_acc = {"psnr": 0.0, "ssim": 0.0, "mae": 0.0}
    c_names = {0: "clean", 1: "sp", 2: "blur", 3: "occl"}
    c_acc = {name: {"psnr": 0.0, "ssim": 0.0, "count": 0} for name in c_names.values()}
    sample_corrupted, sample_clean, sample_restored = None, None, None

    with torch.no_grad():
        for corrupted, clean, c_labels in dataloader:
            corrupted, clean = corrupted.to(device), clean.to(device)
            restored = model(corrupted)
            val_loss += criterion(restored, clean).item()

            batch_metrics = evaluate_metrics(clean, restored)
            for k in metrics_acc:
                metrics_acc[k] += batch_metrics[k]

            for i, label_idx in enumerate(c_labels.tolist()):
                c_name = c_names.get(label_idx, "unknown")
                if c_name in c_acc:
                    sub_m = evaluate_metrics(clean[i : i + 1], restored[i : i + 1])
                    c_acc[c_name]["psnr"] += sub_m["psnr"]
                    c_acc[c_name]["ssim"] += sub_m["ssim"]
                    c_acc[c_name]["count"] += 1

            if sample_corrupted is None:
                sample_corrupted, sample_clean, sample_restored = corrupted[:8].cpu(), clean[:8].cpu(), restored[:8].cpu()
            num_batches += 1

    avg_metrics = {k: v / max(1, num_batches) for k, v in metrics_acc.items()}
    avg_metrics["val_loss"] = val_loss / max(1, num_batches)
    for c_name, data in c_acc.items():
        cnt = max(1, data["count"])
        avg_metrics[f"{c_name}_psnr"] = data["psnr"] / cnt
        avg_metrics[f"{c_name}_ssim"] = data["ssim"] / cnt

    assert sample_corrupted is not None and sample_clean is not None and sample_restored is not None
    return avg_metrics, sample_corrupted, sample_restored, sample_clean


def get_dataloaders(
    batch_size: int,
    quick: bool = False,
    cache: bool = True,
    max_train_samples: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Factory creating train and validation loaders with in-memory caching."""
    cfg = settings or get_settings()
    if quick:
        train_ds, val_ds = CorruptedPetDataset("train", settings=cfg), CorruptedPetDataset("val", settings=cfg)
        return (
            DataLoader(Subset(train_ds, list(range(64))), batch_size=16, shuffle=True),
            DataLoader(Subset(val_ds, list(range(32))), batch_size=16, shuffle=False),
        )

    if not cache:
        return (
            get_corrupted_pet_dataloader("train", batch_size=batch_size, settings=cfg),
            get_corrupted_pet_dataloader("val", batch_size=batch_size, shuffle=False, settings=cfg),
        )

    print("Pre-caching dataset tensors into memory for zero-I/O execution...", flush=True)
    base_train = PetDataset("train", settings=cfg)
    total_samples = len(base_train) if max_train_samples is None else min(max_train_samples, len(base_train))
    clean_tensors = [base_train[i] for i in range(total_samples)]
    raw_val = CorruptedPetDataset("val", settings=cfg)
    val_pairs = [raw_val[i] for i in range(len(raw_val))]
    print(f"Cached {len(clean_tensors)} train tensors and {len(val_pairs)} val pairs.", flush=True)

    return (
        DataLoader(DynamicCachedTrainDataset(clean_tensors), batch_size=batch_size, shuffle=True),
        DataLoader(CachedPetDataset(val_pairs), batch_size=batch_size, shuffle=False),
    )


def run_training(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    alpha: float = 0.8,
    bottleneck_dim: int = 256,
    channels: Tuple[int, ...] = (32, 64, 128, 256),
    use_residual: bool = True,
    use_skips: bool = False,
    dropout: float = 0.0,
    patience: int = 8,
    checkpoint_path: Optional[Path] = None,
    experiment_name: str = "task1-universal-ae",
    run_name: str = "baseline",
    sample_interval: int = 5,
    quick: bool = False,
    cache: bool = True,
    max_train_samples: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Execute complete training and validation pipeline."""
    cfg = settings or get_settings()
    ckpt_dir = cfg.checkpoints_dir / "task1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = checkpoint_path or (ckpt_dir / f"{run_name}_best.pth")

    print(f"Initializing Task 1 Training on {cfg.torch_device} | Run: {run_name} | alpha: {alpha}", flush=True)
    model = UniversalAutoencoder(
        in_channels=3, out_channels=3, channels=channels,
        bottleneck_dim=bottleneck_dim, use_residual=use_residual,
        use_skips=use_skips, dropout=dropout,
    ).to(cfg.torch_device)

    criterion = CombinedReconstructionLoss(alpha=alpha).to(cfg.torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader, val_loader = get_dataloaders(batch_size, quick, cache, max_train_samples, cfg)

    tracker = ExperimentTracker(settings=cfg)
    tracker.start_run(run_name=run_name, experiment_name=experiment_name)
    tracker.log_params({
        "alpha": alpha, "lr": lr, "batch_size": batch_size, "bottleneck_dim": bottleneck_dim,
        "channels": str(channels), "use_residual": use_residual, "use_skips": use_skips,
        "dropout": dropout, "parameters": model.count_parameters(),
        "train_samples": len(train_loader.dataset), "val_samples": len(val_loader.dataset),
    })

    best_val_loss, best_metrics, stagnant_epochs = float("inf"), {}, 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "val_mae": []}

    try:
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.torch_device, epoch, epochs)
            val_metrics, s_corr, s_rest, s_clean = validate(model, val_loader, criterion, cfg.torch_device)
            scheduler.step()

            cur_vloss = val_metrics["val_loss"]
            for k in ["train_loss", "val_loss", "val_psnr", "val_ssim", "val_mae"]:
                history[k].append(train_loss if k == "train_loss" else val_metrics[k.replace("val_", "")] if k != "val_loss" else cur_vloss)

            tracker.log_metrics({
                "train_loss": train_loss, "val_loss": cur_vloss, "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"], "val_mae": val_metrics["mae"],
                "clean_ssim": val_metrics.get("clean_ssim", 0.0), "sp_ssim": val_metrics.get("sp_ssim", 0.0),
                "blur_ssim": val_metrics.get("blur_ssim", 0.0), "occl_ssim": val_metrics.get("occl_ssim", 0.0),
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            print(
                f"==> Epoch {epoch:02d}/{epochs:02d} Summary | Train Loss: {train_loss:.4f} | Val Loss: {cur_vloss:.4f} "
                f"| PSNR: {val_metrics['psnr']:.2f}dB | SSIM: {val_metrics['ssim']:.4f} "
                f"| Clean: {val_metrics.get('clean_ssim', 0):.3f} | S&P: {val_metrics.get('sp_ssim', 0):.3f} "
                f"| Blur: {val_metrics.get('blur_ssim', 0):.3f} | Occ: {val_metrics.get('occl_ssim', 0):.3f}",
                flush=True,
            )

            if cur_vloss < best_val_loss:
                best_val_loss, best_metrics, stagnant_epochs = cur_vloss, val_metrics, 0
                model.save_checkpoint(ckpt_out, epoch=epoch, val_loss=best_val_loss, extra={"metrics": best_metrics, "alpha": alpha})
            else:
                stagnant_epochs += 1

            if epoch % sample_interval == 0 or epoch == epochs:
                grid = make_reconstruction_grid(s_corr, s_rest, s_clean, n_samples=8)
                tracker.log_image("reconstructions", grid, step=epoch)

            if stagnant_epochs >= patience:
                print(f"Early stopping triggered after {epoch} epochs.", flush=True)
                break

        # Export training curves and history explicitly to disk
        vis_dir, met_dir = Path("results/task1/visualizations"), Path("results/task1/metrics")
        vis_dir.mkdir(parents=True, exist_ok=True)
        met_dir.mkdir(parents=True, exist_ok=True)

        history_path = met_dir / f"{run_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({"best_val_loss": best_val_loss, "best_metrics": best_metrics, "history": history}, f, indent=2)

        curves_fig = plot_training_curves(history, title=f"Universal Autoencoder — {run_name.title()} Run")
        curves_path = vis_dir / f"{run_name}_training_curves.png"
        curves_fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(curves_fig)

        tracker.log_artifact(str(history_path), artifact_path="metrics")
        tracker.log_artifact(str(curves_path), artifact_path="visualizations")

    finally:
        tracker.end_run()

    print(f"Training completed. Best Val Loss: {best_val_loss:.4f} saved to {ckpt_out}", flush=True)
    return {"best_val_loss": best_val_loss, "best_metrics": best_metrics, "checkpoint": str(ckpt_out)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 1 Universal Autoencoder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick smoke run")
    parser.add_argument("--no-cache", action="store_true", help="Disable in-memory caching")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        bottleneck_dim=args.bottleneck_dim,
        patience=args.patience,
        run_name=args.run_name,
        max_train_samples=args.max_train_samples,
        quick=args.quick,
        cache=not args.no_cache,
    )
