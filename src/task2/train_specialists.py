"""Training pipeline orchestrator for Task 2 Specialist Autoencoders.

Trains dedicated specialist autoencoders for Salt-and-Pepper, Gaussian Blur,
and Rectangular Occlusion with independent checkpointing and MLflow tracking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.config import get_settings
from src.shared.datasets.pets import PetDataset
from src.shared.losses import CombinedReconstructionLoss
from src.shared.metrics import compute_mae, compute_psnr, compute_ssim
from src.shared.tracking import ExperimentTracker
from src.task2.specialist import (
    SpecialistAutoencoder,
    VALID_SPECIALISTS,
    build_specialist,
)
from src.task2.specialist_dataset import build_specialist_dataloaders
from src.task2.visualization import (
    plot_specialists_curves,
    plot_specialists_reconstructions,
)

CKPT_NAME_MAP = {
    "salt_and_pepper": "specialist_salt_best.pt",
    "gaussian_blur": "specialist_blur_best.pt",
    "occlusion": "specialist_occlusion_best.pt",
}


def evaluate_specialist(
    model: SpecialistAutoencoder,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Compute validation loss, PSNR, SSIM, and MAE across deterministic validation set."""
    model.eval()
    total_loss, total_psnr, total_ssim, total_mae, total_count = 0.0, 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for corrupted, clean in dataloader:
            corrupted, clean = corrupted.to(device), clean.to(device)
            restored = model(corrupted)
            loss = criterion(restored, clean)

            bs = clean.size(0)
            total_loss += loss.item() * bs
            for i in range(bs):
                total_psnr += compute_psnr(restored[i], clean[i])
                total_ssim += compute_ssim(restored[i], clean[i])
                total_mae += compute_mae(restored[i], clean[i])
            total_count += bs

    return {
        "val_loss": round(total_loss / max(1, total_count), 4),
        "val_psnr": round(total_psnr / max(1, total_count), 2),
        "val_ssim": round(total_ssim / max(1, total_count), 4),
        "val_mae": round(total_mae / max(1, total_count), 4),
    }


def train_single_specialist(
    corruption_type: str,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    alpha: float = 0.84,
    variant: str = "homogeneous",
    clean_train_tensors: Optional[List[torch.Tensor]] = None,
    device: torch.device = torch.device("cpu"),
    tracker: Optional[ExperimentTracker] = None,
) -> Tuple[SpecialistAutoencoder, Dict[str, Any]]:
    """Train an isolated specialist autoencoder to convergence on single corruption distribution."""
    print("\n" + "=" * 75)
    print(f"TRAINING SPECIALIST: {corruption_type.upper()} ({variant})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr} | Alpha: {alpha}")
    print("=" * 75)

    train_loader, val_loader = build_specialist_dataloaders(
        corruption_type=corruption_type,
        batch_size=batch_size,
        clean_train_tensors=clean_train_tensors,
    )

    model = build_specialist(corruption_type, variant=variant).to(device)
    criterion = CombinedReconstructionLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path("checkpoints/task2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / CKPT_NAME_MAP[corruption_type]

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_mae": [],
    }

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        running_train_loss, num_batches = 0.0, len(train_loader)

        for batch_idx, (corrupted, clean) in enumerate(train_loader, 1):
            corrupted, clean = corrupted.to(device), clean.to(device)
            optimizer.zero_grad()
            restored = model(corrupted)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * clean.size(0)

            if batch_idx % 25 == 0 or batch_idx == num_batches:
                speed = clean.size(0) / max(0.001, (time.time() - start_time) / batch_idx)
                print(
                    f"  [Epoch {epoch:02d}/{epochs:02d} | Batch {batch_idx:02d}/{num_batches:02d} "
                    f"({100 * batch_idx / num_batches:2.0f}%)] Loss: {loss.item():.4f} | Speed: {speed:.1f} img/s",
                    flush=True,
                )

        scheduler.step()
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        val_res = evaluate_specialist(model, val_loader, criterion, device)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(val_res["val_loss"])
        history["val_psnr"].append(val_res["val_psnr"])
        history["val_ssim"].append(val_res["val_ssim"])
        history["val_mae"].append(val_res["val_mae"])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} Finished ({time.time() - start_time:.1f}s) -> "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | "
            f"Val PSNR: {val_res['val_psnr']:.2f} dB | Val SSIM: {val_res['val_ssim']:.4f}"
        )

        if val_res["val_loss"] < best_val_loss:
            best_val_loss = val_res["val_loss"]
            best_metrics = val_res
            model.save_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                val_loss=best_val_loss,
                metrics=best_metrics,
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

        if tracker:
            tracker.log_metrics({f"{corruption_type}_{k}": v for k, v in val_res.items()}, step=epoch)

    return model, {
        "corruption_type": corruption_type,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "checkpoint_path": str(best_ckpt_path),
        "history": history,
    }


def main() -> None:
    """Orchestrate multi-specialist training, evaluation, and artifact generation."""
    parser = argparse.ArgumentParser(description="Train Task 2 Specialist Autoencoders.")
    parser.add_argument(
        "--specialist",
        choices=["all", "salt_and_pepper", "gaussian_blur", "occlusion"],
        default="all",
        help="Specialist to train.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per specialist.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--alpha", type=float, default=0.84, help="L1 vs SSIM loss weight.")
    parser.add_argument("--variant", default="homogeneous", help="Architectural variant.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max clean images for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} | Variant: {args.variant} | Target: {args.specialist}")

    specialists_to_train = (
        list(VALID_SPECIALISTS) if args.specialist == "all" else [args.specialist]
    )

    # Pre-cache clean base training tensors in memory once
    print("Pre-loading Oxford Pets training split into memory...")
    settings = get_settings()
    base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=settings)
    clean_train_tensors = [base_train[i] for i in range(len(base_train))]
    if args.max_train_samples is not None:
        clean_train_tensors = clean_train_tensors[: args.max_train_samples]
    print(f"Pre-loaded {len(clean_train_tensors)} clean images.")

    tracker = ExperimentTracker(settings=settings)
    tracker.start_run(run_name="baseline_specialists", experiment_name="task2-specialists")
    tracker.log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "alpha": args.alpha,
        "variant": args.variant,
        "specialists": specialists_to_train,
    })

    all_histories: Dict[str, Dict[str, List[float]]] = {}
    summary_results: Dict[str, Any] = {}
    sample_reconstructions: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for corr in specialists_to_train:
        model, results = train_single_specialist(
            corruption_type=corr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=args.alpha,
            variant=args.variant,
            clean_train_tensors=clean_train_tensors,
            device=device,
            tracker=tracker,
        )
        all_histories[corr] = results["history"]
        summary_results[corr] = results

        # Grab a sample validation pair for qualitative visual inspection
        _, val_loader = build_specialist_dataloaders(corr, batch_size=4)
        sample_corr, sample_clean = next(iter(val_loader))
        model.eval()
        with torch.no_grad():
            sample_restored = model(sample_corr[:1].to(device)).cpu()
        sample_reconstructions[corr] = (sample_corr[0], sample_restored[0], sample_clean[0])

    # Save summary metrics JSON
    results_dir = Path("results/task2")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "specialists_baseline_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    # Save visual training curves and sample reconstructions
    curves_path = results_dir / "specialists_baseline_curves.png"
    recons_path = results_dir / "specialists_sample_reconstructions.png"
    plot_specialists_curves(all_histories, curves_path)
    plot_specialists_reconstructions(sample_reconstructions, recons_path)

    tracker.log_artifact(str(json_path))
    tracker.log_artifact(str(curves_path))
    tracker.log_artifact(str(recons_path))
    tracker.end_run()

    print("\n" + "=" * 80)
    print("ALL SPECIALISTS TRAINING COMPLETE")
    print(f"Metrics saved to {json_path}")
    print(f"Curves saved to {curves_path}")
    print(f"Reconstructions saved to {recons_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
