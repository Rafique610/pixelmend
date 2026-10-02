"""Empirical comparison of L1 vs SSIM loss weight alpha across full [0.0, 1.0] spectrum.

Evaluates alpha in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
for 5 full epochs each on a pre-cached 15% training subset of Oxford Pets
with explicit per-epoch progress logging and fixed validation evaluation.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader

from src.shared.config import get_settings
from src.shared.datasets.corrupted import CorruptedPetDataset
from src.shared.losses import CombinedReconstructionLoss
from src.shared.metrics import evaluate_metrics
from src.task1.autoencoder import UniversalAutoencoder


def load_cached_subsets(
    subset_size: int = 384,
    val_size: int = 64,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Pre-load image tensors into memory to eliminate repeated disk I/O across 11 alpha sweeps."""
    settings = get_settings()
    print(f"Loading {subset_size} training samples and {val_size} validation samples into memory...", flush=True)

    train_ds = CorruptedPetDataset(split="train", settings=settings)
    val_ds = CorruptedPetDataset(split="val", settings=settings)

    torch.manual_seed(settings.seed)
    train_indices = torch.randperm(len(train_ds))[:subset_size].tolist()

    cached_train: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in train_indices:
        corr, clean, _ = train_ds[idx]
        cached_train.append((corr, clean))

    cached_val: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(min(val_size, len(val_ds))):
        corr, clean, _ = val_ds[idx]
        cached_val.append((corr, clean))

    print(f"Pre-caching complete ({len(cached_train)} train, {len(cached_val)} val pairs). Ready for sweeps.\n", flush=True)
    return cached_train, cached_val


def run_full_alpha_spectrum(
    alphas: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    epochs: int = 5,
    batch_size: int = 32,
) -> Dict[float, Dict[str, float]]:
    settings = get_settings()
    device = settings.torch_device
    print(f"Running full alpha spectrum on {device} across {len(alphas)} alpha values for {epochs} epochs each.\n", flush=True)

    cached_train, cached_val = load_cached_subsets(subset_size=384, val_size=64)
    train_loader = DataLoader(cached_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(cached_val, batch_size=batch_size, shuffle=False)

    results: Dict[float, Dict[str, float]] = {}

    for alpha in alphas:
        print(f"==================================================", flush=True)
        print(f"  Testing alpha = {alpha:.1f} (5 Full Epochs)", flush=True)
        print(f"==================================================", flush=True)

        torch.manual_seed(settings.seed)
        model = UniversalAutoencoder(
            in_channels=3,
            out_channels=3,
            channels=(16, 32, 64, 128),
            bottleneck_dim=128,
            use_residual=True,
            use_skips=False,
        ).to(device)

        criterion = CombinedReconstructionLoss(alpha=alpha).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        start_time = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            batch_count = 0
            for corrupted, clean in train_loader:
                corrupted = corrupted.to(device)
                clean = clean.to(device)

                optimizer.zero_grad()
                restored = model(corrupted)
                loss = criterion(restored, clean)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_count += 1

            epoch_train_loss = train_loss / max(1, batch_count)
            print(f"  [alpha={alpha:.1f}] Epoch {epoch}/{epochs} -> Train Loss: {epoch_train_loss:.4f}", flush=True)

        elapsed = time.perf_counter() - start_time

        # Validation evaluation
        model.eval()
        val_losses = []
        psnr_list, ssim_list, mae_list = [], [], []

        with torch.no_grad():
            for corrupted, clean in val_loader:
                corrupted = corrupted.to(device)
                clean = clean.to(device)
                restored = model(corrupted)

                loss = criterion(restored, clean)
                val_losses.append(loss.item())

                metrics = evaluate_metrics(clean, restored)
                psnr_list.append(metrics["psnr"])
                ssim_list.append(metrics["ssim"])
                mae_list.append(metrics["mae"])

        avg_val_loss = float(sum(val_losses) / len(val_losses))
        avg_psnr = float(sum(psnr_list) / len(psnr_list))
        avg_ssim = float(sum(ssim_list) / len(ssim_list))
        avg_mae = float(sum(mae_list) / len(mae_list))

        print(
            f"  --> Completed 5 epochs in {elapsed:.1f}s | "
            f"Val Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.2f} dB | "
            f"SSIM: {avg_ssim:.4f} | MAE: {avg_mae:.4f}\n",
            flush=True,
        )

        results[alpha] = {
            "val_loss": avg_val_loss,
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
            "val_mae": avg_mae,
            "elapsed": elapsed,
        }

    return results


if __name__ == "__main__":
    results = run_full_alpha_spectrum()

    print("\n" + "=" * 80)
    print("FINAL RESULTS: FULL ALPHA SPECTRUM (5 EPOCHS EACH)")
    print("=" * 80)
    print(f"{'Alpha':<8} | {'Val Loss':<10} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'MAE':<10} | {'Time (s)':<8}")
    print("-" * 80)
    for alpha, m in results.items():
        print(
            f"{alpha:<8.1f} | {m['val_loss']:<10.4f} | {m['val_psnr']:<12.2f} | "
            f"{m['val_ssim']:<10.4f} | {m['val_mae']:<10.4f} | {m['elapsed']:<8.1f}"
        )
    print("=" * 80)
