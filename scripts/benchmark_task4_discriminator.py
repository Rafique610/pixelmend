"""
scripts/benchmark_task4_discriminator.py
----------------------------------------
Empirical GPU benchmark for Task 4 Step 3:
Evaluates Discriminator Variants:
1. 70x70 PatchGAN (pix2pix default, 5 conv stages, receptive field ~70x70)
2. 16x16 PatchGAN (shallower downsampling, receptive field ~16x16)
3. Multi-Scale PatchGAN (dual-scale on 128x128 and 64x64)

Benchmarks parameters, GPU latency, peak VRAM, adversarial discrimination,
and gradient feedback on real FS2K pairs on NVIDIA RTX 3050 GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.config import get_settings
from src.task4.dataset import FS2KDataset
from src.task4.discriminator import MultiScaleDiscriminator, PatchGANDiscriminator
from src.task4.generator import UNetGenerator


def benchmark_discriminator(
    d_model: nn.Module,
    g_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    is_multiscale: bool = False,
    num_steps: int = 35,
) -> dict:
    d_model.train()
    g_model.eval()

    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    timings = []
    real_losses = []
    fake_losses = []
    g_grad_norms = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step = 0
    for batch in data_loader:
        if step >= num_steps:
            break
        photo = batch["photo"].to(device)
        sketch_real = batch["sketch"].to(device)
        style = batch["style"].to(device)

        with torch.no_grad():
            sketch_fake = g_model(photo, style)

        t0 = time.perf_counter()
        optimizer_d.zero_grad()

        if is_multiscale:
            d_real1, d_real2 = d_model(photo, sketch_real, style)
            loss_d_real = 0.5 * (
                criterion(d_real1, torch.ones_like(d_real1))
                + criterion(d_real2, torch.ones_like(d_real2))
            )

            d_fake1, d_fake2 = d_model(photo, sketch_fake.detach(), style)
            loss_d_fake = 0.5 * (
                criterion(d_fake1, torch.zeros_like(d_fake1))
                + criterion(d_fake2, torch.zeros_like(d_fake2))
            )
        else:
            d_real = d_model(photo, sketch_real, style)
            loss_d_real = criterion(d_real, torch.ones_like(d_real))

            d_fake = d_model(photo, sketch_fake.detach(), style)
            loss_d_fake = criterion(d_fake, torch.zeros_like(d_fake))

        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        loss_d.backward()
        optimizer_d.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

        real_losses.append(loss_d_real.item())
        fake_losses.append(loss_d_fake.item())

        # Test gradient backpropagation into Generator from D
        sketch_fake_grad = g_model(photo, style)
        if is_multiscale:
            d_fake_g1, d_fake_g2 = d_model(photo, sketch_fake_grad, style)
            adv_loss = 0.5 * (
                criterion(d_fake_g1, torch.ones_like(d_fake_g1))
                + criterion(d_fake_g2, torch.ones_like(d_fake_g2))
            )
        else:
            d_fake_g = d_model(photo, sketch_fake_grad, style)
            adv_loss = criterion(d_fake_g, torch.ones_like(d_fake_g))

        g_model.zero_grad()
        adv_loss.backward()
        g_norm = 0.0
        for p in g_model.parameters():
            if p.grad is not None:
                g_norm += p.grad.data.norm(2).item() ** 2
        g_grad_norms.append(g_norm ** 0.5)

        step += 1

    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )

    n_params = sum(p.numel() for p in d_model.parameters() if p.requires_grad)

    return {
        "parameters": n_params,
        "params_millions": round(n_params / 1e6, 2),
        "mean_latency_ms": round(float(np.mean(timings[5:])), 3),
        "p95_latency_ms": round(float(np.percentile(timings[5:], 95)), 3),
        "peak_vram_mb": round(float(peak_vram), 2),
        "d_real_loss": round(float(np.mean(real_losses[-10:])), 4),
        "d_fake_loss": round(float(np.mean(fake_losses[-10:])), 4),
        "d_total_loss": round(float(0.5 * (np.mean(real_losses[-10:]) + np.mean(fake_losses[-10:]))), 4),
        "generator_grad_norm": round(float(np.mean(g_grad_norms[-10:])), 4),
    }


def main() -> None:
    settings = get_settings()
    device = settings.torch_device
    print(f"=== Task 4 Discriminator Benchmark on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    train_ds = FS2KDataset(root_dir="data/fs2k", split="train", augment=True, seed=42)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

    # Reference Generator
    g_model = UNetGenerator(base_channels=64, embed_dim=16).to(device)

    # Candidates
    candidates = {
        "70x70 PatchGAN (pix2pix default)": (
            PatchGANDiscriminator(in_channels=3, sketch_channels=3, embed_dim=16, base_channels=64, n_layers=3).to(device),
            False,
            "14x14",
            "~70x70",
        ),
        "16x16 PatchGAN (shallower RF)": (
            PatchGANDiscriminator(in_channels=3, sketch_channels=3, embed_dim=16, base_channels=64, n_layers=1).to(device),
            False,
            "31x31",
            "~16x16",
        ),
        "Multi-Scale PatchGAN (pix2pixHD)": (
            MultiScaleDiscriminator(in_channels=3, sketch_channels=3, embed_dim=16, base_channels=64).to(device),
            True,
            "14x14 + 7x7",
            "Dual (70x70 + coarse)",
        ),
    }

    results = {}
    for name, (d_model, is_multi, grid_shape, rf_size) in candidates.items():
        print(f"\nBenchmarking {name}...")
        metrics = benchmark_discriminator(
            d_model, g_model, train_loader, device, is_multiscale=is_multi, num_steps=30
        )
        metrics["output_grid"] = grid_shape
        metrics["receptive_field"] = rf_size
        results[name] = metrics
        print(f" -> Params: {metrics['parameters']:,} | Latency: {metrics['mean_latency_ms']} ms | VRAM: {metrics['peak_vram_mb']} MB | D Total Loss: {metrics['d_total_loss']} | G Grad: {metrics['generator_grad_norm']}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
        },
        "candidates": results,
    }

    out_file = Path("results/task4/discriminator_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved empirical discriminator benchmark to {out_file}")

    try:
        mlflow.set_experiment("genai-task4-research")
        with mlflow.start_run(run_name="discriminator-architecture-benchmark"):
            mlflow.log_params({"dataset": "FS2K", "batch_size": 8})
            for c_name, c_metrics in results.items():
                c_clean = c_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                for k, v in c_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{c_clean}_{k}", v)
            mlflow.log_artifact(str(out_file))
            print("Successfully logged discriminator benchmark to MLflow.")
    except Exception as e:
        print(f"MLflow warning: {e}")


if __name__ == "__main__":
    main()
