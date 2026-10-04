"""
scripts/benchmark_task4_research.py
-----------------------------------
Rigorous GPU-accelerated benchmarking for Task 4 Steps 1 & 2:
1. Generator Architecture Research (Vanilla U-Net vs ResNet U-Net vs Attention U-Net)
2. Style Conditioning Research (Spatial Concat vs FiLM vs AdaIN, across embedding dims)

Runs on real FS2K pairs on NVIDIA GeForce RTX 3050.
Persists tangible metrics to results/task4/architecture_conditioning_benchmark.json
and logs to MLflow run 'genai-task4-research'.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.config import get_settings
from src.task4.dataset import FS2KDataset
from src.task4.generator_variants import (
    AttentionUNetGenerator,
    ResNetUNetGenerator,
    VanillaUNetGenerator,
)


def measure_latency_and_memory(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    num_runs: int = 30,
    warmup: int = 10,
) -> dict:
    """Measure forward latency (ms) and peak VRAM allocation (MB)."""
    model.eval()
    x = torch.randn(batch_size, 3, 128, 128, device=device)
    s = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, s)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                _ = model(x, s)
                end_event.record()
                torch.cuda.synchronize(device)
                timings.append(start_event.elapsed_time(end_event))

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        mean_latency = float(np.mean(timings))
        p95_latency = float(np.percentile(timings, 95))
    else:
        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(x, s)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)
        peak_vram_mb = 0.0
        mean_latency = float(np.mean(timings))
        p95_latency = float(np.percentile(timings, 95))

    return {
        "mean_latency_ms": round(mean_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def evaluate_short_convergence(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_steps: int = 40,
) -> dict:
    """Train for a small fixed step budget on real FS2K pairs to measure gradient stability and convergence."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.L1Loss()

    grad_norms = []
    train_losses = []

    step = 0
    start_time = time.time()
    for batch in train_loader:
        if step >= num_steps:
            break
        photo = batch["photo"].to(device)
        sketch = batch["sketch"].to(device)
        style = batch["style"].to(device)

        optimizer.zero_grad()
        pred = model(photo, style)
        loss = criterion(pred, sketch)
        loss.backward()

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        optimizer.step()
        train_losses.append(loss.item())
        step += 1

    train_duration = time.time() - start_time

    # Validate on fixed validation batch
    model.eval()
    val_l1_losses = []
    style_distances = []
    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            if i >= 10:
                break
            v_photo = val_batch["photo"].to(device)
            v_sketch = val_batch["sketch"].to(device)
            v_style = val_batch["style"].to(device)

            pred_val = model(v_photo, v_style)
            val_l1_losses.append(criterion(pred_val, v_sketch).item())

            # Style sensitivity: difference between style 0 and style 1 synthesis
            s0 = torch.zeros(v_photo.size(0), dtype=torch.long, device=device)
            s1 = torch.ones(v_photo.size(0), dtype=torch.long, device=device)
            out_s0 = model(v_photo, s0)
            out_s1 = model(v_photo, s1)
            dist = torch.mean(torch.abs(out_s0 - out_s1)).item()
            style_distances.append(dist)

    return {
        "final_train_l1": round(float(np.mean(train_losses[-10:])), 4),
        "val_l1": round(float(np.mean(val_l1_losses)), 4),
        "mean_grad_norm": round(float(np.mean(grad_norms)), 4),
        "style_diff_distance": round(float(np.mean(style_distances)), 4),
        "train_time_sec": round(train_duration, 2),
    }


def check_onnx_export(model: nn.Module, device: torch.device) -> bool:
    """Test ONNX graph exportability at opset 17."""
    model.eval()
    dummy_x = torch.randn(1, 3, 128, 128, device=device)
    dummy_s = torch.zeros(1, dtype=torch.long, device=device)
    buf = io.BytesIO()
    try:
        torch.onnx.export(
            model,
            (dummy_x, dummy_s),
            buf,
            opset_version=17,
            input_names=["photo", "style_idx"],
            output_names=["sketch"],
            dynamic_axes={"photo": {0: "batch"}, "style_idx": {0: "batch"}, "sketch": {0: "batch"}},
        )
        return True
    except Exception as e:
        print(f"ONNX export check failed: {e}")
        return False


def main() -> None:
    settings = get_settings()
    device = settings.torch_device
    print(f"=== Task 4 Benchmark Execution on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    train_ds = FS2KDataset(root_dir="data/fs2k", split="train", augment=True, seed=42)
    val_ds = FS2KDataset(root_dir="data/fs2k", split="val", augment=False, seed=42)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # 1. ARCHITECTURE CANDIDATES (Step 1)
    arch_candidates = {
        "Vanilla U-Net (pix2pix)": lambda: VanillaUNetGenerator(conditioning="film", embed_dim=16),
        "ResNet U-Net": lambda: ResNetUNetGenerator(conditioning="film", embed_dim=16),
        "Attention U-Net": lambda: AttentionUNetGenerator(conditioning="film", embed_dim=16),
    }

    arch_results = {}
    for name, builder in arch_candidates.items():
        print(f"\n[Benchmarking Architecture] {name}...")
        model = builder().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        gpu_lat_b1 = measure_latency_and_memory(model, device, batch_size=1)
        gpu_lat_b8 = measure_latency_and_memory(model, device, batch_size=8)
        
        # CPU Latency
        model_cpu = builder().to("cpu")
        cpu_lat_b1 = measure_latency_and_memory(model_cpu, torch.device("cpu"), batch_size=1, num_runs=10)

        # Convergence & Stability on real FS2K pairs
        conv = evaluate_short_convergence(model, train_loader, val_loader, device, num_steps=35)
        onnx_ok = check_onnx_export(model, device)

        arch_results[name] = {
            "parameters": n_params,
            "params_millions": round(n_params / 1e6, 2),
            "gpu_latency_b1_ms": gpu_lat_b1["mean_latency_ms"],
            "gpu_latency_b8_ms": gpu_lat_b8["mean_latency_ms"],
            "gpu_peak_vram_mb": gpu_lat_b8["peak_vram_mb"],
            "cpu_latency_b1_ms": cpu_lat_b1["mean_latency_ms"],
            "val_l1_reconstruction": conv["val_l1"],
            "mean_grad_norm": conv["mean_grad_norm"],
            "style_diff_distance": conv["style_diff_distance"],
            "onnx_exportable": onnx_ok,
        }
        print(f" -> Params: {n_params:,} | GPU (B=1): {gpu_lat_b1['mean_latency_ms']} ms | Val L1: {conv['val_l1']} | Grad Norm: {conv['mean_grad_norm']}")

    # 2. STYLE CONDITIONING CANDIDATES (Step 2)
    conditioning_candidates = {
        "Spatial Concatenation": lambda: VanillaUNetGenerator(conditioning="spatial", embed_dim=16),
        "FiLM (Feature Modulation)": lambda: VanillaUNetGenerator(conditioning="film", embed_dim=16),
        "AdaIN (Adaptive Norm)": lambda: VanillaUNetGenerator(conditioning="adain", embed_dim=16),
    }

    cond_results = {}
    for name, builder in conditioning_candidates.items():
        print(f"\n[Benchmarking Conditioning] {name}...")
        model = builder().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        gpu_lat = measure_latency_and_memory(model, device, batch_size=8)
        conv = evaluate_short_convergence(model, train_loader, val_loader, device, num_steps=35)
        onnx_ok = check_onnx_export(model, device)

        cond_results[name] = {
            "parameters": n_params,
            "params_millions": round(n_params / 1e6, 2),
            "gpu_latency_b8_ms": gpu_lat["mean_latency_ms"],
            "val_l1_reconstruction": conv["val_l1"],
            "style_diff_distance": conv["style_diff_distance"],
            "mean_grad_norm": conv["mean_grad_norm"],
            "onnx_exportable": onnx_ok,
        }
        print(f" -> Style Distance: {conv['style_diff_distance']} | Val L1: {conv['val_l1']} | ONNX: {onnx_ok}")

    # 3. STYLE EMBEDDING DIMENSION SEARCH (Step 2 Open Question)
    dim_candidates = {
        "d_s = 8": lambda: VanillaUNetGenerator(conditioning="film", embed_dim=8),
        "d_s = 16": lambda: VanillaUNetGenerator(conditioning="film", embed_dim=16),
        "d_s = 32": lambda: VanillaUNetGenerator(conditioning="film", embed_dim=32),
    }
    dim_results = {}
    for name, builder in dim_candidates.items():
        print(f"\n[Benchmarking Embedding Dimension] {name}...")
        model = builder().to(device)
        conv = evaluate_short_convergence(model, train_loader, val_loader, device, num_steps=35)
        dim_results[name] = {
            "val_l1": conv["val_l1"],
            "style_diff_distance": conv["style_diff_distance"],
            "mean_grad_norm": conv["mean_grad_norm"],
        }
        print(f" -> Style Distance: {conv['style_diff_distance']} | Val L1: {conv['val_l1']}")

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
        },
        "step1_architectures": arch_results,
        "step2_conditioning": cond_results,
        "step2_embedding_dims": dim_results,
    }

    out_file = Path("results/task4/architecture_conditioning_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"\nSaved empirical benchmark results to {out_file}")

    # Log to MLflow
    try:
        mlflow.set_experiment("genai-task4-research")
        with mlflow.start_run(run_name="arch-and-conditioning-benchmark"):
            mlflow.log_params({
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "val_ratio": 0.15,
                "dataset": "FS2K",
            })
            for arch_name, metrics in arch_results.items():
                clean_name = arch_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{clean_name}_{k}", v)
            mlflow.log_artifact(str(out_file))
            print("Successfully logged benchmark to MLflow (genai-task4-research).")
    except Exception as e:
        print(f"MLflow logging warning: {e}")


if __name__ == "__main__":
    main()
