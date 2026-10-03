"""
src/task1/export_onnx.py
------------------------
Production ONNX computational graph export, structure validation,
numerical parity assertion, and latency benchmarking for Task 1 Universal Autoencoder.
Prepares canonical artifact for FastAPI Workspace 1 (/api/v1/restore/universal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.shared.tracking import ExperimentTracker
from src.task1.autoencoder import UniversalAutoencoder


def export_model_to_onnx(
    model: UniversalAutoencoder,
    output_path: Path,
    opset_version: int = 17,
    input_shape: tuple[int, ...] = (1, 3, 128, 128),
) -> Path:
    """Export PyTorch UniversalAutoencoder to ONNX graph with dynamic batch axis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    print(f"Exporting ONNX model to {output_path} (opset {opset_version})...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX graph validated successfully. File size: {file_size_mb:.2f} MB")
    return output_path


def verify_numerical_parity(
    model: UniversalAutoencoder,
    onnx_path: Path,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Assert output equivalence between PyTorch and ONNX Runtime across batch sizes."""
    model.eval()
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity_results: dict[str, Any] = {}
    all_passed = True

    print("\nVerifying numerical parity between PyTorch and ONNX Runtime:")
    for b in batch_sizes:
        torch.manual_seed(42 + b)
        x = torch.randn(b, 3, 128, 128, dtype=torch.float32)

        with torch.no_grad():
            y_torch = model(x).cpu().numpy()

        y_onnx = ort_session.run(None, {"input": x.numpy()})[0]

        max_abs_diff = float(np.max(np.abs(y_torch - y_onnx)))
        mean_abs_diff = float(np.mean(np.abs(y_torch - y_onnx)))
        is_close = bool(np.allclose(y_torch, y_onnx, atol=atol))
        if not is_close:
            all_passed = False

        status_str = "PASS" if is_close else "FAIL"
        print(f"  Batch {b}: [{status_str}] max_diff = {max_abs_diff:.3e}, mean_diff = {mean_abs_diff:.3e}")
        parity_results[f"batch_{b}"] = {
            "batch_size": b,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "is_close": is_close,
        }

    parity_results["all_passed"] = all_passed
    parity_results["tolerance_atol"] = atol
    return parity_results


def benchmark_inference_latency(
    model: UniversalAutoencoder,
    onnx_path: Path,
    num_warmup: int = 100,
    num_runs: int = 200,
) -> dict[str, Any]:
    """Measure single-image inference latency (ms) and throughput (img/s) on CPU."""
    model.eval()
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_tensor = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    x_numpy = x_tensor.numpy()

    print(f"\nBenchmarking inference latency ({num_warmup} warm-up, {num_runs} timed runs):")
    # 1. PyTorch CPU benchmark
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x_tensor)

        torch_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(x_tensor)
            torch_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. ONNX Runtime CPU benchmark
    for _ in range(num_warmup):
        _ = ort_session.run(None, {"input": x_numpy})

    ort_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {"input": x_numpy})
        ort_times.append((time.perf_counter() - t0) * 1000.0)

    torch_mean = float(np.mean(torch_times))
    ort_mean = float(np.mean(ort_times))
    speedup = float(torch_mean / max(1e-6, ort_mean))

    benchmark_stats = {
        "pytorch": {
            "mean_ms": torch_mean,
            "median_ms": float(np.median(torch_times)),
            "p95_ms": float(np.percentile(torch_times, 95)),
            "p99_ms": float(np.percentile(torch_times, 99)),
            "std_ms": float(np.std(torch_times)),
            "throughput_fps": float(1000.0 / torch_mean),
        },
        "onnxruntime": {
            "mean_ms": ort_mean,
            "median_ms": float(np.median(ort_times)),
            "p95_ms": float(np.percentile(ort_times, 95)),
            "p99_ms": float(np.percentile(ort_times, 99)),
            "std_ms": float(np.std(ort_times)),
            "throughput_fps": float(1000.0 / ort_mean),
        },
        "speedup_factor": speedup,
        "num_runs": num_runs,
    }

    print(f"  PyTorch CPU:      {torch_mean:6.2f} ms/image ({benchmark_stats['pytorch']['throughput_fps']:5.1f} img/s)")
    print(f"  ONNX Runtime CPU: {ort_mean:6.2f} ms/image ({benchmark_stats['onnxruntime']['throughput_fps']:5.1f} img/s)")
    print(f"  Speedup Factor:   {speedup:6.2f}x acceleration")
    return benchmark_stats


def run_onnx_export_pipeline(
    checkpoint_path: str = "checkpoints/task1/best_model.pth",
    output_path: str = "models/onnx/task1_universal_ae.onnx",
    metrics_path: str = "results/task1/metrics/onnx_parity_benchmark.json",
    opset_version: int = 17,
    num_warmup: int = 100,
    num_runs: int = 200,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end ONNX export, parity verification, and latency logging."""
    out_file = Path(output_path)
    met_file = Path(metrics_path)
    met_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}...")
    model, chk = UniversalAutoencoder.load_from_checkpoint(checkpoint_path, device="cpu")

    # 1. Export
    export_model_to_onnx(model, out_file, opset_version=opset_version)

    # 2. Parity check
    parity_stats = verify_numerical_parity(model, out_file)

    # 3. Latency benchmark
    latency_stats = benchmark_inference_latency(model, out_file, num_warmup=num_warmup, num_runs=num_runs)

    combined_results = {
        "export_metadata": {
            "checkpoint": checkpoint_path,
            "onnx_model": str(out_file),
            "opset_version": opset_version,
            "model_parameters": model.count_parameters(),
            "file_size_bytes": out_file.stat().st_size,
        },
        "parity": parity_stats,
        "benchmark": latency_stats,
    }

    with open(met_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nSaved benchmark metrics to {met_file}")

    # 4. MLflow Logging
    if log_mlflow:
        print("Logging ONNX export run to MLflow tracker...")
        tracker = ExperimentTracker()
        with tracker.run(run_name="onnx-verify", experiment_name="task1-universal-ae"):
            tracker.log_param("onnx_opset", opset_version)
            tracker.log_param("onnx_file_size_mb", out_file.stat().st_size / (1024 * 1024))
            tracker.log_metric("parity_all_passed", 1.0 if parity_stats["all_passed"] else 0.0)
            tracker.log_metric("parity_max_abs_diff", parity_stats["batch_1"]["max_abs_diff"])
            tracker.log_metric("pytorch_latency_ms", latency_stats["pytorch"]["mean_ms"])
            tracker.log_metric("onnx_latency_ms", latency_stats["onnxruntime"]["mean_ms"])
            tracker.log_metric("onnx_speedup_factor", latency_stats["speedup_factor"])
            tracker.log_artifact(str(met_file))
            print("Successfully logged ONNX export and parity artifacts to MLflow.")

    return combined_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify Task 1 Universal Autoencoder to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/task1/best_model.pth")
    parser.add_argument("--output", type=str, default="models/onnx/task1_universal_ae.onnx")
    parser.add_argument("--metrics", type=str, default="results/task1/metrics/onnx_parity_benchmark.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    run_onnx_export_pipeline(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        metrics_path=args.metrics,
        opset_version=args.opset,
        num_warmup=args.warmup,
        num_runs=args.runs,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
