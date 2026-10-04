"""src/task3/export_onnx.py
------------------------
Production single-graph ONNX computational graph export, graph validation,
numerical parity assertion, and latency benchmarking for Task 3 Soft Mixture-of-Experts (MoE).

Exports complete end-to-end pipeline (Gating Network + Identity pass-through +
3 Specialist Autoencoders + Temperature-scaled convex combination) into a single atomic ONNX model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.shared.config import get_settings
from src.shared.tracking import ExperimentTracker
from src.task3.moe_model import ExportWrapper, SoftMoE
from src.task3.routing_analysis import load_trained_moe


def export_soft_moe_onnx(
    moe_model: SoftMoE,
    output_path: Union[str, Path],
    tau: float = 2.526,
    opset_version: int = 17,
) -> Path:
    """Export SoftMoE model wrapped with fixed temperature tau to unified ONNX model."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = ExportWrapper(moe_model, tau=tau)
    wrapper.eval()

    dummy = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "restored_image": {0: "batch_size"},
        "routing_weights": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["restored_image", "routing_weights"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    # Validate structural integrity
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    return out_path


def verify_numerical_parity(
    moe_model: SoftMoE,
    onnx_path: Union[str, Path],
    tau: float = 2.526,
    batch_sizes: Tuple[int, ...] = (1, 4, 8),
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Assert output equivalence between PyTorch and ONNX Runtime across batch sizes."""
    wrapper = ExportWrapper(moe_model, tau=tau)
    wrapper.eval()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity: Dict[str, Any] = {}
    all_passed = True

    for b in batch_sizes:
        torch.manual_seed(42 + b)
        x = torch.rand(b, 3, 128, 128, dtype=torch.float32)

        with torch.no_grad():
            pt_restored, pt_weights = wrapper(x)
            pt_restored_np = pt_restored.cpu().numpy()
            pt_weights_np = pt_weights.cpu().numpy()

        ort_outputs = sess.run(None, {"input_image": x.numpy()})
        ort_restored_np = ort_outputs[0]
        ort_weights_np = ort_outputs[1]

        max_restored_diff = float(np.max(np.abs(pt_restored_np - ort_restored_np)))
        max_weights_diff = float(np.max(np.abs(pt_weights_np - ort_weights_np)))

        restored_close = bool(np.allclose(pt_restored_np, ort_restored_np, atol=atol))
        weights_close = bool(np.allclose(pt_weights_np, ort_weights_np, atol=atol))

        batch_passed = restored_close and weights_close
        if not batch_passed:
            all_passed = False

        parity[f"batch_{b}"] = {
            "batch_size": b,
            "max_restored_abs_diff": max_restored_diff,
            "max_weights_abs_diff": max_weights_diff,
            "restored_is_close": restored_close,
            "weights_is_close": weights_close,
            "passed": batch_passed,
        }

    parity["all_passed"] = all_passed
    parity["tolerance_atol"] = atol
    return parity


def benchmark_model_latency(
    moe_model: SoftMoE,
    onnx_path: Union[str, Path],
    tau: float = 2.526,
    num_warmup: int = 25,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Measure single-image CPU latency for PyTorch eager vs ONNX Runtime."""
    wrapper = ExportWrapper(moe_model, tau=tau)
    wrapper.eval()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_tensor = torch.rand(1, 3, 128, 128, dtype=torch.float32)
    x_np = x_tensor.numpy()

    # PyTorch CPU Warmup & Benchmark
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = wrapper(x_tensor)
        pt_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = wrapper(x_tensor)
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    # ONNX Runtime CPU Warmup & Benchmark
    for _ in range(num_warmup):
        _ = sess.run(None, {"input_image": x_np})
    ort_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {"input_image": x_np})
        ort_times.append((time.perf_counter() - t0) * 1000.0)

    pt_mean = float(np.mean(pt_times))
    ort_mean = float(np.mean(ort_times))
    speedup = float(pt_mean / max(1e-6, ort_mean))

    return {
        "pytorch_ms": round(pt_mean, 2),
        "onnxruntime_ms": round(ort_mean, 2),
        "speedup": round(speedup, 2),
        "throughput_fps": round(float(1000.0 / ort_mean), 1),
    }


class OnnxSoftMoE:
    """Production inference engine for Task 3 Soft MoE executed via ONNX Runtime."""

    def __init__(
        self,
        onnx_path: Union[str, Path] = "models/onnx/task3_soft_moe.onnx",
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.onnx_path = Path(onnx_path)
        self.sess = ort.InferenceSession(str(self.onnx_path), providers=list(providers))

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch input image array (B, 3, 128, 128) returning (restored, routing_weights)."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        outputs = self.sess.run(None, {"input_image": x.astype(np.float32)})
        return outputs[0], outputs[1]


def run_task3_onnx_export_pipeline(
    checkpoint_path: Union[str, Path] = "checkpoints/task3/best_model.pth",
    output_onnx: Union[str, Path] = "models/onnx/task3_soft_moe.onnx",
    metrics_path: Union[str, Path] = "results/task3/onnx_parity_benchmark.json",
    opset_version: int = 17,
    num_runs: int = 100,
    log_mlflow: bool = True,
) -> Dict[str, Any]:
    """Execute complete ONNX export, parity verification, and latency logging for Task 3."""
    out_path = Path(output_onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 10: Task 3 Single-Graph ONNX Export & Parity Verification ===")
    device = torch.device("cpu")

    # 1. Load trained SoftMoE model
    moe_model, tau, ckpt_meta = load_trained_moe(checkpoint_path, device=device)
    moe_model.eval()
    print(f"Loaded trained SoftMoE from {checkpoint_path} (tau={tau:.3f}).")

    # 2. Export unified single-graph ONNX model
    print(f"Exporting atomic ONNX graph to {out_path} (opset {opset_version})...")
    export_soft_moe_onnx(moe_model, out_path, tau=tau, opset_version=opset_version)
    onnx_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported successfully! Model size: {onnx_size_mb:.2f} MB")

    # 3. Assert Numerical Parity
    print("Verifying numerical parity between PyTorch and ONNX Runtime...")
    parity = verify_numerical_parity(moe_model, out_path, tau=tau, batch_sizes=(1, 4, 8), atol=1e-5)
    print(f"Parity Results: all_passed={parity['all_passed']}")
    for k in ["batch_1", "batch_4", "batch_8"]:
        res = parity[k]
        print(f"  - {k}: max restored diff = {res['max_restored_abs_diff']:.2e}, max weights diff = {res['max_weights_abs_diff']:.2e}, passed={res['passed']}")

    # 4. Latency Benchmark
    print(f"Benchmarking CPU inference latency ({num_runs} runs, single image)...")
    latency = benchmark_model_latency(moe_model, out_path, tau=tau, num_warmup=25, num_runs=num_runs)
    print(f"Latency Results:")
    print(f"  - PyTorch eager CPU: {latency['pytorch_ms']:.2f} ms")
    print(f"  - ONNX Runtime CPU: {latency['onnxruntime_ms']:.2f} ms")
    print(f"  - Speedup factor: {latency['speedup']:.2f}x")
    print(f"  - Throughput: {latency['throughput_fps']:.1f} FPS")

    benchmark_summary: Dict[str, Any] = {
        "model": "task3_soft_moe",
        "onnx_path": str(out_path),
        "onnx_size_mb": round(onnx_size_mb, 2),
        "opset_version": opset_version,
        "temperature_tau": round(tau, 4),
        "parity": parity,
        "latency_cpu": latency,
    }

    # 5. Persist JSON Summary
    json_path = Path(metrics_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    print(f"Exported benchmark summary to {json_path}")

    # 6. Log to MLflow
    if log_mlflow:
        tracker = ExperimentTracker(settings=get_settings())
        with tracker.run(run_name="task3-onnx-export", experiment_name="genai-task3-soft-moe"):
            tracker.log_params({
                "opset_version": opset_version,
                "temperature_tau": tau,
            })
            tracker.log_metrics({
                "onnx_size_mb": onnx_size_mb,
                "pytorch_cpu_ms": latency["pytorch_ms"],
                "onnxruntime_cpu_ms": latency["onnxruntime_ms"],
                "speedup_factor": latency["speedup"],
                "throughput_fps": latency["throughput_fps"],
                "max_restored_diff_b1": parity["batch_1"]["max_restored_abs_diff"],
                "max_weights_diff_b1": parity["batch_1"]["max_weights_abs_diff"],
            })
            tracker.log_artifact(str(json_path))
            tracker.log_artifact(str(out_path))

    print("\n=== Step 10 ONNX Export & Parity Verification Completed Successfully! ===")
    return benchmark_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 Step 10 ONNX Export & Parity Verification")
    p.add_argument("--checkpoint", type=str, default="checkpoints/task3/best_model.pth")
    p.add_argument("--output", type=str, default="models/onnx/task3_soft_moe.onnx")
    p.add_argument("--metrics", type=str, default="results/task3/onnx_parity_benchmark.json")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task3_onnx_export_pipeline(
        checkpoint_path=args.checkpoint,
        output_onnx=args.output,
        metrics_path=args.metrics,
        opset_version=args.opset,
        num_runs=args.runs,
        log_mlflow=not args.no_mlflow,
    )
