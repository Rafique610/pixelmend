"""
src/task4/export_onnx.py
------------------------
Production ONNX computational graph export, graph structural validation,
numerical parity assertion against native PyTorch, and latency benchmarking
for Task 4: Style-Conditioned Face-to-Sketch Synthesis (UNetGenerator).

Exports model with:
- Dynamic batch dimension on both photo input, style index, and sketch output.
- Fixed opset >= 17 compatibility.
- Embedded style embedding layer.
- Numerical parity verification (max abs diff < 1e-5).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.shared.config import get_settings
from src.task4.generator import UNetGenerator


def export_generator_onnx(
    checkpoint_path: str | Path = "checkpoints/task4/best_generator.pth",
    output_path: str | Path = "models/onnx/task4_generator.onnx",
    opset_version: int = 17,
) -> Path:
    """Export UNetGenerator PyTorch model to ONNX."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize PyTorch generator in eval mode on CPU for portable graph creation
    net_g = UNetGenerator(base_channels=64, embed_dim=16)
    cp = Path(checkpoint_path)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")
    net_g.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
    net_g.eval()

    dummy_photo = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    dummy_style = torch.tensor([0], dtype=torch.long)

    dynamic_axes = {
        "photo": {0: "batch_size"},
        "style_index": {0: "batch_size"},
        "sketch": {0: "batch_size"},
    }

    print(f"Exporting UNetGenerator to {out_file} (opset {opset_version})...")
    torch.onnx.export(
        net_g,
        (dummy_photo, dummy_style),
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["photo", "style_index"],
        output_names=["sketch"],
        dynamic_axes=dynamic_axes,
    )

    # Verify structural validity of ONNX graph
    model_onnx = onnx.load(str(out_file))
    onnx.checker.check_model(model_onnx)
    print("ONNX model structure check passed successfully!")

    return out_file


def verify_numerical_parity(
    onnx_path: str | Path = "models/onnx/task4_generator.onnx",
    checkpoint_path: str | Path = "checkpoints/task4/best_generator.pth",
    batch_sizes: Tuple[int, ...] = (1, 4, 8),
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Verify numerical agreement between PyTorch and ONNX Runtime across batch sizes."""
    net_g = UNetGenerator(base_channels=64, embed_dim=16)
    net_g.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net_g.eval()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    results: Dict[str, Any] = {}
    all_passed = True

    for bs in batch_sizes:
        torch.manual_seed(42 + bs)
        sample_photo = torch.randn(bs, 3, 128, 128, dtype=torch.float32)
        sample_style = torch.randint(0, 3, (bs,), dtype=torch.long)

        # PyTorch reference
        with torch.no_grad():
            pt_out = net_g(sample_photo, sample_style).numpy()

        # ONNX Runtime
        ort_inputs = {
            "photo": sample_photo.numpy(),
            "style_index": sample_style.numpy(),
        }
        ort_out = session.run(["sketch"], ort_inputs)[0]

        max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
        mean_abs_diff = float(np.mean(np.abs(pt_out - ort_out)))
        passed = bool(max_abs_diff < tolerance)

        if not passed:
            all_passed = False

        results[f"batch_{bs}"] = {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "passed": passed,
        }
        print(f"Batch {bs}: Max Diff = {max_abs_diff:.2e}, Mean Diff = {mean_abs_diff:.2e} -> {'PASS' if passed else 'FAIL'}")

    results["all_passed"] = all_passed
    return results


def benchmark_inference_latency(
    onnx_path: str | Path = "models/onnx/task4_generator.onnx",
    checkpoint_path: str | Path = "checkpoints/task4/best_generator.pth",
    n_warmup: int = 10,
    n_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark single-image inference latency on CPU: PyTorch vs ONNX Runtime."""
    # PyTorch CPU
    net_g = UNetGenerator(base_channels=64, embed_dim=16)
    net_g.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net_g.eval()

    # ONNX Runtime CPU
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    x_pt = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    s_pt = torch.tensor([0], dtype=torch.long)
    x_np = x_pt.numpy()
    s_np = s_pt.numpy()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = net_g(x_pt, s_pt)
        _ = session.run(["sketch"], {"photo": x_np, "style_index": s_np})

    # Benchmark PyTorch
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = net_g(x_pt, s_pt)
    pt_latency_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    # Benchmark ONNX Runtime
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = session.run(["sketch"], {"photo": x_np, "style_index": s_np})
    ort_latency_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    speedup = pt_latency_ms / ort_latency_ms if ort_latency_ms > 0 else 1.0

    metrics = {
        "pytorch_cpu_ms": round(pt_latency_ms, 2),
        "onnxruntime_cpu_ms": round(ort_latency_ms, 2),
        "speedup_ratio": round(speedup, 2),
        "fps_ort": round(1000.0 / ort_latency_ms, 1),
    }

    print(f"\nLatency Benchmark (CPU single-image):")
    print(f"  PyTorch:       {metrics['pytorch_cpu_ms']} ms")
    print(f"  ONNX Runtime:  {metrics['onnxruntime_cpu_ms']} ms ({metrics['fps_ort']} FPS)")
    print(f"  Speedup:       {metrics['speedup_ratio']}x")

    return metrics


def run_export_and_verification(
    checkpoint_path: str | Path = "checkpoints/task4/best_generator.pth",
    output_path: str | Path = "models/onnx/task4_generator.onnx",
    results_path: str | Path = "results/task4/onnx_parity_benchmark.json",
) -> Dict[str, Any]:
    """Run full export, parity verification, and latency benchmarking."""
    onnx_file = export_generator_onnx(checkpoint_path=checkpoint_path, output_path=output_path)
    parity_results = verify_numerical_parity(onnx_path=onnx_file, checkpoint_path=checkpoint_path)
    latency_results = benchmark_inference_latency(onnx_path=onnx_file, checkpoint_path=checkpoint_path)

    summary = {
        "onnx_model_path": str(onnx_file),
        "file_size_bytes": onnx_file.stat().st_size,
        "parity": parity_results,
        "latency": latency_results,
    }

    res_file = Path(results_path)
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported ONNX parity and benchmark summary to {res_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Task 4 generator to ONNX and verify parity")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/task4/best_generator.pth")
    parser.add_argument("--output", type=str, default="models/onnx/task4_generator.onnx")
    args = parser.parse_args()

    run_export_and_verification(checkpoint_path=args.checkpoint, output_path=args.output)
