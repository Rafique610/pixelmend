"""
src/task2/export_onnx.py
------------------------
Production ONNX computational graph export, graph validation, numerical parity
assertion, and latency benchmarking for Task 2 Hard-Routing Restoration system.
Exports corruption classifier and 3 specialist autoencoders with dynamic batching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.shared.tracking import ExperimentTracker
from src.task2.router import _load_model_from_checkpoint


def export_model_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    output_name: str = "output",
    opset_version: int = 17,
) -> Path:
    """Export PyTorch module to ONNX graph with dynamic batch axis."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    dynamic_axes = {"input": {0: "batch"}, output_name: {0: "batch"}}

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    return out


def verify_numerical_parity(
    model: nn.Module,
    onnx_path: Union[str, Path],
    batch_sizes: Tuple[int, ...] = (1, 4, 8),
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Assert output equivalence between PyTorch and ONNX Runtime across batch sizes."""
    model.eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity: Dict[str, Any] = {}
    all_passed = True

    for b in batch_sizes:
        torch.manual_seed(42 + b)
        x = torch.rand(b, 3, 128, 128, dtype=torch.float32)

        with torch.no_grad():
            y_pt = model(x).cpu().numpy()

        y_ort = sess.run(None, {"input": x.numpy()})[0]
        max_diff = float(np.max(np.abs(y_pt - y_ort)))
        mean_diff = float(np.mean(np.abs(y_pt - y_ort)))
        is_close = bool(np.allclose(y_pt, y_ort, atol=atol))
        if not is_close:
            all_passed = False

        parity[f"batch_{b}"] = {
            "batch_size": b,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "is_close": is_close,
        }

    parity["all_passed"] = all_passed
    parity["tolerance_atol"] = atol
    return parity


def benchmark_model_latency(
    model: nn.Module,
    onnx_path: Union[str, Path],
    num_warmup: int = 25,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Measure single-image CPU latency for PyTorch vs ONNX Runtime."""
    model.eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_tensor = torch.rand(1, 3, 128, 128, dtype=torch.float32)
    x_np = x_tensor.numpy()

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x_tensor)
        pt_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(x_tensor)
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(num_warmup):
        _ = sess.run(None, {"input": x_np})
    ort_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {"input": x_np})
        ort_times.append((time.perf_counter() - t0) * 1000.0)

    pt_mean = float(np.mean(pt_times))
    ort_mean = float(np.mean(ort_times))
    return {
        "pytorch_ms": round(pt_mean, 2),
        "onnxruntime_ms": round(ort_mean, 2),
        "speedup": round(float(pt_mean / max(1e-6, ort_mean)), 2),
        "throughput_fps": round(float(1000.0 / ort_mean), 1),
    }


class OnnxHardRouter:
    """Hard-Routing inference engine powered entirely by ONNX Runtime sessions."""

    def __init__(self, classifier_path: Path, salt_path: Path, blur_path: Path, occ_path: Path) -> None:
        self.clf_sess = ort.InferenceSession(str(classifier_path), providers=["CPUExecutionProvider"])
        self.specialists = {
            1: ort.InferenceSession(str(salt_path), providers=["CPUExecutionProvider"]),
            2: ort.InferenceSession(str(blur_path), providers=["CPUExecutionProvider"]),
            3: ort.InferenceSession(str(occ_path), providers=["CPUExecutionProvider"]),
        }

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch input image array (B, 3, H, W) through classifier and specialist sessions."""
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        logits = self.clf_sess.run(None, {"input": x.astype(np.float32)})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        decisions = np.argmax(exp_l / np.sum(exp_l, axis=1, keepdims=True), axis=1)

        restored = np.empty_like(x, dtype=np.float32)
        for class_idx in np.unique(decisions):
            mask = decisions == class_idx
            sub_x = x[mask].astype(np.float32)
            if class_idx == 0:
                restored[mask] = sub_x  # Identity bypass (0 FLOPS)
            elif class_idx in self.specialists:
                restored[mask] = self.specialists[class_idx].run(None, {"input": sub_x})[0]
            else:
                restored[mask] = sub_x

        return restored, decisions


def run_task2_onnx_export_pipeline(
    classifier_ckpt: str = "checkpoints/task2/classifier_best.pt",
    salt_ckpt: str = "checkpoints/task2/specialist_salt_best.pt",
    blur_ckpt: str = "checkpoints/task2/specialist_blur_best.pt",
    occlusion_ckpt: str = "checkpoints/task2/specialist_occlusion_best.pt",
    output_dir: str = "models/onnx",
    metrics_path: str = "results/task2/onnx_parity_benchmark.json",
    opset_version: int = 17,
    log_mlflow: bool = True,
) -> Dict[str, Any]:
    """Execute complete ONNX export, parity verification, and latency logging for Task 2."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    models_meta = [
        ("classifier", classifier_ckpt, "task2_classifier.onnx", "logits", "classifier"),
        ("specialist_salt", salt_ckpt, "task2_specialist_salt.onnx", "output", "salt_and_pepper"),
        ("specialist_blur", blur_ckpt, "task2_specialist_blur.onnx", "output", "gaussian_blur"),
        ("specialist_occlusion", occlusion_ckpt, "task2_specialist_occlusion.onnx", "output", "occlusion"),
    ]

    results: Dict[str, Any] = {"models": {}, "all_passed": True}

    for name, ckpt_file, onnx_name, out_node, mod_type in models_meta:
        onnx_file = out_dir / onnx_name
        model = _load_model_from_checkpoint(ckpt_file, mod_type, device)
        export_model_onnx(model, onnx_file, output_name=out_node, opset_version=opset_version)

        parity = verify_numerical_parity(model, onnx_file)
        latency = benchmark_model_latency(model, onnx_file)
        if not parity["all_passed"]:
            results["all_passed"] = False

        results["models"][name] = {
            "checkpoint": str(ckpt_file),
            "onnx_path": str(onnx_file),
            "file_size_mb": round(onnx_file.stat().st_size / (1024 * 1024), 2),
            "parity": parity,
            "latency": latency,
        }

    # Benchmark complete OnnxHardRouter
    onnx_router = OnnxHardRouter(
        out_dir / "task2_classifier.onnx",
        out_dir / "task2_specialist_salt.onnx",
        out_dir / "task2_specialist_blur.onnx",
        out_dir / "task2_specialist_occlusion.onnx",
    )
    dummy_img = np.random.rand(1, 3, 128, 128).astype(np.float32)
    for _ in range(15):
        _ = onnx_router.predict(dummy_img)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = onnx_router.predict(dummy_img)
        times.append((time.perf_counter() - t0) * 1000.0)
    results["onnx_router_pipeline_latency_ms"] = round(float(np.mean(times)), 2)
    results["onnx_router_throughput_fps"] = round(float(1000.0 / np.mean(times)), 1)

    met_path = Path(metrics_path)
    met_path.parent.mkdir(parents=True, exist_ok=True)
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if log_mlflow:
        tracker = ExperimentTracker()
        with tracker.run(run_name="onnx-verify", experiment_name="task2-hard-routing"):
            tracker.log_param("onnx_opset", opset_version)
            tracker.log_metric("parity_all_passed", 1.0 if results["all_passed"] else 0.0)
            tracker.log_metric("onnx_router_latency_ms", results["onnx_router_pipeline_latency_ms"])
            tracker.log_metric("onnx_router_fps", results["onnx_router_throughput_fps"])
            tracker.log_artifact(str(met_path))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Task 2 models to ONNX and verify parity")
    parser.add_argument("--classifier-ckpt", default="checkpoints/task2/classifier_best.pt")
    parser.add_argument("--salt-ckpt", default="checkpoints/task2/specialist_salt_best.pt")
    parser.add_argument("--blur-ckpt", default="checkpoints/task2/specialist_blur_best.pt")
    parser.add_argument("--occlusion-ckpt", default="checkpoints/task2/specialist_occlusion_best.pt")
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--metrics", default="results/task2/onnx_parity_benchmark.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    results = run_task2_onnx_export_pipeline(
        classifier_ckpt=args.classifier_ckpt,
        salt_ckpt=args.salt_ckpt,
        blur_ckpt=args.blur_ckpt,
        occlusion_ckpt=args.occlusion_ckpt,
        output_dir=args.output_dir,
        metrics_path=args.metrics,
        opset_version=args.opset,
        log_mlflow=not args.no_mlflow,
    )
    print(f"\nTask 2 ONNX Export Complete: All Parity Passed = {results['all_passed']}")
    print(f"ONNX Router Latency: {results['onnx_router_pipeline_latency_ms']} ms ({results['onnx_router_throughput_fps']} FPS)")


if __name__ == "__main__":
    main()
