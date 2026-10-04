"""
src/task2/inference.py
----------------------
Inference harness and benchmarking suite for Task 2 Hard-Routing Pipeline.

Provides:
- `restore_image`: Single-image inference with format conversions and metadata.
- `benchmark_router`: Comprehensive latency, throughput, and sub-module profiling
  across single-sample and batched execution.
- CLI entrypoint for interactive benchmarking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.shared.config import Settings, get_settings
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_val_manifest
from src.shared.corruptions import apply_corruption
from src.task2.router import HardRouter, load_hard_router


def _get_device(model: nn.Module, device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restore_image(
    router: HardRouter,
    image_input: Union[str, Path, Image.Image, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    oracle_label: Optional[int] = None,
) -> Dict[str, Any]:
    """Restore an input image using the hard-routing pipeline.

    Args:
        router: Initialized HardRouter model.
        image_input: Path to image file, PIL Image, or tensor (3, H, W).
        device: Target torch device.
        oracle_label: Optional ground-truth corruption label (0-3).

    Returns:
        Dict containing reconstructed tensor, restored PIL image, predicted class,
        probabilities, selected expert, and latency timings.
    """
    dev = _get_device(router, device)

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB").resize((128, 128))
        tensor = TF.to_tensor(img).to(dev)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB").resize((128, 128))
        tensor = TF.to_tensor(img).to(dev)
    elif isinstance(image_input, torch.Tensor):
        tensor = image_input.to(dev)
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    result = router.forward(tensor, oracle_labels=oracle_label)

    recon_tensor = result["reconstructed"].cpu()
    recon_np = (recon_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    recon_pil = Image.fromarray(recon_np)

    result["restored_image"] = recon_pil
    return result


def benchmark_router(
    router: HardRouter,
    test_tensors: List[Tuple[torch.Tensor, int]],
    batch_size: int = 16,
    num_warmup: int = 10,
    num_runs: int = 40,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Empirically benchmark HardRouter throughput, latency, and component breakdowns.

    Args:
        router: Loaded HardRouter instance.
        test_tensors: List of (corrupted_tensor, label) pairs.
        batch_size: Batch size for batched inference benchmark.
        num_warmup: Warmup iterations before timing.
        num_runs: Iterations for latency measurement.
        device: Device to profile on.

    Returns:
        Dictionary of latency, throughput, and per-class routing metrics.
    """
    dev = _get_device(router, device)
    router.eval()

    # 1. Benchmark Single Image Latency (Batch size = 1)
    single_tensor = test_tensors[0][0].unsqueeze(0).to(dev)
    for _ in range(num_warmup):
        _ = router.forward(single_tensor)

    single_latencies: List[float] = []
    clf_latencies: List[float] = []
    rest_latencies: List[float] = []

    for _ in range(num_runs):
        t0 = time.perf_counter()
        res = router.forward(single_tensor)
        dt = (time.perf_counter() - t0) * 1000.0
        single_latencies.append(dt)
        clf_latencies.append(res["classifier_latency_ms"])
        rest_latencies.append(res["restoration_latency_ms"])

    mean_single_ms = float(np.mean(single_latencies))
    std_single_ms = float(np.std(single_latencies))
    single_fps = 1000.0 / mean_single_ms if mean_single_ms > 0 else 0.0

    # 2. Benchmark Batched Latency (Batch size = batch_size)
    n_samples = min(len(test_tensors), batch_size)
    batch_tensors = torch.stack([test_tensors[i][0] for i in range(n_samples)]).to(dev)

    for _ in range(num_warmup):
        _ = router.forward(batch_tensors)

    batch_latencies: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = router.forward(batch_tensors)
        dt = (time.perf_counter() - t0) * 1000.0
        batch_latencies.append(dt)

    mean_batch_ms = float(np.mean(batch_latencies))
    std_batch_ms = float(np.std(batch_latencies))
    batched_throughput_fps = (n_samples * 1000.0) / mean_batch_ms if mean_batch_ms > 0 else 0.0

    # 3. Profile Latency per Corruption Branch
    per_branch_latency: Dict[str, float] = {}
    class_names = ["clean", "salt_pepper", "blur", "occlusion"]
    for class_id, cname in enumerate(class_names):
        # Filter a sample of this corruption class
        matching = [t[0] for t in test_tensors if t[1] == class_id]
        if matching:
            sample_inp = matching[0].unsqueeze(0).to(dev)
            branch_times: List[float] = []
            for _ in range(15):
                t0 = time.perf_counter()
                _ = router.forward(sample_inp, oracle_labels=class_id)
                branch_times.append((time.perf_counter() - t0) * 1000.0)
            per_branch_latency[cname] = round(float(np.mean(branch_times)), 3)

    return {
        "device": str(dev),
        "single_image": {
            "mean_latency_ms": round(mean_single_ms, 3),
            "std_latency_ms": round(std_single_ms, 3),
            "fps": round(single_fps, 2),
            "mean_classifier_ms": round(float(np.mean(clf_latencies)), 3),
            "mean_restoration_ms": round(float(np.mean(rest_latencies)), 3),
        },
        "batched": {
            "batch_size": n_samples,
            "mean_latency_ms": round(mean_batch_ms, 3),
            "std_latency_ms": round(std_batch_ms, 3),
            "throughput_fps": round(batched_throughput_fps, 2),
        },
        "per_branch_oracle_latency_ms": per_branch_latency,
    }


def main() -> None:
    """CLI for profiling and benchmarking HardRouter."""
    parser = argparse.ArgumentParser(description="Benchmark Task 2 HardRouter pipeline.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for profiling.")
    parser.add_argument("--runs", type=int, default=30, help="Profiling iterations.")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cpu', 'cuda').")
    parser.add_argument("--output", type=str, default="results/task2/router_benchmark.json")
    args = parser.parse_args()

    cfg = get_settings()
    dev = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading HardRouter on device: {dev}...")
    router = load_hard_router(device=dev)

    # Pre-cache representative validation samples across all 4 corruptions
    print("Loading test samples for benchmarking...")
    val_dataset = PetDataset(split="val", image_size=128, return_labels=False, settings=cfg)
    val_manifest = load_val_manifest(settings=cfg)

    test_pairs: List[Tuple[torch.Tensor, int]] = []
    # Pick first 64 items (16 per class)
    for item in val_manifest[:64]:
        clean = val_dataset[item["val_id"]]
        corr = apply_corruption(clean, item["corruption_type"], item["params"])
        test_pairs.append((corr, item["corruption_label"]))

    print(f"Running benchmark with {len(test_pairs)} real samples ({args.runs} runs)...")
    results = benchmark_router(
        router=router,
        test_tensors=test_pairs,
        batch_size=args.batch_size,
        num_runs=args.runs,
        device=dev,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("HARD-ROUTER BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Device: {results['device']}")
    print(f"Single Image Latency: {results['single_image']['mean_latency_ms']} ms ({results['single_image']['fps']} FPS)")
    print(f"  - Classifier: {results['single_image']['mean_classifier_ms']} ms")
    print(f"  - Restoration: {results['single_image']['mean_restoration_ms']} ms")
    print(f"Batched (B={args.batch_size}) Latency: {results['batched']['mean_latency_ms']} ms ({results['batched']['throughput_fps']} FPS)")
    print(f"Per-Branch Latency: {results['per_branch_oracle_latency_ms']}")
    print(f"Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
