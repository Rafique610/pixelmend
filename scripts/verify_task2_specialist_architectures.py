"""scripts/verify_task2_specialist_architectures.py.

Empirical architectural research and benchmarking for Task 2 Specialist Autoencoders:
1. Alternative 1: Homogeneous Task 1 Architecture (channels=(32, 64, 128, 256), bottleneck=256, ResBlocks)
2. Alternative 2: Lightweight Shared Variant (channels=(32, 64, 128), bottleneck=128, no ResBlocks)
3. Alternative 3: Corruption-Specific Tailored Architectures (SP: 3-stage ResBlock, Blur: 4-stage ResBlock, Occ: 4-stage ResBlock)

Measures:
- Parameters per specialist & system total (3 specialists)
- Model size in MB
- CPU Inference Latency (Batch=1 and Batch=16)
- Throughput (FPS on CPU)
- Compression ratio
- 3-epoch empirical restoration convergence on real Oxford Pets corrupted images:
  - Initial Loss, Final Train Loss, Val Loss, Val PSNR (dB), Val SSIM, Epoch Duration
- Persists results to results/task2/specialist_architecture_benchmark.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.shared.config import get_settings
from src.shared.corruptions import (
    CorruptionType,
    apply_corruption,
    sample_random_corruption,
)
from src.shared.datasets.pets import PetDataset
from src.shared.losses import CombinedReconstructionLoss
from src.shared.manifests import load_val_manifest
from src.shared.metrics import compute_psnr, compute_ssim
from src.task2.specialist import (
    SpecialistAutoencoder,
    VALID_SPECIALISTS,
    build_specialist,
)


class RestorationPairDataset(Dataset):
    """Memory-cached dataset for fast empirical restoration benchmarking."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


def build_restoration_subsets(
    train_per_corruption: int = 64,
    val_per_corruption: int = 32,
    seed: int = 42,
) -> Tuple[Dict[str, RestorationPairDataset], Dict[str, RestorationPairDataset]]:
    """Build isolated cached train and validation datasets for each corruption type."""
    settings = get_settings()
    torch.manual_seed(seed)

    print(f"Loading Oxford Pets data ({train_per_corruption} train / {val_per_corruption} val per corruption)...")
    base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=settings)
    base_val = PetDataset(split="val", image_size=128, return_labels=False, settings=settings)
    val_manifest = load_val_manifest(settings=settings)

    # Label mapping in manifest: 1 -> salt_and_pepper, 2 -> gaussian_blur, 3 -> occlusion
    label_to_corr = {
        1: "salt_and_pepper",
        2: "gaussian_blur",
        3: "occlusion",
    }

    # 1. Build validation sets from deterministic manifest
    val_manifest_by_corr: Dict[str, List[dict]] = {c: [] for c in VALID_SPECIALISTS}
    for item in val_manifest:
        lbl = item["corruption_label"]
        if lbl in label_to_corr:
            corr_name = label_to_corr[lbl]
            val_manifest_by_corr[corr_name].append(item)

    val_datasets: Dict[str, RestorationPairDataset] = {}
    for c_name, items in val_manifest_by_corr.items():
        subset = items[:val_per_corruption]
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for it in subset:
            val_id = it["val_id"]
            clean_tensor = base_val[val_id]
            corrupted_tensor = apply_corruption(clean_tensor, it["corruption_type"], it["params"])
            pairs.append((corrupted_tensor, clean_tensor))
        val_datasets[c_name] = RestorationPairDataset(pairs)

    # 2. Build training sets dynamically from train split
    train_datasets: Dict[str, RestorationPairDataset] = {}
    perm = torch.randperm(len(base_train)).tolist()

    for c_name in VALID_SPECIALISTS:
        pairs = []
        for i in range(train_per_corruption):
            img_idx = perm[(i * 3 + list(VALID_SPECIALISTS).index(c_name)) % len(perm)]
            clean_tensor = base_train[img_idx]

            if c_name == "salt_and_pepper":
                prob = float(torch.empty(1).uniform_(0.02, 0.15).item())
                corrupted = apply_corruption(clean_tensor, CorruptionType.SALT_AND_PEPPER, {"prob": prob})
            elif c_name == "gaussian_blur":
                k = int(torch.tensor([3, 5, 7])[torch.randint(0, 3, (1,)).item()])
                sigma = float(torch.empty(1).uniform_(0.5, 2.5).item())
                corrupted = apply_corruption(clean_tensor, CorruptionType.GAUSSIAN_BLUR, {"kernel_size": k, "sigma": sigma})
            else:  # occlusion
                boxes = [[25, 25, 75, 75]]
                corrupted = apply_corruption(clean_tensor, CorruptionType.OCCLUSION, {"boxes": boxes, "fill_value": 0.0})

            pairs.append((corrupted, clean_tensor))
        train_datasets[c_name] = RestorationPairDataset(pairs)

    return train_datasets, val_datasets


def profile_model_efficiency(
    model: SpecialistAutoencoder,
    device: torch.device,
    iters: int = 30,
    warmup: int = 5,
) -> Dict[str, Any]:
    """Measure parameters, model size, and inference latency on CPU."""
    model.eval()
    model.to(device)

    params = model.count_parameters()
    model_size_mb = (params * 4) / (1024 * 1024)
    compression = model.compression_ratio((3, 128, 128))

    # Latency Batch=1
    x1 = torch.rand(1, 3, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x1)
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(x1)
        latency_b1_ms = ((time.perf_counter() - start) / iters) * 1000

    # Latency Batch=16
    x16 = torch.rand(16, 3, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x16)
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(x16)
        latency_b16_ms = ((time.perf_counter() - start) / iters) * 1000

    fps = 1000.0 / latency_b1_ms

    return {
        "params": params,
        "model_size_mb": round(model_size_mb, 2),
        "compression_ratio": round(compression, 1),
        "latency_b1_ms": round(latency_b1_ms, 2),
        "latency_b16_ms": round(latency_b16_ms, 2),
        "throughput_fps": round(fps, 1),
    }


def evaluate_restoration(
    model: SpecialistAutoencoder,
    dataset: RestorationPairDataset,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate validation loss, PSNR, and SSIM across validation pairs."""
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        for corrupted, clean in loader:
            corrupted, clean = corrupted.to(device), clean.to(device)
            pred = model(corrupted)
            loss = criterion(pred, clean)

            bs = clean.size(0)
            total_loss += loss.item() * bs
            for i in range(bs):
                total_psnr += compute_psnr(pred[i], clean[i])
                total_ssim += compute_ssim(pred[i], clean[i])
            total_samples += bs

    return {
        "val_loss": round(total_loss / total_samples, 4),
        "val_psnr": round(total_psnr / total_samples, 2),
        "val_ssim": round(total_ssim / total_samples, 4),
    }


def train_and_evaluate_specialist(
    model: SpecialistAutoencoder,
    train_dataset: RestorationPairDataset,
    val_dataset: RestorationPairDataset,
    epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Train specialist for empirical convergence profiling."""
    model.to(device)
    criterion = CombinedReconstructionLoss(alpha=0.84)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Measure initial loss before training
    init_eval = evaluate_restoration(model, val_dataset, criterion, device)
    initial_loss = init_eval["val_loss"]

    epoch_times = []
    final_train_loss = 0.0

    for epoch in range(epochs):
        model.train()
        start = time.perf_counter()
        running_train_loss = 0.0

        for corrupted, clean in loader:
            corrupted, clean = corrupted.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(corrupted)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * clean.size(0)

        epoch_times.append(time.perf_counter() - start)
        final_train_loss = running_train_loss / len(train_dataset)

    # Final validation evaluation
    final_eval = evaluate_restoration(model, val_dataset, criterion, device)

    return {
        "initial_loss": round(initial_loss, 4),
        "final_train_loss": round(final_train_loss, 4),
        "val_loss": final_eval["val_loss"],
        "val_psnr": final_eval["val_psnr"],
        "val_ssim": final_eval["val_ssim"],
        "mean_epoch_time_s": round(sum(epoch_times) / len(epoch_times), 3),
    }


def run_benchmark() -> None:
    """Execute complete architecture benchmark across all 3 alternatives."""
    device = torch.device("cpu")
    print("=" * 80)
    print("TASK 2: SPECIALIST AUTOENCODER ARCHITECTURE BENCHMARK")
    print(f"Device: {device}")
    print("=" * 80)

    train_sets, val_sets = build_restoration_subsets(train_per_corruption=64, val_per_corruption=32)

    alternatives = [
        ("Alternative 1: Homogeneous Task 1 AE", "homogeneous"),
        ("Alternative 2: Lightweight Shared Variant", "lightweight"),
        ("Alternative 3: Corruption-Tailored Topologies", "tailored"),
    ]

    benchmark_results: Dict[str, Any] = {}

    for alt_name, variant in alternatives:
        print(f"\nEvaluating {alt_name} ({variant})...")
        alt_summary: Dict[str, Any] = {
            "variant": variant,
            "specialists": {},
        }

        total_params = 0
        total_size_mb = 0.0
        avg_psnr_list = []
        avg_ssim_list = []
        avg_loss_list = []

        for corr_type in VALID_SPECIALISTS:
            model = build_specialist(corr_type, variant=variant)
            eff = profile_model_efficiency(model, device)
            train_perf = train_and_evaluate_specialist(
                model, train_sets[corr_type], val_sets[corr_type], epochs=3, device=device
            )

            total_params += eff["params"]
            total_size_mb += eff["model_size_mb"]
            avg_psnr_list.append(train_perf["val_psnr"])
            avg_ssim_list.append(train_perf["val_ssim"])
            avg_loss_list.append(train_perf["val_loss"])

            alt_summary["specialists"][corr_type] = {
                "efficiency": eff,
                "convergence": train_perf,
            }

            print(
                f"  [{corr_type:<15}] Params: {eff['params']:,} | Size: {eff['model_size_mb']}MB | "
                f"Lat: {eff['latency_b1_ms']}ms | Val Loss: {train_perf['val_loss']} | "
                f"PSNR: {train_perf['val_psnr']} dB | SSIM: {train_perf['val_ssim']}"
            )

        alt_summary["system_total"] = {
            "total_params": total_params,
            "total_size_mb": round(total_size_mb, 2),
            "mean_val_loss": round(sum(avg_loss_list) / len(avg_loss_list), 4),
            "mean_val_psnr": round(sum(avg_psnr_list) / len(avg_psnr_list), 2),
            "mean_val_ssim": round(sum(avg_ssim_list) / len(avg_ssim_list), 4),
        }
        benchmark_results[alt_name] = alt_summary

    # Save results to disk
    out_dir = Path("results/task2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "specialist_architecture_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 90)
    print("SYSTEM ARCHITECTURE BENCHMARK SUMMARY")
    print("=" * 90)
    print(
        f"{'Alternative':<42} | {'Total Params':<13} | {'Total Size':<10} | "
        f"{'Mean PSNR':<10} | {'Mean SSIM':<10} | {'Mean Val Loss'}"
    )
    print("-" * 90)
    for alt_name, data in benchmark_results.items():
        st = data["system_total"]
        print(
            f"{alt_name:<42} | {st['total_params']:<13,d} | {st['total_size_mb']} MB   | "
            f"{st['mean_val_psnr']} dB    | {st['mean_val_ssim']:<10} | {st['mean_val_loss']}"
        )
    print("=" * 90)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
