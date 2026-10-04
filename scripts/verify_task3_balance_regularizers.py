"""scripts/verify_task3_balance_regularizers.py
--------------------------------------------
Empirical benchmark comparing Task 3 Balance Regularizers:
1. L2 Deviation from Uniformity: sum((mean_w - 0.25) ** 2)
2. Batch-Mean Negative Entropy Maximization: sum(mean_w * log(mean_w + 1e-8))
3. Switch Transformer Load Balancing Loss (Fedus et al., 2021): 4 * sum(f_k * mean_w_k)

Evaluates on balanced Oxford-IIIT Pet data on GPU (CUDA) measuring:
- Validation PSNR, SSIM, L1 loss
- Average routing vector [w_clean, w_salt, w_blur, w_occ]
- Routing collapse detection (asserts all experts maintain >= 5% average utilization)
- Persists results to results/task3/balance_regularizers_benchmark.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_msssim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.shared.config import get_settings
from src.shared.corruptions import apply_corruption
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_val_manifest
from src.task3.moe_model import build_soft_moe


def compute_balance_loss(w: torch.Tensor, variant: str) -> torch.Tensor:
    """Compute candidate balance loss over batch routing weights w: (B, 4)."""
    mean_w = w.mean(dim=0)
    if variant == "l2_deviation":
        return torch.sum((mean_w - 0.25) ** 2)
    elif variant == "entropy":
        # Negative entropy: minimizing this maximizes entropy
        return torch.sum(mean_w * torch.log(mean_w + 1e-8))
    elif variant == "switch":
        # Switch Transformer loss: 4 * sum(f_k * mean_w_k)
        hard_assign = torch.argmax(w.detach(), dim=-1)
        f = torch.zeros(4, device=w.device)
        for k in range(4):
            f[k] = (hard_assign == k).float().mean()
        return 4.0 * torch.sum(f * mean_w)
    else:
        raise ValueError(f"Unknown regularizer variant: {variant}")


class CachedPairDataset(Dataset):
    """Simple in-memory dataset of (corrupted, clean, label) tuples."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.pairs[idx]


def build_benchmark_data(
    train_per_class: int = 64,
    val_per_class: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build balanced datasets for training and validation."""
    settings = get_settings()
    torch.manual_seed(seed)

    base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=settings)
    base_val = PetDataset(split="val", image_size=128, return_labels=False, settings=settings)
    val_manifest = load_val_manifest(settings=settings)

    val_by_class: Dict[int, List[dict]] = {0: [], 1: [], 2: [], 3: []}
    for item in val_manifest:
        val_by_class[item["corruption_label"]].append(item)

    val_pairs: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for cls_idx in range(4):
        items = val_by_class[cls_idx][:val_per_class]
        for it in items:
            clean = base_val[it["val_id"]]
            corr = apply_corruption(clean, it["corruption_type"], it["params"])
            val_pairs.append((corr, clean, cls_idx))

    train_pairs: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    perm = torch.randperm(len(base_train)).tolist()
    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for idx in perm:
        if all(counts[c] >= train_per_class for c in range(4)):
            break
        clean = base_train[idx]
        for c_label in range(4):
            if counts[c_label] < train_per_class:
                if c_label == 0:
                    corr = clean
                elif c_label == 1:
                    corr = apply_corruption(clean, "salt_and_pepper", {"p": 0.08})
                elif c_label == 2:
                    params_blur = {"kernel_size": 5, "sigma": 1.5}
                    corr = apply_corruption(clean, "gaussian_blur", params_blur)
                elif c_label == 3:
                    params_occ = {"boxes": [[32, 32, 80, 80]], "fill_value": 0.0}
                    corr = apply_corruption(clean, "occlusion", params_occ)
                train_pairs.append((corr, clean, c_label))
                counts[c_label] += 1

    train_loader = DataLoader(CachedPairDataset(train_pairs), batch_size=32, shuffle=True)
    val_loader = DataLoader(CachedPairDataset(val_pairs), batch_size=32, shuffle=False)
    return train_loader, val_loader


def evaluate_moe(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Evaluate SoftMoE model on validation loader and check for expert collapse."""
    model.eval()
    all_psnr, all_ssim, all_l1 = [], [], []
    all_w: List[np.ndarray] = []

    with torch.no_grad():
        for corr, clean, _ in val_loader:
            corr, clean = corr.to(device), clean.to(device)
            recon, w, _ = model(corr)

            mse = torch.mean((recon - clean) ** 2, dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            ssim = pytorch_msssim.ssim(recon, clean, data_range=1.0, size_average=False)
            l1 = torch.mean(torch.abs(recon - clean), dim=[1, 2, 3])

            all_psnr.extend(psnr.cpu().tolist())
            all_ssim.extend(ssim.cpu().tolist())
            all_l1.extend(l1.cpu().tolist())
            all_w.append(w.cpu().numpy())

    avg_w = np.concatenate(all_w, axis=0).mean(axis=0).tolist()
    min_util = float(min(avg_w))
    max_util = float(max(avg_w))
    # Non-collapse threshold: min routing >= 5% (0.05) and max routing <= 90% (0.90)
    collapse = bool(max_util > 0.90 or min_util < 0.05)

    return {
        "val_psnr": round(float(np.mean(all_psnr)), 2),
        "val_ssim": round(float(np.mean(all_ssim)), 4),
        "val_l1": round(float(np.mean(all_l1)), 4),
        "avg_routing_vector": [round(x, 4) for x in avg_w],
        "min_utilization": round(min_util, 4),
        "max_utilization": round(max_util, 4),
        "collapse_detected": collapse,
    }


def benchmark_regularizers(epochs: int = 5) -> Dict[str, Any]:
    """Run empirical balance regularizer benchmark across L2, Entropy, and Switch variants."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Balance Regularizer Benchmark on {device} ({epochs} epochs each)...")

    variants = ["l2_deviation", "entropy", "switch"]
    variant_names = {
        "l2_deviation": "L2 Deviation",
        "entropy": "Entropy Maximization",
        "switch": "Switch Load Balance",
    }
    results = {}

    train_loader, val_loader = build_benchmark_data(train_per_class=64, val_per_class=32, seed=42)

    l1_loss = nn.L1Loss()
    ce_loss = nn.CrossEntropyLoss()

    for var in variants:
        print(f"\nEvaluating: {variant_names[var]}...")
        torch.manual_seed(42)
        model = build_soft_moe(
            classifier_path="checkpoints/task2/classifier_best.pt",
            salt_path="checkpoints/task2/specialist_salt_best.pt",
            blur_path="checkpoints/task2/specialist_blur_best.pt",
            occlusion_path="checkpoints/task2/specialist_occlusion_best.pt",
        ).to(device)

        # Train gate and specialists with AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        for ep in range(1, epochs + 1):
            model.train()
            for corr, clean, lbl in train_loader:
                corr, clean, lbl = corr.to(device), clean.to(device), lbl.to(device)
                optimizer.zero_grad()
                recon, w, logits = model(corr)

                rec_l1 = l1_loss(recon, clean)
                rec_ssim = 1.0 - pytorch_msssim.ssim(recon, clean, data_range=1.0)
                loss_ce = ce_loss(logits, lbl)
                loss_bal = compute_balance_loss(w, var)

                # Total loss: 0.8 L1 + 0.2 (1-SSIM) + 0.1 CE + 0.01 Balance
                loss_total = 0.8 * rec_l1 + 0.2 * rec_ssim + 0.1 * loss_ce + 0.01 * loss_bal
                loss_total.backward()
                optimizer.step()

        val_metrics = evaluate_moe(model, val_loader, device)
        results[var] = {
            "name": variant_names[var],
            **val_metrics,
        }
        print(
            f"  Result -> PSNR: {val_metrics['val_psnr']} dB | SSIM: {val_metrics['val_ssim']} | "
            f"Routing: {val_metrics['avg_routing_vector']} | "
            f"Collapse: {val_metrics['collapse_detected']}"
        )

        # Assert no routing collapse (every expert >= 5% average utilization)
        assert not val_metrics["collapse_detected"], (
            f"Collapse detected for {variant_names[var]}: "
            f"min {val_metrics['min_utilization']:.4f} < 0.05 "
            f"or max {val_metrics['max_utilization']:.4f} > 0.90"
        )
        assert val_metrics["min_utilization"] >= 0.05, (
            f"Starvation detected for {variant_names[var]}: "
            f"min {val_metrics['min_utilization']:.4f} < 0.05"
        )

    out_path = Path("results/task3/balance_regularizers_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nPersisted benchmark results to {out_path}")
    return results


if __name__ == "__main__":
    benchmark_regularizers(epochs=5)
