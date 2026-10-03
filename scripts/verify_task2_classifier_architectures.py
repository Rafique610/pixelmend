"""
scripts/verify_task2_classifier_architectures.py
------------------------------------------------
Empirical benchmark comparing Task 2 Corruption Classifier architectures:
1. CustomConvClassifier (4-stage Conv-BN-LeakyReLU-MaxPool + GAP + Linear head)
2. MobileNetClassifier (Depthwise separable inverted residuals + GAP + Linear head)
3. ResNet18Classifier (Adapted ResNet-18 backbone)

Measures:
- Parameter counts (trainable & total)
- Model size (MB)
- Inference latency (batch=1 and batch=16 on CPU)
- Throughput (samples/sec)
- 2-epoch empirical training & convergence test on real balanced Oxford-IIIT Pet corrupted data:
  - Initial Loss, Final Loss, Val Accuracy, Val Macro-F1, Epoch Time
- Persists all results to results/task2/classifier_architecture_benchmark.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from src.shared.config import get_settings
from src.shared.corruptions import (
    CorruptionType,
    apply_corruption,
    sample_random_corruption,
)
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_val_manifest
from src.task2.classifier import (
    CorruptionClassifier,
    CustomConvClassifier,
    MobileNetClassifier,
    ResNet18Classifier,
    build_classifier,
)


class SimpleTensorDataset(Dataset):
    """Memory-cached dataset for fast empirical benchmark."""

    def __init__(self, data: List[Tuple[torch.Tensor, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx]


def build_balanced_benchmark_subsets(
    train_per_class: int = 64,
    val_per_class: int = 32,
    seed: int = 42,
) -> Tuple[SimpleTensorDataset, SimpleTensorDataset]:
    """Build balanced cached subsets for fair multi-class convergence comparison."""
    settings = get_settings()
    torch.manual_seed(seed)

    print(f"Loading data to build balanced benchmark subsets ({train_per_class*4} train, {val_per_class*4} val)...")
    base_train = PetDataset(split="train", image_size=128, return_labels=False, settings=settings)
    base_val = PetDataset(split="val", image_size=128, return_labels=False, settings=settings)
    val_manifest = load_val_manifest(settings=settings)

    # 1. Build balanced validation subset from deterministic val_manifest
    val_by_class: Dict[int, List[dict]] = {0: [], 1: [], 2: [], 3: []}
    for item in val_manifest:
        val_by_class[item["corruption_label"]].append(item)

    val_pairs: List[Tuple[torch.Tensor, int]] = []
    for cls_idx in range(4):
        items = val_by_class[cls_idx][:val_per_class]
        for it in items:
            val_id = it["val_id"]
            clean_tensor = base_val[val_id]
            corr = apply_corruption(clean_tensor, it["corruption_type"], it["params"])
            val_pairs.append((corr, cls_idx))

    # 2. Build balanced training subset dynamically from train split
    train_by_class: Dict[int, List[Tuple[torch.Tensor, int]]] = {0: [], 1: [], 2: [], 3: []}
    target_train_total = train_per_class * 4
    perm = torch.randperm(len(base_train)).tolist()

    idx = 0
    while any(len(train_by_class[c]) < train_per_class for c in range(4)) and idx < len(perm):
        img_idx = perm[idx]
        clean_tensor = base_train[img_idx]
        for c_label in range(4):
            if len(train_by_class[c_label]) < train_per_class:
                if c_label == 0:
                    train_by_class[0].append((clean_tensor, 0))
                else:
                    c_type, l_sampled, params = sample_random_corruption(image_size=128)
                    # Force the sampled corruption to match target class
                    if c_label == 1:
                        c_type = CorruptionType.SALT_AND_PEPPER
                        params = {"prob": 0.08}
                    elif c_label == 2:
                        c_type = CorruptionType.GAUSSIAN_BLUR
                        params = {"kernel_size": 5, "sigma": 1.5}
                    elif c_label == 3:
                        c_type = CorruptionType.OCCLUSION
                        params = {"boxes": [[20, 20, 70, 70]], "fill_value": 0.0}
                    corr = apply_corruption(clean_tensor, c_type, params)
                    train_by_class[c_label].append((corr, c_label))
        idx += 1

    train_pairs: List[Tuple[torch.Tensor, int]] = []
    for c in range(4):
        train_pairs.extend(train_by_class[c][:train_per_class])

    # Shuffle training pairs
    shuffled_idx = torch.randperm(len(train_pairs)).tolist()
    train_pairs = [train_pairs[i] for i in shuffled_idx]

    print(f"Subsets prepared: {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs.")
    return SimpleTensorDataset(train_pairs), SimpleTensorDataset(val_pairs)


def profile_architecture(
    model: nn.Module,
    name: str,
    device: torch.device,
    iters: int = 50,
    warmup: int = 10,
) -> Dict[str, Any]:
    """Measure structural parameters and CPU inference latency."""
    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Latency: Batch size 1
    x1 = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x1)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x1)
        lat_b1_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # Latency: Batch size 16
    x16 = torch.randn(16, 3, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x16)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x16)
        lat_b16_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    throughput_fps = (16.0 / (lat_b16_ms / 1000.0))

    return {
        "name": name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "latency_b1_ms": round(lat_b1_ms, 2),
        "latency_b16_ms": round(lat_b16_ms, 2),
        "throughput_fps": round(throughput_fps, 1),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Compute validation loss, accuracy (%), and macro-F1 score."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    val_loss = total_loss / max(1, total_samples)
    val_acc = (total_correct / max(1, total_samples)) * 100.0
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, float(val_macro_f1)


def run_convergence_benchmark(
    models_dict: Dict[str, nn.Module],
    train_ds: SimpleTensorDataset,
    val_ds: SimpleTensorDataset,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> Dict[str, Dict[str, Any]]:
    """Empirically test initial loss, training convergence, and validation accuracy across 5 full epochs."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    convergence_results: Dict[str, Dict[str, Any]] = {}

    for name, model in models_dict.items():
        print(f"\n--- Training candidate: {name} ({epochs} epochs) ---", flush=True)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Pre-training validation (Epoch 0)
        init_loss, init_acc, init_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f"  Init Val Loss: {init_loss:.4f} | Init Val Acc: {init_acc:.1f}% | Init F1: {init_f1:.4f}", flush=True)

        epoch_times: List[float] = []
        val_losses: List[float] = []
        val_accs: List[float] = []
        val_f1s: List[float] = []

        for ep in range(1, epochs + 1):
            t_start = time.perf_counter()
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)

            t_ep = time.perf_counter() - t_start
            epoch_times.append(t_ep)

            # Evaluate after epoch
            v_loss, v_acc, v_f1 = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(v_loss)
            val_accs.append(v_acc)
            val_f1s.append(v_f1)
            print(
                f"  Epoch {ep}/{epochs} ({t_ep:.2f}s) | Train Loss: {train_loss/len(train_ds):.4f} "
                f"| Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.1f}% | Val Macro-F1: {v_f1:.4f}",
                flush=True,
            )

        convergence_results[name] = {
            "initial_val_loss": round(init_loss, 4),
            "initial_val_acc": round(init_acc, 2),
            "final_val_loss": round(val_losses[-1], 4),
            "final_val_acc": round(val_accs[-1], 2),
            "final_val_f1": round(val_f1s[-1], 4),
            "mean_epoch_time_s": round(sum(epoch_times) / len(epoch_times), 2),
        }

    return convergence_results


def main():
    settings = get_settings()
    device = settings.torch_device
    print(f"Task 2 Classifier Architecture Empirical Benchmark")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Instantiate the 3 candidate models
    torch.manual_seed(42)
    models_dict: Dict[str, nn.Module] = {
        "CustomConvClassifier": build_classifier(backbone="custom_conv", channels=(32, 64, 128, 256)),
        "MobileNetClassifier": build_classifier(backbone="mobilenet", channels=(32, 64, 96, 160)),
        "ResNet18Classifier": build_classifier(backbone="resnet18", pretrained=False),
    }

    # 2. Structural & Latency Profiling
    print("\n--- Structural & Latency Profiling ---")
    profiles: Dict[str, Dict[str, Any]] = {}
    for name, model in models_dict.items():
        prof = profile_architecture(model, name, device)
        profiles[name] = prof
        print(
            f"  {name:<22} | Params: {prof['trainable_params']:>10,d} | Size: {prof['model_size_mb']:>5.2f} MB "
            f"| B=1: {prof['latency_b1_ms']:>5.2f} ms | B=16: {prof['latency_b16_ms']:>5.2f} ms | FPS: {prof['throughput_fps']:>5.1f}"
        )

    # 3. Balanced Dataset Construction
    train_ds, val_ds = build_balanced_benchmark_subsets(train_per_class=64, val_per_class=32, seed=42)

    # 4. Empirical Convergence Run
    conv_results = run_convergence_benchmark(
        models_dict=models_dict,
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        epochs=5,
        batch_size=16,
        lr=1e-3,
    )

    # 5. Consolidate and Export
    results_dir = Path("results/task2")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "classifier_architecture_benchmark.json"

    consolidated = {}
    for name in models_dict:
        consolidated[name] = {
            **profiles[name],
            **conv_results[name],
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    print("\n" + "=" * 110)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 110)
    header = (
        f"{'Candidate':<22} | {'Params':<10} | {'Size(MB)':<8} | {'Lat B=1':<8} | {'Lat B=16':<9} | "
        f"{'FPS':<7} | {'Val Acc(%)':<10} | {'Macro F1':<8} | {'Ep Time(s)'}"
    )
    print(header)
    print("-" * 110)
    for name, d in consolidated.items():
        row = (
            f"{name:<22} | {d['trainable_params']:<10,d} | {d['model_size_mb']:<8.2f} | "
            f"{d['latency_b1_ms']:<8.2f} | {d['latency_b16_ms']:<9.2f} | {d['throughput_fps']:<7.1f} | "
            f"{d['final_val_acc']:<10.1f} | {d['final_val_f1']:<8.4f} | {d['mean_epoch_time_s']:<9.2f}"
        )
        print(row)
    print("=" * 110)
    print(f"Results successfully persisted to: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()
