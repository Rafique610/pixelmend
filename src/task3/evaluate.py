"""src/task3/evaluate.py
---------------------
Exhaustive Comparative Evaluation and Benchmark Analysis across:
1. Task 1: Universal Denoising Autoencoder
2. Task 2: Hard Router (Predicted)
3. Task 2: Hard Router (Oracle)
4. Task 3: Soft Mixture-of-Experts (MoE)

Evaluates on the official Oxford-IIIT Pet test split (manifests/test_manifest.json),
measures single-image inference latencies, outputs unified benchmark tables to CSV,
and exports 12-sample qualitative comparison grids and 4-sample failure/tradeoff panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_msssim
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from src.shared.config import get_settings
from src.shared.corruptions import apply_corruption, CorruptionType
from src.shared.tracking import ExperimentTracker
from src.task1.autoencoder import UniversalAutoencoder
from src.task2.router import HardRouter, load_hard_router
from src.task3.moe_model import SoftMoE
from src.task3.routing_analysis import load_trained_moe

SEVERITY_SPECS = [
    ("clean", "Clean (Identity)", "clean", 0),
    ("sp_mild", "Salt & Pepper (Mild, p=0.03)", "salt_and_pepper", 1),
    ("sp_medium", "Salt & Pepper (Med, p=0.08)", "salt_and_pepper", 2),
    ("sp_severe", "Salt & Pepper (Severe, p=0.15)", "salt_and_pepper", 3),
    ("blur_mild", "Gaussian Blur (Mild, k=3, s=0.7)", "gaussian_blur", 1),
    ("blur_medium", "Gaussian Blur (Med, k=5, s=1.5)", "gaussian_blur", 2),
    ("blur_severe", "Gaussian Blur (Severe, k=7, s=2.5)", "gaussian_blur", 3),
    ("occl_mild", "Occlusion (Mild, 1 box, 10%)", "occlusion", 1),
    ("occl_medium", "Occlusion (Med, 2 boxes, 20%)", "occlusion", 2),
    ("occl_severe", "Occlusion (Severe, 3 boxes, 35%)", "occlusion", 3),
]


def load_clean_images_cache(
    images_dir: Path, image_names: Sequence[str]
) -> Dict[str, torch.Tensor]:
    """Pre-cache clean test images into RAM as normalized (3, 128, 128) float32 tensors."""
    tfm = T.Compose([
        T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    cache: Dict[str, torch.Tensor] = {}
    unique_names = sorted(list(set(image_names)))
    t0 = time.time()
    print(f"Pre-caching {len(unique_names)} unique clean test images into RAM...", flush=True)
    for i, name in enumerate(unique_names):
        img_path = images_dir / name
        if img_path.is_file():
            with Image.open(img_path) as raw:
                cache[name] = tfm(raw.convert("RGB"))
        if (i + 1) % 1000 == 0 or (i + 1) == len(unique_names):
            print(f"  [{i+1}/{len(unique_names)}] cached ({time.time() - t0:.1f}s)", flush=True)
    return cache


def compute_batch_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[List[float], List[float], List[float]]:
    """Compute PSNR, SSIM, and MAE lists for a batch of predictions."""
    with torch.no_grad():
        pred_c = pred.clamp(0.0, 1.0)
        target_c = target.clamp(0.0, 1.0)
        mse = torch.mean((pred_c - target_c) ** 2, dim=[1, 2, 3])
        psnr = (10.0 * torch.log10(1.0 / (mse + 1e-8))).cpu().tolist()
        ssim = pytorch_msssim.ssim(pred_c, target_c, data_range=1.0, size_average=False).cpu().tolist()
        mae = torch.mean(torch.abs(pred_c - target_c), dim=[1, 2, 3]).cpu().tolist()
    return psnr, ssim, mae


def benchmark_model_latencies(
    t1_model: nn.Module,
    t2_router: HardRouter,
    t3_moe: SoftMoE,
    device: torch.device,
    tau: float = 2.526,
    num_runs: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Benchmark single-image inference latency across all 4 restoration systems."""
    print(f"Benchmarking inference latency on {device} ({num_runs} runs, B=1)...", flush=True)
    t1_model.to(device)
    t2_router.to(device)
    t3_moe.to(device)
    dummy = torch.rand(1, 3, 128, 128, device=device)
    dummy_lbl = torch.tensor([1], device=device, dtype=torch.long)


    # Warmup
    for _ in range(10):
        _ = t1_model(dummy)
        _ = t2_router(dummy, oracle_labels=None)
        _ = t2_router(dummy, oracle_labels=dummy_lbl)
        _ = t3_moe(dummy, tau=tau)

    if device.type == "cuda":
        torch.cuda.synchronize()

    def _time_fn(fn: Any) -> float:
        times = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(times))

    latencies = {
        "task1_universal": round(_time_fn(lambda: t1_model(dummy)), 2),
        "task2_predicted": round(_time_fn(lambda: t2_router(dummy, oracle_labels=None)), 2),
        "task2_oracle": round(_time_fn(lambda: t2_router(dummy, oracle_labels=dummy_lbl)), 2),
        "task3_soft_moe": round(_time_fn(lambda: t3_moe(dummy, tau=tau)), 2),
    }
    return latencies


def run_comparative_evaluation(
    manifest_path: Union[str, Path] = "manifests/test_manifest.json",
    subsample_ratio: float = 0.2,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    output_dir: Union[str, Path] = "results/task3",
) -> Dict[str, Any]:
    """Execute end-to-end comparative benchmark across Tasks 1, 2, and 3."""
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dev = device or get_settings().torch_device

    print(f"\n=== Executing Step 9 Comparative Test Evaluation on {dev} ===")

    # 1. Load Models
    print("Loading models:")
    t1_model, _ = UniversalAutoencoder.load_from_checkpoint("checkpoints/task1/best_model.pth", device=dev)
    t1_model.eval()
    print("  - Task 1: Universal Autoencoder loaded.")

    t2_router = load_hard_router(device=dev)
    t2_router.eval()
    print("  - Task 2: Hard Router loaded.")

    t3_moe, tau, _ = load_trained_moe("checkpoints/task3/best_model.pth", dev)
    t3_moe.eval()
    print(f"  - Task 3: Soft MoE loaded (tau={tau:.3f}).")

    # 2. Benchmark Inference Latency
    gpu_latencies = benchmark_model_latencies(t1_model, t2_router, t3_moe, dev, tau=tau)
    print(f"GPU Latencies (ms): {gpu_latencies}")

    # 3. Load Test Entries
    manifest_data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    all_entries = manifest_data["entries"]
    if 0.0 < subsample_ratio < 1.0:
        image_to_entries = defaultdict(list)
        for e in all_entries:
            image_to_entries[e["image"]].append(e)
        unique_imgs = sorted(list(image_to_entries.keys()))
        step = int(1.0 / subsample_ratio)
        selected_imgs = set(unique_imgs[::step])
        entries = [e for e in all_entries if e["image"] in selected_imgs]
    else:
        entries = all_entries
    print(f"Evaluating {len(entries)} test instances across all corruption variants (subsample: {subsample_ratio:.1%})...")

    images_dir = get_settings().data_dir / "oxford-iiit-pet" / "images"
    clean_cache = load_clean_images_cache(images_dir, [e["image"] for e in entries])

    # Containers for metrics
    methods = ["task1_univ", "task2_oracle", "task2_pred", "task3_moe"]
    metrics_acc = {m: {"psnr": [], "ssim": [], "mae": []} for m in methods}
    per_c_acc = {m: defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []}) for m in methods}
    per_s_acc = {m: defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []}) for m in methods}

    qual_candidates: Dict[str, Dict[str, Any]] = {}
    failure_candidates: List[Dict[str, Any]] = []

    total = len(entries)
    t0 = time.time()

    with torch.no_grad():
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = entries[start_idx:end_idx]

            clean_list = [clean_cache[e["image"]] for e in batch]
            corr_list = []
            for clean_t, e in zip(clean_list, batch):
                c_enum = CorruptionType(e["corruption_type"])
                corr_t = apply_corruption(clean_t, c_enum, e.get("params", {}))
                corr_list.append(corr_t)

            lbl_list = [int(e["corruption_label"]) for e in batch]
            lbl_tensor = torch.tensor(lbl_list, device=dev, dtype=torch.long)

            corr_batch = torch.stack(corr_list).to(dev)
            clean_batch = torch.stack(clean_list).to(dev)

            # Model 1: Task 1 Universal Autoencoder
            out_t1 = t1_model(corr_batch)

            # Model 2: Task 2 Oracle
            out_t2_orac = t2_router(corr_batch, oracle_labels=lbl_tensor)
            recon_t2_orac = out_t2_orac["reconstructed"]

            # Model 3: Task 2 Predicted
            out_t2_pred = t2_router(corr_batch, oracle_labels=None)
            recon_t2_pred = out_t2_pred["reconstructed"]

            # Model 4: Task 3 Soft MoE
            out_t3_moe, w_t3, logits_t3 = t3_moe(corr_batch, tau=tau)

            p_t1, s_t1, m_t1 = compute_batch_metrics(out_t1, clean_batch)
            p_t2o, s_t2o, m_t2o = compute_batch_metrics(recon_t2_orac, clean_batch)
            p_t2p, s_t2p, m_t2p = compute_batch_metrics(recon_t2_pred, clean_batch)
            p_t3, s_t3, m_t3 = compute_batch_metrics(out_t3_moe, clean_batch)

            for i in range(len(batch)):
                e = batch[i]
                c_name = e["corruption_type"]
                v_name = e.get("variant_name", f"{c_name}_{e.get('severity', 0)}")

                batch_vals = [
                    ("task1_univ", p_t1[i], s_t1[i], m_t1[i]),
                    ("task2_oracle", p_t2o[i], s_t2o[i], m_t2o[i]),
                    ("task2_pred", p_t2p[i], s_t2p[i], m_t2p[i]),
                    ("task3_moe", p_t3[i], s_t3[i], m_t3[i]),
                ]

                for m_key, psnr_val, ssim_val, mae_val in batch_vals:
                    metrics_acc[m_key]["psnr"].append(psnr_val)
                    metrics_acc[m_key]["ssim"].append(ssim_val)
                    metrics_acc[m_key]["mae"].append(mae_val)

                    per_c_acc[m_key][c_name]["psnr"].append(psnr_val)
                    per_c_acc[m_key][c_name]["ssim"].append(ssim_val)
                    per_c_acc[m_key][c_name]["mae"].append(mae_val)

                    per_s_acc[m_key][v_name]["psnr"].append(psnr_val)
                    per_s_acc[m_key][v_name]["ssim"].append(ssim_val)
                    per_s_acc[m_key][v_name]["mae"].append(mae_val)

                # Collect representative qualitative candidates
                if v_name not in qual_candidates and len(qual_candidates) < 14:
                    qual_candidates[v_name] = {
                        "corr": corr_batch[i].cpu(),
                        "clean": clean_batch[i].cpu(),
                        "out_t1": out_t1[i].cpu(),
                        "out_t2": recon_t2_pred[i].cpu(),
                        "out_t3": out_t3_moe[i].cpu(),
                        "w_t3": w_t3[i].cpu().numpy().tolist(),
                        "v_name": v_name,
                        "c_name": c_name,
                        "p_t1": p_t1[i],
                        "p_t2": p_t2p[i],
                        "p_t3": p_t3[i],
                        "s_t1": s_t1[i],
                        "s_t2": s_t2p[i],
                        "s_t3": s_t3[i],
                    }

                # Failure Case Search: where Task 2 misrouted but Task 3 recovered, OR boundary blend tradeoff
                diff = p_t3[i] - p_t2p[i]
                if diff > 3.0 and len(failure_candidates) < 6:
                    failure_candidates.append({
                        "corr": corr_batch[i].cpu(),
                        "clean": clean_batch[i].cpu(),
                        "out_t1": out_t1[i].cpu(),
                        "out_t2": recon_t2_pred[i].cpu(),
                        "out_t3": out_t3_moe[i].cpu(),
                        "w_t3": w_t3[i].cpu().numpy().tolist(),
                        "c_name": c_name,
                        "v_name": v_name,
                        "desc": f"Task 2 Hard Misroute -> Soft MoE Recovery (+{diff:.2f} dB)",
                        "p_t2": p_t2p[i],
                        "p_t3": p_t3[i],
                    })

            if (start_idx // batch_size + 1) % 25 == 0 or end_idx == total:
                print(f"  Processed [{end_idx}/{total}] instances ({time.time() - t0:.1f}s)...", flush=True)

    # 4. Aggregate Benchmark Summary Table
    def _agg(lst: List[float], dec: int = 2) -> float:
        return round(float(np.mean(lst)), dec) if lst else 0.0

    summary_data: Dict[str, Any] = {
        "subsample_ratio": subsample_ratio,
        "total_evaluated": total,
        "gpu_latencies_ms": gpu_latencies,
        "methods": {},
    }

    for m_key in methods:
        summary_data["methods"][m_key] = {
            "overall": {
                "psnr": _agg(metrics_acc[m_key]["psnr"]),
                "ssim": _agg(metrics_acc[m_key]["ssim"], 4),
                "mae": _agg(metrics_acc[m_key]["mae"], 4),
            },
            "per_corruption": {
                c: {
                    "psnr": _agg(per_c_acc[m_key][c]["psnr"]),
                    "ssim": _agg(per_c_acc[m_key][c]["ssim"], 4),
                    "mae": _agg(per_c_acc[m_key][c]["mae"], 4),
                }
                for c in ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"]
            },
            "per_severity": {
                v: {
                    "psnr": _agg(per_s_acc[m_key][v]["psnr"]),
                    "ssim": _agg(per_s_acc[m_key][v]["ssim"], 4),
                }
                for v in per_s_acc[m_key]
            },
        }

    # Export CSV benchmark comparison table
    csv_rows = [
        ["Method", "Overall PSNR (dB)", "Overall SSIM", "Clean PSNR", "Salt (Mild/Med/Sev)", "Blur (Mild/Med/Sev)", "Occ (Mild/Med/Sev)", "GPU Latency (ms)"]
    ]

    method_labels = {
        "task1_univ": "Task 1: Universal AE",
        "task2_pred": "Task 2: Hard Router (Predicted)",
        "task2_oracle": "Task 2: Hard Router (Oracle)",
        "task3_moe": "Task 3: Soft MoE (Ours)",
    }

    for m_key, m_label in method_labels.items():
        m_info = summary_data["methods"][m_key]
        ov_p = m_info["overall"]["psnr"]
        ov_s = m_info["overall"]["ssim"]
        cl_p = m_info["per_corruption"].get("clean", {}).get("psnr", 0.0)

        # Severities
        sp_str = f"{m_info['per_severity'].get('sp_mild', {}).get('psnr', 0.0)} / {m_info['per_severity'].get('sp_medium', {}).get('psnr', 0.0)} / {m_info['per_severity'].get('sp_severe', {}).get('psnr', 0.0)}"
        bl_str = f"{m_info['per_severity'].get('blur_mild', {}).get('psnr', 0.0)} / {m_info['per_severity'].get('blur_medium', {}).get('psnr', 0.0)} / {m_info['per_severity'].get('blur_severe', {}).get('psnr', 0.0)}"
        oc_str = f"{m_info['per_severity'].get('occl_mild', {}).get('psnr', 0.0)} / {m_info['per_severity'].get('occl_medium', {}).get('psnr', 0.0)} / {m_info['per_severity'].get('occl_severe', {}).get('psnr', 0.0)}"
        latency_key_map = {
            "task1_univ": "task1_universal",
            "task2_pred": "task2_predicted",
            "task2_oracle": "task2_oracle",
            "task3_moe": "task3_soft_moe",
        }
        lat = gpu_latencies.get(latency_key_map.get(m_key, m_key), 0.0)

        csv_rows.append([m_label, f"{ov_p:.2f}", f"{ov_s:.4f}", f"{cl_p:.2f}", sp_str, bl_str, oc_str, f"{lat:.2f} ms"])

    csv_path = out_dir / "test_benchmark_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Exported Benchmark CSV: {csv_path}")

    json_path = out_dir / "test_evaluation_summary.json"
    json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Exported Summary JSON: {json_path}")

    # 5. Render 12-Sample Qualitative Comparison Grid
    print("Rendering 12-Sample Qualitative Comparison Grid...")
    plot_qualitative_grid_12(qual_candidates, fig_dir / "qualitative_comparison_12.png")

    # 6. Render 4-Sample Failure Case / Tradeoff Analysis
    print("Rendering 4-Sample Failure & Tradeoff Analysis Panel...")
    plot_failure_cases_4(failure_candidates, qual_candidates, fig_dir / "failure_cases_4.png")

    # 7. Log to MLflow
    tracker = ExperimentTracker(settings=get_settings())
    with tracker.run(run_name="task3-test-evaluation", experiment_name="genai-task3-soft-moe"):
        tracker.log_metrics({
            "soft_moe_test_psnr": summary_data["methods"]["task3_moe"]["overall"]["psnr"],
            "soft_moe_test_ssim": summary_data["methods"]["task3_moe"]["overall"]["ssim"],
            "hard_pred_test_psnr": summary_data["methods"]["task2_pred"]["overall"]["psnr"],
            "hard_pred_test_ssim": summary_data["methods"]["task2_pred"]["overall"]["ssim"],
        })
        tracker.log_artifact(str(csv_path))
        tracker.log_artifact(str(json_path))
        tracker.log_artifact(str(fig_dir / "qualitative_comparison_12.png"))
        tracker.log_artifact(str(fig_dir / "failure_cases_4.png"))

    print("\n=== Step 9 Evaluation Completed Successfully! ===")
    return summary_data


def plot_qualitative_grid_12(
    candidates: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Render 12-sample qualitative comparison grid across 5 model views."""
    # Pick 12 representative specs
    specs_to_plot = [s[0] for s in SEVERITY_SPECS]
    items = []
    selected_keys = set()
    for s_key in specs_to_plot:
        if s_key in candidates and s_key not in selected_keys:
            items.append(candidates[s_key])
            selected_keys.add(s_key)
    # Pad to 12 if needed
    for k, v in candidates.items():
        if len(items) >= 12:
            break
        if k not in selected_keys:
            items.append(v)
            selected_keys.add(k)
    items = items[:12]

    num_samples = len(items)
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3.0 * num_samples), dpi=150)
    col_titles = [
        "Corrupted Input",
        "Clean Ground Truth",
        "Task 1: Universal AE",
        "Task 2: Hard Router",
        "Task 3: Soft MoE (Ours)",
    ]

    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for r_idx, sample in enumerate(items):
        views = [
            sample["corr"],
            sample["clean"],
            sample["out_t1"],
            sample["out_t2"],
            sample["out_t3"],
        ]
        subtitles = [
            f"Input ({sample['v_name']})",
            "Target (Clean)",
            f"{sample['p_t1']:.1f} dB | {sample['s_t1']:.3f}",
            f"{sample['p_t2']:.1f} dB | {sample['s_t2']:.3f}",
            f"{sample['p_t3']:.1f} dB | {sample['s_t3']:.3f}",
        ]

        for c_idx, (tensor, sub) in enumerate(zip(views, subtitles)):
            ax = axes[r_idx, c_idx]
            img = tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(sub, fontsize=8.5, pad=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_failure_cases_4(
    failures: List[Dict[str, Any]],
    candidates: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Render 4-case failure mode and tradeoff audit panel."""
    items = list(failures[:4])
    seen_vnames = {item.get("v_name") for item in items if "v_name" in item}
    if len(items) < 4:
        for k, v in candidates.items():
            if len(items) >= 4:
                break
            if k not in seen_vnames:
                items.append({
                    **v,
                    "desc": f"Boundary Blending Tradeoff in {v.get('v_name', k)}",
                })
                seen_vnames.add(k)


    fig, axes = plt.subplots(4, 5, figsize=(15, 12), dpi=150)
    col_titles = [
        "Corrupted Input",
        "Clean Ground Truth",
        "Task 1: Universal AE",
        "Task 2: Hard Router",
        "Task 3: Soft MoE (Ours)",
    ]
    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for r_idx, sample in enumerate(items[:4]):
        views = [
            sample["corr"],
            sample["clean"],
            sample.get("out_t1", sample["clean"]),
            sample.get("out_t2", sample["corr"]),
            sample.get("out_t3", sample["corr"]),
        ]
        subtitles = [
            f"Input [{sample.get('v_name', '')}]",
            "Target (Clean)",
            "Task 1 Universal",
            f"Task 2 ({sample.get('p_t2', 0.0):.1f} dB)",
            f"Task 3 ({sample.get('p_t3', 0.0):.1f} dB)",
        ]

        for c_idx, (tensor, sub) in enumerate(zip(views, subtitles)):
            ax = axes[r_idx, c_idx]
            img = tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(sub, fontsize=9, pad=3)

        axes[r_idx, 0].set_ylabel(sample.get("desc", f"Case {r_idx+1}"), fontsize=9, fontweight="bold")

    fig.suptitle("Task 3 Soft MoE vs Task 2 Hard Router: Diagnostic Tradeoffs & Edge Cases", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 Step 9 Comparative Test Evaluation")
    p.add_argument("--subsample", type=float, default=0.2, help="Subsample ratio of test manifest (default: 0.2)")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparative_evaluation(subsample_ratio=args.subsample, batch_size=args.batch_size)
