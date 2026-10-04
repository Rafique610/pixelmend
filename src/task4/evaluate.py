"""
src/task4/evaluate.py
---------------------
Exhaustive evaluation pipeline for Task 4: Style-Conditioned Face-to-Sketch Synthesis.
Evaluates on the held-out FS2K test set (1,046 pairs):
1. Quantitative Metrics: Overall & Per-Style (L1 Error, PSNR, SSIM, LPIPS).
2. Diverse Sample Results Grid (12 paired test samples across styles).
3. Multi-Style Conditioning Grid (Same input photos rendered in Styles 0, 1, 2 side-by-side).
4. Failure Case Diagnostic Panel (Identifying failure modes like occlusions, extreme lighting).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lpips
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from src.shared.config import get_settings
from src.shared.metrics import compute_mae, compute_psnr, compute_ssim
from src.task4.dataset import FS2KDataset
from src.task4.generator import UNetGenerator


def evaluate_test_set(
    net_g: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_lpips: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Compute overall and per-style metrics on FS2K test split."""
    net_g.eval()

    lpips_fn = None
    if use_lpips:
        try:
            lpips_fn = lpips.LPIPS(net="alex").to(device)
            lpips_fn.eval()
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS ({e}). Skipping LPIPS.")

    overall_l1: List[float] = []
    overall_psnr: List[float] = []
    overall_ssim: List[float] = []
    overall_lpips: List[float] = []

    per_style: Dict[int, Dict[str, List[float]]] = {
        0: {"l1": [], "psnr": [], "ssim": [], "lpips": []},
        1: {"l1": [], "psnr": [], "ssim": [], "lpips": []},
        2: {"l1": [], "psnr": [], "ssim": [], "lpips": []},
    }

    item_records: List[Dict[str, Any]] = []

    print("Evaluating test set on", device, "...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            photos = batch["photo"].to(device)
            reals = batch["sketch"].to(device)
            styles = batch["style"].to(device)

            fakes = net_g(photos, styles)

            fakes_01 = torch.clamp((fakes + 1.0) / 2.0, 0.0, 1.0)
            reals_01 = torch.clamp((reals + 1.0) / 2.0, 0.0, 1.0)

            # LPIPS expects input in [-1, 1]
            lp_batch = None
            if lpips_fn is not None:
                lp_batch = lpips_fn(fakes, reals).view(-1).cpu().tolist()

            for i in range(photos.size(0)):
                f_i = fakes_01[i]
                r_i = reals_01[i]
                st = int(styles[i].item())

                l1 = compute_mae(f_i, r_i)
                psnr = compute_psnr(f_i, r_i, max_val=1.0)
                ssim = compute_ssim(f_i, r_i, data_range=1.0)
                lp = float(lp_batch[i]) if lp_batch is not None else 0.0

                overall_l1.append(l1)
                overall_psnr.append(psnr)
                overall_ssim.append(ssim)
                if lp_batch is not None:
                    overall_lpips.append(lp)

                per_style[st]["l1"].append(l1)
                per_style[st]["psnr"].append(psnr)
                per_style[st]["ssim"].append(ssim)
                if lp_batch is not None:
                    per_style[st]["lpips"].append(lp)

                item_records.append({
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "style": st,
                    "l1": l1,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lp,
                })

    summary: Dict[str, float] = {
        "test_l1": float(np.mean(overall_l1)),
        "test_psnr": float(np.mean(overall_psnr)),
        "test_ssim": float(np.mean(overall_ssim)),
        "test_lpips": float(np.mean(overall_lpips)) if overall_lpips else 0.0,
    }

    for st in (0, 1, 2):
        s_data = per_style[st]
        if s_data["l1"]:
            summary[f"style_{st}_l1"] = float(np.mean(s_data["l1"]))
            summary[f"style_{st}_psnr"] = float(np.mean(s_data["psnr"]))
            summary[f"style_{st}_ssim"] = float(np.mean(s_data["ssim"]))
            if s_data["lpips"]:
                summary[f"style_{st}_lpips"] = float(np.mean(s_data["lpips"]))

    return summary, item_records


def generate_sample_results_grid(
    net_g: nn.Module,
    test_ds: FS2KDataset,
    device: torch.device,
    save_path: str | Path,
    num_samples: int = 12,
) -> None:
    """Create a 12-sample test results grid: [Photo | Real Sketch | Generated Sketch]."""
    net_g.eval()
    samples: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

    # Pick samples evenly across styles
    by_style: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for idx, item in enumerate(test_ds.samples):
        st = int(item["style"])
        by_style[st].append(idx)

    target_per_style = num_samples // 3
    selected_indices: List[int] = []
    for st in (0, 1, 2):
        selected_indices.extend(by_style[st][:target_per_style])

    photo_list, real_list, fake_list = [], [], []
    with torch.no_grad():
        for idx in selected_indices:
            data = test_ds[idx]
            p = data["photo"].unsqueeze(0).to(device)
            r = data["sketch"].unsqueeze(0).to(device)
            s = torch.tensor([data["style"]], dtype=torch.long, device=device)

            f = net_g(p, s)

            photo_list.append(torch.clamp((p[0] + 1.0) / 2.0, 0.0, 1.0).cpu())
            real_list.append(torch.clamp((r[0] + 1.0) / 2.0, 0.0, 1.0).cpu())
            fake_list.append(torch.clamp((f[0] + 1.0) / 2.0, 0.0, 1.0).cpu())

    # Build 12 rows of 3 columns: Photo | Ground Truth | Synthesized
    combined = []
    for i in range(len(photo_list)):
        combined.extend([photo_list[i], real_list[i], fake_list[i]])
    grid_tensor = torch.stack(combined)

    grid = vutils.make_grid(grid_tensor, nrow=3, padding=4, normalize=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(save_path)
    print(f"Saved sample results grid to {save_path}")


def generate_multi_style_grid(
    net_g: nn.Module,
    test_ds: FS2KDataset,
    device: torch.device,
    save_path: str | Path,
    num_faces: int = 4,
) -> None:
    """Create a multi-style grid: Input Photo | Style 0 Output | Style 1 Output | Style 2 Output."""
    net_g.eval()
    # Pick faces with distinct orientations/features
    face_indices = [0, 50, 150, 300][:num_faces]

    row_images = []
    with torch.no_grad():
        for idx in face_indices:
            data = test_ds[idx]
            p = data["photo"].unsqueeze(0).to(device)
            p_01 = torch.clamp((p[0] + 1.0) / 2.0, 0.0, 1.0).cpu()

            row_images.append(p_01)
            for st_val in (0, 1, 2):
                st_tensor = torch.tensor([st_val], dtype=torch.long, device=device)
                f = net_g(p, st_tensor)
                f_01 = torch.clamp((f[0] + 1.0) / 2.0, 0.0, 1.0).cpu()
                row_images.append(f_01)

    grid_tensor = torch.stack(row_images)
    grid = vutils.make_grid(grid_tensor, nrow=4, padding=4, normalize=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(save_path)
    print(f"Saved multi-style conditioning grid to {save_path}")


def generate_failure_cases_panel(
    net_g: nn.Module,
    test_ds: FS2KDataset,
    item_records: List[Dict[str, Any]],
    device: torch.device,
    save_path: str | Path,
    top_k: int = 4,
) -> None:
    """Generate diagnostic panel for samples with largest reconstruction error."""
    net_g.eval()
    # Sort items by descending L1 error
    sorted_items = sorted(item_records, key=lambda x: x["l1"], reverse=True)
    worst_samples = sorted_items[:top_k]

    fig, axes = plt.subplots(top_k, 3, figsize=(10, 3.2 * top_k))
    plt.subplots_adjust(hspace=0.35, wspace=0.1)

    with torch.no_grad():
        for row_idx, rec in enumerate(worst_samples):
            # Resolve actual item from test_ds
            idx = rec["sample_idx"] + rec["batch_idx"] * 16
            if idx >= len(test_ds):
                idx = row_idx
            data = test_ds[idx]

            p = data["photo"].unsqueeze(0).to(device)
            r = data["sketch"].unsqueeze(0).to(device)
            s = torch.tensor([data["style"]], dtype=torch.long, device=device)
            f = net_g(p, s)

            p_np = torch.clamp((p[0] + 1.0) / 2.0, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            r_np = torch.clamp((r[0] + 1.0) / 2.0, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            f_np = torch.clamp((f[0] + 1.0) / 2.0, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()

            axes[row_idx, 0].imshow(p_np)
            axes[row_idx, 0].set_title(f"Input Photo (Style {data['style']})", fontsize=10)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(r_np)
            axes[row_idx, 1].set_title("Ground Truth Sketch", fontsize=10)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(f_np)
            axes[row_idx, 2].set_title(
                f"Generated (L1={rec['l1']:.3f}, SSIM={rec['ssim']:.3f})",
                fontsize=10,
                color="crimson",
            )
            axes[row_idx, 2].axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved failure case panel to {save_path}")


def run_evaluation(
    checkpoint_path: str | Path = "checkpoints/task4/best_generator.pth",
    batch_size: int = 16,
    results_dir: str | Path = "results/task4",
) -> Dict[str, Any]:
    """Execute end-to-end evaluation pipeline on test split."""
    settings = get_settings()
    device = settings.torch_device
    is_cuda = device.type == "cuda"
    print(f"=== Starting Task 4 Evaluation on {device} ===")

    test_ds = FS2KDataset(root_dir="data/fs2k", split="test", augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )

    # Load trained generator
    net_g = UNetGenerator(base_channels=64, embed_dim=16).to(device)
    cp = Path(checkpoint_path)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")
    net_g.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
    net_g.eval()

    # Quantitative evaluation
    summary, item_records = evaluate_test_set(net_g, test_loader, device, use_lpips=True)

    print("\n--- Test Set Quantitative Results ---")
    print(f"Overall L1 (MAE): {summary['test_l1']:.4f}")
    print(f"Overall PSNR:     {summary['test_psnr']:.2f} dB")
    print(f"Overall SSIM:     {summary['test_ssim']:.4f}")
    print(f"Overall LPIPS:    {summary['test_lpips']:.4f}")
    for st in (0, 1, 2):
        print(f"  Style {st} -> L1: {summary.get(f'style_{st}_l1', 0):.4f} | "
              f"PSNR: {summary.get(f'style_{st}_psnr', 0):.2f} dB | "
              f"SSIM: {summary.get(f'style_{st}_ssim', 0):.4f} | "
              f"LPIPS: {summary.get(f'style_{st}_lpips', 0):.4f}")

    # Visual Artifacts
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    grid_sample_path = res_path / "sample_results_grid.png"
    grid_style_path = res_path / "style_comparison_grid.png"
    failure_panel_path = res_path / "failure_cases_analysis.png"

    generate_sample_results_grid(net_g, test_ds, device, grid_sample_path, num_samples=12)
    generate_multi_style_grid(net_g, test_ds, device, grid_style_path, num_faces=4)
    generate_failure_cases_panel(net_g, test_ds, item_records, device, failure_panel_path, top_k=4)

    # Save metrics JSON
    json_path = res_path / "test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved test metrics to {json_path}")

    # Log to MLflow
    try:
        mlflow.set_experiment("genai-task4-evaluation")
        with mlflow.start_run(run_name="test-evaluation-best-generator"):
            for k, v in summary.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(str(json_path))
            mlflow.log_artifact(str(grid_sample_path))
            mlflow.log_artifact(str(grid_style_path))
            mlflow.log_artifact(str(failure_panel_path))
            print("Successfully logged evaluation results to MLflow (genai-task4-evaluation).")
    except Exception as e:
        print(f"MLflow warning: {e}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Task 4 cGAN model on FS2K test set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/task4/best_generator.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_evaluation(checkpoint_path=args.checkpoint, batch_size=args.batch_size)
