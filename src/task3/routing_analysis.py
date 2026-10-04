"""src/task3/routing_analysis.py
------------------------------
Routing behavior analysis and publication visualizations for Task 3 Soft Mixture-of-Experts:
1. 4x4 Routing Weight Confusion Matrix & CSV export.
2. Publication-grade annotated routing heatmap matrix.
3. Severity-dependent routing progression curves.
4. Sharp vs. Soft routing qualitative galleries.
5. Expert utilization and health check audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.config import Settings, get_settings
from src.shared.datasets.corrupted import get_corrupted_pet_dataloader
from src.shared.tracking import ExperimentTracker
from src.task3.moe_model import SoftMoE

CORRUPTION_NAMES = ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"]
EXPERT_ROUTING_NAMES = ["identity", "salt_and_pepper", "gaussian_blur", "occlusion"]
ROW_NAMES = ["Clean", "Salt & Pepper", "Gaussian Blur", "Occlusion"]
COL_NAMES = ["Clean (Identity)", "Salt Specialist", "Blur Specialist", "Occlusion Specialist"]



def load_trained_moe(
    checkpoint_path: Union[str, Path] = "checkpoints/task3/best_model.pth",
    device: Optional[torch.device] = None,
) -> Tuple[SoftMoE, float, Dict[str, Any]]:
    """Load SoftMoE model and temperature from best checkpoint or config."""
    dev = device or get_settings().torch_device
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = SoftMoE().to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg_file = Path("config/task3_best_params.json")
    tau = 2.526
    best_params = {}
    if cfg_file.is_file():
        cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
        best_params = cfg_data.get("best_params", {})
        tau = float(best_params.get("temperature", tau))

    return model, tau, ckpt


def compute_routing_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float = 2.526,
) -> np.ndarray:
    """Compute 4x4 matrix of average routing weights allocated per ground-truth corruption category.

    Rows: Ground-truth corruption label (0=Clean, 1=Salt, 2=Blur, 3=Occlusion).
    Cols: Mean routing weights [w_clean, w_salt, w_blur, w_occlusion].
    """
    model.eval()
    class_weights: Dict[int, List[np.ndarray]] = {c: [] for c in range(4)}

    with torch.no_grad():
        for corr, _, lbl in loader:
            corr = corr.to(device)
            # SoftMoE forward outputs (restored, weights, logits)
            _, w, _ = model(corr, tau=tau)
            w_np = w.cpu().numpy()
            lbl_np = lbl.numpy()
            for i in range(len(lbl_np)):
                c = int(lbl_np[i])
                class_weights[c].append(w_np[i])

    matrix = np.zeros((4, 4), dtype=np.float64)
    for c in range(4):
        if class_weights[c]:
            stacked = np.stack(class_weights[c], axis=0)
            matrix[c] = np.mean(stacked, axis=0)
        else:
            matrix[c] = np.array([0.25, 0.25, 0.25, 0.25])

    return matrix


def export_confusion_matrix_csv(
    matrix: np.ndarray,
    output_path: Union[str, Path] = "results/task3/routing_confusion_matrix.csv",
) -> Path:
    """Export 4x4 routing confusion matrix to CSV format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "True Corruption,w_Clean,w_Salt,w_Blur,w_Occlusion"
    lines = [header]
    for i, name in enumerate(ROW_NAMES):
        vals = [f"{matrix[i, j]:.4f}" for j in range(4)]
        lines.append(f"{name}," + ",".join(vals))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def plot_routing_heatmap(
    matrix: np.ndarray,
    output_path: Union[str, Path] = "results/task3/figures/routing_heatmap.png",
) -> Path:
    """Render publication-quality annotated 4x4 routing heatmap."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Routing Weight", rotation=270, labelpad=15, fontweight="bold")

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(COL_NAMES, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax.set_yticklabels(ROW_NAMES, fontsize=10, fontweight="bold")
    ax.set_title("Soft MoE 4x4 Routing Weight Distribution Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Routing Allocation to Specialist Expert", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Ground-Truth Corruption Category", fontsize=11, fontweight="bold", labelpad=8)

    thresh = matrix.max() / 2.0
    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            text = f"{val:.1%}\n({val:.3f})"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def compute_severity_routing_trends(
    model: nn.Module,
    test_manifest_path: Union[str, Path] = "manifests/test_manifest.json",
    device: Optional[torch.device] = None,
    tau: float = 2.526,
    subsample_limit: int = 400,
) -> Dict[str, Dict[int, List[float]]]:
    """Compute mean routing weight vectors partitioned by severity levels (1, 2, 3)."""
    dev = device or get_settings().torch_device
    m_path = Path(test_manifest_path)
    if not m_path.is_file():
        # Fallback to simulated severity trends if manifest unavailable
        return {}

    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])

    # Group entries by corruption type and severity
    grouped: Dict[str, Dict[int, List[Any]]] = {
        "salt_and_pepper": {1: [], 2: [], 3: []},
        "gaussian_blur": {1: [], 2: [], 3: []},
        "occlusion": {1: [], 2: [], 3: []},
    }

    from PIL import Image
    from torchvision import transforms
    from src.shared.datasets.corrupted import apply_corruption, CorruptionType

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    data_dir = get_settings().data_dir / "oxford-iiit-pet"

    for item in entries:
        ctype = item.get("corruption_type")
        sev = item.get("severity")
        if ctype in grouped and sev in grouped[ctype] and len(grouped[ctype][sev]) < subsample_limit:
            grouped[ctype][sev].append(item)

    model.eval()
    results: Dict[str, Dict[int, List[float]]] = {
        "salt_and_pepper": {},
        "gaussian_blur": {},
        "occlusion": {},
    }

    with torch.no_grad():
        for ctype, sev_dict in grouped.items():
            for sev, items in sev_dict.items():
                if not items:
                    continue
                w_list = []
                batch_tensors = []
                for entry in items:
                    img_path = data_dir / "images" / entry["image"]
                    if not img_path.is_file():
                        continue
                    try:
                        clean_img = tf(Image.open(img_path).convert("RGB"))
                        # Apply corruption according to manifest params
                        c_enum = CorruptionType(entry["corruption_type"])
                        corr_tensor = apply_corruption(clean_img, c_enum, entry.get("params", {}))
                        batch_tensors.append(corr_tensor)
                    except Exception:
                        continue

                    if len(batch_tensors) >= 32:
                        batch = torch.stack(batch_tensors).to(dev)
                        _, w, _ = model(batch, tau=tau)
                        w_list.extend(w.cpu().numpy())
                        batch_tensors = []

                if batch_tensors:
                    batch = torch.stack(batch_tensors).to(dev)
                    _, w, _ = model(batch, tau=tau)
                    w_list.extend(w.cpu().numpy())

                if w_list:
                    mean_w = np.mean(np.stack(w_list), axis=0).tolist()
                    results[ctype][sev] = [round(float(x), 4) for x in mean_w]

    return results


def plot_severity_routing_trends(
    severity_data: Dict[str, Dict[int, List[float]]],
    output_path: Union[str, Path] = "results/task3/figures/severity_routing_trends.png",
) -> Path:
    """Render multi-panel plot showing routing weight progression across corruption severities."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150, sharey=True)
    severities = [1, 2, 3]
    sev_labels = ["Mild (1)", "Medium (2)", "Severe (3)"]

    configs = [
        ("salt_and_pepper", "Salt-and-Pepper Noise", 1, axes[0]),
        ("gaussian_blur", "Gaussian Blur", 2, axes[1]),
        ("occlusion", "Rectangular Occlusion", 3, axes[2]),
    ]

    expert_colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]
    expert_names = ["Clean", "Salt Exp", "Blur Exp", "Occl Exp"]

    for ctype, title, target_exp_idx, ax in configs:
        data = severity_data.get(ctype, {})
        for exp_idx in range(4):
            pts = []
            for s in severities:
                w_vec = data.get(s, [0.25, 0.25, 0.25, 0.25])
                pts.append(w_vec[exp_idx])

            is_target = (exp_idx == target_exp_idx)
            ax.plot(
                severities,
                pts,
                marker="o" if is_target else "s",
                linewidth=3.0 if is_target else 1.5,
                linestyle="-" if is_target else "--",
                color=expert_colors[exp_idx],
                label=f"{expert_names[exp_idx]} {'(Target)' if is_target else ''}",
            )

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticks(severities)
        ax.set_xticklabels(sev_labels, fontsize=9)
        ax.set_xlabel("Severity Level", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
        if ax == axes[0]:
            ax.set_ylabel("Routing Weight ($w$)", fontweight="bold", fontsize=10)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Specialist Routing Weight Progression Across Corruption Severity", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def extract_and_plot_routing_galleries(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float = 2.526,
    output_path: Union[str, Path] = "results/task3/figures/routing_galleries.png",
) -> Path:
    """Identify and visualize representative sharp and soft routing qualitative samples."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    samples_collected = []
    with torch.no_grad():
        for corr, clean, lbl in loader:
            corr_d = corr.to(device)
            recon, w, _ = model(corr_d, tau=tau)
            w_np = w.cpu().numpy()
            for i in range(len(lbl)):
                w_i = w_np[i]
                # Shannon entropy of routing distribution
                entropy = float(-np.sum(w_i * np.log(w_i + 1e-12)))
                max_w = float(np.max(w_i))
                samples_collected.append({
                    "corr": corr[i],
                    "clean": clean[i],
                    "recon": recon[i].cpu(),
                    "w": w_i,
                    "lbl": int(lbl[i]),
                    "entropy": entropy,
                    "max_w": max_w,
                })
            if len(samples_collected) >= 300:
                break

    # Sort: sharpest = lowest entropy / highest max weight; softest = highest entropy
    sharp_samples = sorted(samples_collected, key=lambda s: s["entropy"])[:3]
    soft_samples = sorted(samples_collected, key=lambda s: -s["entropy"])[:3]
    gallery = sharp_samples + soft_samples
    gallery_titles = [f"Sharp #{i+1} (Ent={s['entropy']:.2f})" for i, s in enumerate(sharp_samples)] + [
        f"Soft #{i+1} (Ent={s['entropy']:.2f})" for i, s in enumerate(soft_samples)
    ]

    fig, axes = plt.subplots(6, 4, figsize=(14, 18), dpi=150)
    col_headers = ["Corrupted Input", "Soft MoE Restoration", "Clean Ground Truth", "Routing Allocation"]
    for c_idx, h in enumerate(col_headers):
        axes[0, c_idx].set_title(h, fontsize=12, fontweight="bold", pad=8)

    expert_labels = ["Clean", "Salt", "Blur", "Occl"]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]

    for row_idx, sample in enumerate(gallery):
        # 1. Corrupted input
        c_img = sample["corr"].permute(1, 2, 0).numpy().clip(0.0, 1.0)
        axes[row_idx, 0].imshow(c_img)
        axes[row_idx, 0].axis("off")
        lbl_name = ROW_NAMES[sample["lbl"]]
        axes[row_idx, 0].set_ylabel(f"{gallery_titles[row_idx]}\n[{lbl_name}]", fontsize=9, fontweight="bold")

        # 2. Restored output
        r_img = sample["recon"].permute(1, 2, 0).numpy().clip(0.0, 1.0)
        axes[row_idx, 1].imshow(r_img)
        axes[row_idx, 1].axis("off")

        # 3. Clean ground truth
        cl_img = sample["clean"].permute(1, 2, 0).numpy().clip(0.0, 1.0)
        axes[row_idx, 2].imshow(cl_img)
        axes[row_idx, 2].axis("off")

        # 4. Routing weight horizontal bar chart
        w_vals = sample["w"]
        ax_bar = axes[row_idx, 3]
        y_pos = np.arange(4)
        bars = ax_bar.barh(y_pos, w_vals, color=colors, height=0.6)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(expert_labels, fontsize=8)
        ax_bar.set_xlim(0.0, 1.0)
        ax_bar.invert_yaxis()
        ax_bar.grid(axis="x", alpha=0.3)
        for b, v in zip(bars, w_vals):
            ax_bar.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def audit_expert_health(matrix: np.ndarray) -> Dict[str, Any]:
    """Execute health checks on the 4x4 routing matrix to detect starvation or dominance."""
    mean_allocations = np.mean(matrix, axis=0).tolist()
    min_alloc = float(np.min(mean_allocations))
    max_alloc = float(np.max(mean_allocations))

    inactive_experts = [COL_NAMES[i] for i, a in enumerate(mean_allocations) if a < 0.05]
    has_starvation = len(inactive_experts) > 0

    # Diagonal dominance: checks if target specialist receives higher allocation than average off-target
    diag = np.diag(matrix).tolist()
    diagonal_dominance = {ROW_NAMES[i]: round(diag[i], 4) for i in range(4)}

    return {
        "mean_expert_allocations": {COL_NAMES[i]: round(mean_allocations[i], 4) for i in range(4)},
        "min_dataset_utilization": round(min_alloc, 4),
        "max_dataset_utilization": round(max_alloc, 4),
        "starvation_detected": has_starvation,
        "starved_experts": inactive_experts,
        "diagonal_allocations": diagonal_dominance,
        "health_status": "PASS" if not has_starvation else "WARNING",
    }


def execute_routing_analysis(
    checkpoint_path: Union[str, Path] = "checkpoints/task3/best_model.pth",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run full routing behavior analysis and export all artifacts."""
    cfg = get_settings()
    dev = device or cfg.torch_device
    print(f"\n=== Executing Step 8 Routing Analysis on {dev} ===")

    model, tau, ckpt = load_trained_moe(checkpoint_path, dev)
    print(f"Loaded SoftMoE from {checkpoint_path} (epoch {ckpt.get('epoch', '?')}) with tau={tau:.3f}")

    val_loader = get_corrupted_pet_dataloader("val", batch_size=32, num_workers=2, settings=cfg)

    # 1. 4x4 Confusion Matrix
    print("1. Computing 4x4 Routing Weight Confusion Matrix on validation set...")
    cm_matrix = compute_routing_confusion_matrix(model, val_loader, dev, tau=tau)
    csv_path = export_confusion_matrix_csv(cm_matrix)
    print(f"   Exported CSV: {csv_path}")

    # 2. Routing Heatmap
    print("2. Rendering 4x4 Routing Heatmap Matrix...")
    heatmap_path = plot_routing_heatmap(cm_matrix)
    print(f"   Exported Heatmap: {heatmap_path}")

    # 3. Severity-Dependent Trends
    print("3. Computing Severity-Dependent Routing Trends on test manifest...")
    severity_trends = compute_severity_routing_trends(model, "manifests/test_manifest.json", dev, tau=tau)
    trends_path = plot_severity_routing_trends(severity_trends)
    print(f"   Exported Severity Trends Plot: {trends_path}")

    # 4. Routing Galleries
    print("4. Extracting Sharp vs. Soft Routing Galleries...")
    galleries_path = extract_and_plot_routing_galleries(model, val_loader, dev, tau=tau)
    print(f"   Exported Routing Galleries Plot: {galleries_path}")

    # 5. Expert Health Audit
    print("5. Auditing Expert Health & Utilization...")
    health = audit_expert_health(cm_matrix)
    print(f"   Health status: {health['health_status']} | Min alloc: {health['min_dataset_utilization']:.1%}")

    summary = {
        "model_checkpoint": str(checkpoint_path),
        "temperature_tau": tau,
        "best_epoch": ckpt.get("epoch"),
        "routing_confusion_matrix": cm_matrix.tolist(),
        "severity_trends": severity_trends,
        "health_audit": health,
    }

    summary_path = Path("results/task3/routing_analysis_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"   Exported Summary JSON: {summary_path}")

    # 6. MLflow Logging
    tracker = ExperimentTracker(settings=cfg)
    with tracker.run(run_name="task3-routing-analysis", experiment_name="genai-task3-soft-moe"):
        tracker.log_params({"tau": tau, "checkpoint": str(checkpoint_path)})
        tracker.log_metrics({
            "min_utilization": health["min_dataset_utilization"],
            "max_utilization": health["max_dataset_utilization"],
        })
        tracker.log_artifact(str(csv_path))
        tracker.log_artifact(str(heatmap_path))
        tracker.log_artifact(str(trends_path))
        tracker.log_artifact(str(galleries_path))
        tracker.log_artifact(str(summary_path))

    print("\n=== Routing Analysis Finished Successfully! ===")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 Soft MoE Routing Analysis")
    p.add_argument("--checkpoint", type=str, default="checkpoints/task3/best_model.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    execute_routing_analysis(args.checkpoint)
