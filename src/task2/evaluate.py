"""
src/task2/evaluate.py
---------------------
Comparative Test Set Evaluation for Task 2: Hard-Routing Restoration System.
Evaluates Oracle vs Predicted routing against Task 1 Universal Autoencoder baseline.
Audits the 4 classifier-induced misrouting failure modes and produces visual galleries.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_msssim
import torch
from PIL import Image
from torchvision import transforms as T

from src.shared.config import get_settings
from src.shared.corruptions import apply_corruption
from src.shared.tracking import ExperimentTracker
from src.shared.visualization import plot_qualitative_quads
from src.task2.router import HardRouter, load_hard_router

SPECS = [
    ("clean", "Clean / Uncorrupted"), ("sp_mild", "SP (Mild, p=0.03)"),
    ("sp_medium", "SP (Med, p=0.08)"), ("sp_severe", "SP (Severe, p=0.15)"),
    ("blur_mild", "Blur (Mild, k=3)"), ("blur_medium", "Blur (Med, k=5)"),
    ("blur_severe", "Blur (Severe, k=7)"), ("occl_mild", "Occl (Mild, 1 box)"),
    ("occl_medium", "Occl (Med, 2 boxes)"), ("occl_severe", "Occl (Severe, 3 boxes)"),
]


def load_clean_images(images_dir: Path, image_names: List[str]) -> Dict[str, torch.Tensor]:
    """Pre-cache clean test images into RAM as normalized (3, 128, 128) float32 tensors."""
    tfm = T.Compose([T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()])
    cache: Dict[str, torch.Tensor] = {}
    unique_names = sorted(list(set(image_names)))
    t0 = time.time()
    print(f"Pre-caching {len(unique_names)} unique clean test images into memory...", flush=True)
    for i, name in enumerate(unique_names):
        with Image.open(images_dir / name) as raw:
            cache[name] = tfm(raw.convert("RGB"))
        if (i + 1) % 1000 == 0 or (i + 1) == len(unique_names):
            print(f"  [{i+1}/{len(unique_names)}] cached ({time.time() - t0:.1f}s)", flush=True)
    return cache


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ssim = pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=False)
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    mae = torch.mean(torch.abs(pred - target), dim=[1, 2, 3])
    return psnr, ssim, mae


def evaluate_test_set(
    router: HardRouter,
    entries: List[Dict[str, Any]],
    clean_cache: Dict[str, torch.Tensor],
    batch_size: int = 64,
    device: str = "cpu",
    heartbeat_interval: int = 25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute comparative batched evaluation (Oracle vs Predicted) and collect failure cases."""
    router.eval()
    pred_m = {"psnr": [], "ssim": [], "mae": []}
    orac_m = {"psnr": [], "ssim": [], "mae": []}
    per_c_pred = defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []})
    per_c_orac = defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []})
    per_s_pred = defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []})
    per_s_orac = defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []})

    qual_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    failure_candidates: Dict[str, List[Dict[str, Any]]] = {
        "false_clean": [], "sp_to_blur": [], "blur_to_occl": [], "clean_to_corr": [],
    }

    total, t_start = len(entries), time.time()
    print(f"\nStarting evaluation on {total} instances (batch size: {batch_size})...", flush=True)

    with torch.no_grad():
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = entries[start_idx:end_idx]

            clean_list = [clean_cache[e["image"]] for e in batch]
            corr_list = [apply_corruption(clean, e["corruption_type"], e.get("params", {})) for clean, e in zip(clean_list, batch)]
            true_labels = torch.tensor([e["corruption_label"] for e in batch], device=device, dtype=torch.long)

            corr_tensor = torch.stack(corr_list).to(device)
            clean_tensor = torch.stack(clean_list).to(device)

            # 1. Predicted Forward Pass
            res_pred = router(corr_tensor)
            rest_pred = res_pred["reconstructed"]
            decisions = torch.tensor(res_pred["routing_decision"], device=device, dtype=torch.long)

            # 2. Oracle Restoration (reuse matching predictions to avoid redundant compute)
            rest_orac = rest_pred.clone()
            mismatch_mask = decisions != true_labels
            if mismatch_mask.any():
                sub_corr = corr_tensor[mismatch_mask]
                sub_labels = true_labels[mismatch_mask]
                res_orac_sub = router(sub_corr, oracle_labels=sub_labels)
                rest_orac[mismatch_mask] = res_orac_sub["reconstructed"]

            p_psnr, p_ssim, p_mae = _compute_metrics(rest_pred, clean_tensor)
            o_psnr, o_ssim, o_mae = _compute_metrics(rest_orac, clean_tensor)

            for k, e in enumerate(batch):
                c_type, s_name = e["corruption_type"], e.get("variant_name", e["corruption_type"])
                pp, ps, pm = float(p_psnr[k]), float(p_ssim[k]), float(p_mae[k])
                op, os, om = float(o_psnr[k]), float(o_ssim[k]), float(o_mae[k])

                for store, p, s, m in ((pred_m, pp, ps, pm), (orac_m, op, os, om),
                                       (per_c_pred[c_type], pp, ps, pm), (per_c_orac[c_type], op, os, om),
                                       (per_s_pred[s_name], pp, ps, pm), (per_s_orac[s_name], op, os, om)):
                    store["psnr"].append(p)
                    store["ssim"].append(s)
                    store["mae"].append(m)

                if len(qual_map[s_name]) < 3:
                    qual_map[s_name].append({
                        "image": e["image"], "variant_name": s_name, "corruption_type": c_type,
                        "clean": clean_list[k].cpu(), "corrupted": corr_list[k].cpu(),
                        "restored": rest_pred[k].cpu(), "psnr": pp, "ssim": ps, "mae": pm,
                    })

                t_lbl, p_lbl = int(true_labels[k]), int(decisions[k])
                rec_fail = {
                    "image": e["image"], "variant_name": s_name, "corruption_type": c_type,
                    "clean": clean_list[k].cpu(), "corrupted": corr_list[k].cpu(),
                    "restored": rest_pred[k].cpu(), "psnr": pp, "ssim": ps, "mae": pm,
                    "true_label": t_lbl, "pred_label": p_lbl,
                }
                if t_lbl != 0 and p_lbl == 0: failure_candidates["false_clean"].append(rec_fail)
                elif t_lbl == 1 and p_lbl == 2: failure_candidates["sp_to_blur"].append(rec_fail)
                elif t_lbl == 2 and p_lbl == 3: failure_candidates["blur_to_occl"].append(rec_fail)
                elif t_lbl == 0 and p_lbl != 0: failure_candidates["clean_to_corr"].append(rec_fail)

            batch_idx = end_idx // batch_size
            if batch_idx % heartbeat_interval == 0 or end_idx == total:
                elapsed = time.time() - t_start
                rate = end_idx / max(1e-4, elapsed)
                eta_s = (total - end_idx) / max(1e-4, rate)
                print(f"  [Eval {end_idx:5d}/{total} ({end_idx/total*100:4.1f}%)] Rate: {rate:4.1f} img/s | "
                      f"ETA: {int(eta_s//60):02d}m {int(eta_s%60):02d}s | "
                      f"Pred PSNR: {np.mean(pred_m['psnr']):5.2f} dB | Orac PSNR: {np.mean(orac_m['psnr']):5.2f} dB", flush=True)

    summary = {
        "overall_predicted": {k: float(np.mean(v)) for k, v in pred_m.items()},
        "overall_oracle": {k: float(np.mean(v)) for k, v in orac_m.items()},
        "per_corruption_predicted": {c: {k: float(np.mean(vals)) for k, vals in d.items()} for c, d in per_c_pred.items()},
        "per_corruption_oracle": {c: {k: float(np.mean(vals)) for k, vals in d.items()} for c, d in per_c_orac.items()},
        "per_severity_predicted": {s: {k: float(np.mean(vals)) for k, vals in d.items()} for s, d in per_s_pred.items()},
        "per_severity_oracle": {s: {k: float(np.mean(vals)) for k, vals in d.items()} for s, d in per_s_orac.items()},
        "total_evaluated": total, "runtime_seconds": float(time.time() - t_start),
    }

    # Select representative failure cases
    audit_failures: List[Dict[str, Any]] = []
    diag_labels = [
        ("false_clean", "Failure Mode 1: False Clean Bypass (Corrupted -> Clean Bypass)"),
        ("sp_to_blur", "Failure Mode 2: Cross-Contamination (Salt&Pepper -> Blur Specialist)"),
        ("blur_to_occl", "Failure Mode 3: Inpainting Artifact (Gaussian Blur -> Occlusion Specialist)"),
        ("clean_to_corr", "Failure Mode 4: Over-Processing (Clean Image -> Specialist)"),
    ]
    for key, desc in diag_labels:
        cand_list = failure_candidates[key]
        if cand_list:
            cand_list.sort(key=lambda x: x["psnr"])
            audit_failures.append({**cand_list[0], "notes": desc, "label_str": f"{cand_list[0]['image']}\n{desc}"})

    return summary, [item for group in qual_map.values() for item in group], audit_failures


def format_tables_and_export(summary: Dict[str, Any], task1_summary_path: Path, out_dir: Path) -> str:
    """Generate Markdown and CSV comparative tables against Task 1 Universal AE."""
    t1_ov, t1_corr = {"psnr": 20.16, "ssim": 0.5935, "mae": 0.0742}, {}
    if task1_summary_path.is_file():
        with open(task1_summary_path, "r", encoding="utf-8") as f:
            t1_data = json.load(f)
            t1_ov, t1_corr = t1_data.get("overall", t1_ov), t1_data.get("per_corruption", {})

    lines = [
        "### System-Level Comparison: Task 1 Universal AE vs Task 2 Hard-Routing\n",
        "| Evaluation Domain | Metric | Task 1: Universal AE | Task 2: Oracle Routing | Task 2: Predicted Routing | Routing Gap (Oracle - Pred) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    p_ov, o_ov = summary["overall_predicted"], summary["overall_oracle"]
    rows = [
        ["Overall Mean", "PSNR (dB)", f"{t1_ov['psnr']:.2f}", f"{o_ov['psnr']:.2f}", f"{p_ov['psnr']:.2f}", f"{o_ov['psnr'] - p_ov['psnr']:+.2f}"],
        ["Overall Mean", "SSIM", f"{t1_ov['ssim']:.4f}", f"{o_ov['ssim']:.4f}", f"{p_ov['ssim']:.4f}", f"{o_ov['ssim'] - p_ov['ssim']:+.4f}"],
        ["Overall Mean", "MAE", f"{t1_ov['mae']:.4f}", f"{o_ov['mae']:.4f}", f"{p_ov['mae']:.4f}", f"{p_ov['mae'] - o_ov['mae']:+.4f}"],
    ]
    for c in ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"]:
        c_p, c_o = summary["per_corruption_predicted"].get(c, {}), summary["per_corruption_oracle"].get(c, {})
        c_t1 = t1_corr.get(c, {"psnr": 0.0, "ssim": 0.0, "mae": 0.0})
        name = c.replace("_", " ").title()
        rows.append([name, "PSNR (dB)", f"{c_t1['psnr']:.2f}", f"{c_o.get('psnr', 0):.2f}", f"{c_p.get('psnr', 0):.2f}", f"{c_o.get('psnr', 0) - c_p.get('psnr', 0):+.2f}"])
        rows.append([name, "SSIM", f"{c_t1['ssim']:.4f}", f"{c_o.get('ssim', 0):.4f}", f"{c_p.get('ssim', 0):.4f}", f"{c_o.get('ssim', 0) - c_p.get('ssim', 0):+.4f}"])

    lines.extend([f"| {' | '.join(r)} |" for r in rows])
    with open(out_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "Metric", "Task 1 Universal AE", "Task 2 Oracle", "Task 2 Predicted", "Routing Gap"])
        writer.writerows(rows)
    return "\n".join(lines)


def run_evaluation(
    manifest_path: str = "manifests/test_manifest.json",
    limit: Optional[int] = None,
    batch_size: int = 64,
    device: str = "cpu",
    output_dir: str = "results/task2",
    log_mlflow: bool = True,
) -> Dict[str, Any]:
    """Execute complete Task 2 evaluation and visual reporting pipeline."""
    out_dir = Path(output_dir)
    vis_dir = out_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HardRouter on device: {device}...")
    router = load_hard_router(device=device)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    entries = manifest_data["entries"] if limit is None else manifest_data["entries"][:limit]

    images_dir = get_settings().data_dir / "oxford-iiit-pet" / "images"
    clean_cache = load_clean_images(images_dir, [e["image"] for e in entries])

    summary, qual_samples, failures = evaluate_test_set(
        router, entries, clean_cache, batch_size=batch_size, device=device
    )

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table_md = format_tables_and_export(summary, Path("results/task1/metrics_summary.json"), out_dir)
    with open(out_dir / "test_summary_table.md", "w", encoding="utf-8") as f:
        f.write(table_md)
    print("\n" + table_md + "\n")

    # Qualitative Gallery (12 samples)
    lookup = {item["variant_name"]: item for item in qual_samples}
    selected_qual = [{**lookup[k], "label_str": f"{lookup[k]['image']}\n{desc}"} for k, desc in SPECS if k in lookup]
    qual_path = vis_dir / "qualitative_comparison_grid.png"
    plot_qualitative_quads(selected_qual[:12], output_path=qual_path, title="Task 2 Hard-Routing: Qualitative Test Restorations")
    print(f"Saved qualitative grid to {qual_path}")
    fail_path = vis_dir / "routing_failure_cases.png"
    if failures:
        plot_qualitative_quads(failures, output_path=fail_path, title="Task 2 Hard-Routing: Misrouting Failure Mode Analysis")
        print(f"Saved {len(failures)} failure cases to {fail_path}")

    if log_mlflow:
        print("Logging results to MLflow experiment 'task2-evaluation'...")
        tracker = ExperimentTracker()
        with tracker.run(run_name="test_evaluation", experiment_name="task2-evaluation"):
            for k, v in summary["overall_predicted"].items():
                tracker.log_metric(f"pred_overall_{k}", v)
            for k, v in summary["overall_oracle"].items():
                tracker.log_metric(f"orac_overall_{k}", v)
            for p in (out_dir / "metrics_summary.json", out_dir / "metrics_summary.csv", qual_path):
                tracker.log_artifact(str(p))
            if failures:
                tracker.log_artifact(str(fail_path))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 2 HardRouter on Test Set")
    parser.add_argument("--manifest", type=str, default="manifests/test_manifest.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        manifest_path=args.manifest, limit=args.limit, batch_size=args.batch_size,
        device=args.device, log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()

