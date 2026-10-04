"""scripts/verify_challenger2_m2_baseline.py
-------------------------------------------
Empirical verification and stress testing of Task 3 Soft MoE baseline checkpoint
(checkpoints/task3/baseline_best.pth) on the full validation dataset.

Executed by: Milestone 2 Challenger 2
Assertions:
    1. Validation SSIM >= 0.68 (Task 1 baseline: 0.5935).
    2. Validation PSNR > 20.16 dB (Task 1 baseline: 20.16 dB).
    3. Non-collapse guarantee: Every expert routing weight in [0.05, 0.90].
    4. Adversarial boundary input stability (black, white, random noise).
    5. Temperature sensitivity analysis (tau in [0.2, 5.0]).
    6. Per-corruption class routing weight distribution analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytorch_msssim
import torch
from torch.utils.data import DataLoader

from src.shared.config import get_settings
from src.shared.datasets.corrupted import get_corrupted_pet_dataloader
from src.task3.moe_model import SoftMoE
from src.task3.training_utils import evaluate_validation


def run_full_validation_audit(
    model: SoftMoE, val_loader: DataLoader, device: torch.device
) -> Dict[str, Any]:
    """Run standard evaluate_validation() and record metrics."""
    print("=" * 70)
    print("1. FULL VALIDATION DATASET EVALUATION")
    print("=" * 70)
    t0 = time.time()
    metrics = evaluate_validation(
        model,
        val_loader,
        device=device,
        lambdas=(0.8, 0.2, 0.1, 0.01),
        balance_variant="l2_deviation",
        tau=1.0,
    )
    elapsed = time.time() - t0
    print(f"Validation completed in {elapsed:.2f}s on {device}")
    print(f"  - Validation SSIM:       {metrics['val_ssim']:.4f}")
    print(f"  - Validation PSNR:       {metrics['val_psnr']:.2f} dB")
    print(f"  - Validation Loss:       {metrics['val_loss']:.4f}")
    print(f"  - Validation MAE:        {metrics['val_mae']:.4f}")
    print(f"  - Validation Accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"  - Avg Routing Vector:    {metrics['avg_routing_vector']}")
    print(f"  - Min Branch Weight:     {metrics['min_utilization']:.4f}")
    print(f"  - Max Branch Weight:     {metrics['max_utilization']:.4f}")
    print(f"  - Collapse Warning:      {metrics['collapse_warning']}")

    # Required Assertions
    assert metrics["val_ssim"] >= 0.68, f"SSIM {metrics['val_ssim']} < 0.68 threshold!"
    assert metrics["val_psnr"] > 20.16, f"PSNR {metrics['val_psnr']} <= 20.16 dB threshold!"
    for i, w in enumerate(metrics["avg_routing_vector"]):
        assert w >= 0.05, f"Branch {i} weight {w} < 0.05 (starvation)!"
        assert w <= 0.90, f"Branch {i} weight {w} > 0.90 (dominance)!"
    assert not metrics["collapse_warning"], "Routing collapse flag is True!"
    print(">>> ALL BASELINE CRITERIA PASSED! <<<")
    return metrics


def run_per_corruption_breakdown(
    model: SoftMoE, val_loader: DataLoader, device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """Evaluate routing and restoration partitioned by corruption type."""
    print("\n" + "=" * 70)
    print("2. PER-CORRUPTION PARTITION ANALYSIS")
    print("=" * 70)
    class_names = ["clean_identity", "salt_and_pepper", "gaussian_blur", "occlusion"]
    per_class_psnr: Dict[int, List[float]] = {i: [] for i in range(4)}
    per_class_ssim: Dict[int, List[float]] = {i: [] for i in range(4)}
    per_class_weights: Dict[int, List[torch.Tensor]] = {i: [] for i in range(4)}

    model.eval()
    with torch.no_grad():
        for corr, clean, lbl in val_loader:
            corr, clean, lbl = corr.to(device), clean.to(device), lbl.to(device)
            recon, w, _ = model(corr, tau=1.0)
            mse = torch.mean((recon - clean) ** 2, dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            ssim = pytorch_msssim.ssim(recon, clean, data_range=1.0, size_average=False)

            for b in range(corr.size(0)):
                c = int(lbl[b].item())
                per_class_psnr[c].append(float(psnr[b].item()))
                per_class_ssim[c].append(float(ssim[b].item()))
                per_class_weights[c].append(w[b : b + 1].cpu())

    results = {}
    for c, name in enumerate(class_names):
        avg_w = torch.cat(per_class_weights[c], dim=0).mean(dim=0).tolist()
        c_psnr = float(sum(per_class_psnr[c]) / len(per_class_psnr[c]))
        c_ssim = float(sum(per_class_ssim[c]) / len(per_class_ssim[c]))
        peak_b = int(torch.tensor(avg_w).argmax().item())
        results[name] = {
            "count": len(per_class_psnr[c]),
            "psnr": round(c_psnr, 2),
            "ssim": round(c_ssim, 4),
            "avg_routing": [round(x, 4) for x in avg_w],
            "primary_branch": peak_b,
        }
        print(
            f"  [{name:16s}] N={len(per_class_psnr[c]):3d} | "
            f"PSNR={c_psnr:.2f} dB | SSIM={c_ssim:.4f} | "
            f"Peak: Branch {peak_b} | Routing={results[name]['avg_routing']}"
        )
    return results


def run_temperature_stress_test(
    model: SoftMoE, val_loader: DataLoader, device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """Stress-test gating stability across routing temperatures."""
    print("\n" + "=" * 70)
    print("3. ROUTING TEMPERATURE SENSITIVITY STRESS TEST")
    print("=" * 70)
    taus = [0.2, 0.5, 1.0, 2.0, 5.0]
    tau_results = {}

    for tau in taus:
        m = evaluate_validation(
            model,
            val_loader,
            device=device,
            lambdas=(0.8, 0.2, 0.1, 0.01),
            balance_variant="l2_deviation",
            tau=tau,
        )
        tau_results[f"tau_{tau}"] = {
            "tau": tau,
            "ssim": m["val_ssim"],
            "psnr": m["val_psnr"],
            "routing": m["avg_routing_vector"],
            "min_utilization": m["min_utilization"],
            "max_utilization": m["max_utilization"],
        }
        print(
            f"  tau={tau:3.1f} | SSIM={m['val_ssim']:.4f} | PSNR={m['val_psnr']:.2f} dB | "
            f"Min={m['min_utilization']:.4f} | Max={m['max_utilization']:.4f} | "
            f"Routing={m['avg_routing_vector']}"
        )
    return tau_results


def run_adversarial_boundary_test(model: SoftMoE, device: torch.device) -> Dict[str, Any]:
    """Test model on adversarial boundary inputs: black, white, and noise tensors."""
    print("\n" + "=" * 70)
    print("4. ADVERSARIAL BOUNDARY CONDITIONS TEST")
    print("=" * 70)
    model.eval()

    inputs = {
        "all_zeros_black": torch.zeros(4, 3, 128, 128, device=device),
        "all_ones_white": torch.ones(4, 3, 128, 128, device=device),
        "uniform_noise": torch.rand(4, 3, 128, 128, device=device),
        "gaussian_noise": torch.clamp(
            torch.randn(4, 3, 128, 128, device=device) * 0.5 + 0.5, 0.0, 1.0
        ),
    }

    boundary_results = {}
    with torch.no_grad():
        for name, inp in inputs.items():
            recon, w, logits = model(inp, tau=1.0)
            assert not torch.isnan(recon).any(), f"NaN in recon for {name}!"
            assert not torch.isinf(recon).any(), f"Inf in recon for {name}!"
            assert not torch.isnan(w).any(), f"NaN in routing weights for {name}!"
            row_sums = w.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
                f"Weights do not sum to 1.0 for {name}!"
            )
            assert (w >= 0.0).all(), f"Negative weights for {name}!"
            boundary_results[name] = {
                "recon_min": round(float(recon.min().item()), 4),
                "recon_max": round(float(recon.max().item()), 4),
                "recon_mean": round(float(recon.mean().item()), 4),
                "avg_weights": [round(x, 4) for x in w.mean(dim=0).tolist()],
            }
            print(
                f"  [{name:16s}] Bounded: [{boundary_results[name]['recon_min']}, "
                f"{boundary_results[name]['recon_max']}] | "
                f"Routing={boundary_results[name]['avg_weights']}"
            )
    return boundary_results


def run_batch_size_invariance_test(
    model: SoftMoE, val_loader: DataLoader, device: torch.device
) -> float:
    """Assert output invariance across batch sizes {1, 16}."""
    print("\n" + "=" * 70)
    print("5. BATCH SIZE INVARIANCE TEST")
    print("=" * 70)
    model.eval()

    # Grab 16 samples
    sample_batch = None
    for corr, _, _ in val_loader:
        sample_batch = corr[:16].to(device)
        break
    assert sample_batch is not None

    with torch.no_grad():
        recon_batched, w_batched, _ = model(sample_batch, tau=1.0)
        single_recons, single_ws = [], []
        for i in range(sample_batch.size(0)):
            r, w, _ = model(sample_batch[i : i + 1], tau=1.0)
            single_recons.append(r)
            single_ws.append(w)
        recon_single = torch.cat(single_recons, dim=0)
        w_single = torch.cat(single_ws, dim=0)

    max_recon_diff = float((recon_batched - recon_single).abs().max().item())
    max_w_diff = float((w_batched - w_single).abs().max().item())
    print(f"  Max reconstruction diff (batched vs single): {max_recon_diff:.2e}")
    print(f"  Max routing weights diff (batched vs single): {max_w_diff:.2e}")
    assert max_recon_diff < 1e-3, f"Recon diff too large: {max_recon_diff}"
    assert max_w_diff < 1e-4, f"Weight diff too large: {max_w_diff}"
    return max_recon_diff


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Challenger 2 Verification on: {device}")

    ckpt_path = Path("checkpoints/task3/baseline_best.pth")
    assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"

    model = SoftMoE().to(device)
    payload = model.load_checkpoint(ckpt_path, map_location=device)
    print(f"Loaded checkpoint saved at Epoch {payload.get('epoch')}")

    cfg = get_settings()
    val_loader = get_corrupted_pet_dataloader(
        split="val", batch_size=32, num_workers=0, settings=cfg
    )
    assert len(val_loader.dataset) == 736, f"Expected 736 val images, got {len(val_loader.dataset)}"

    # 1. Full Validation Dataset Audit
    val_metrics = run_full_validation_audit(model, val_loader, device)

    # 2. Per-Corruption Partition Analysis
    per_corruption = run_per_corruption_breakdown(model, val_loader, device)

    # 3. Temperature Stress Test
    tau_stress = run_temperature_stress_test(model, val_loader, device)

    # 4. Adversarial Boundary Conditions
    boundary_results = run_adversarial_boundary_test(model, device)

    # 5. Batch Size Invariance
    max_diff = run_batch_size_invariance_test(model, val_loader, device)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt_path),
        "epoch": payload.get("epoch"),
        "hardware": str(device),
        "val_metrics": val_metrics,
        "per_corruption": per_corruption,
        "tau_stress_test": tau_stress,
        "adversarial_boundary": boundary_results,
        "batch_size_invariance_max_diff": max_diff,
        "verdict": "APPROVE",
    }

    out_file = Path("results/task3/challenger2_m2_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPersisted evaluation summary to {out_file}")


if __name__ == "__main__":
    main()
