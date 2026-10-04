"""src/task3/train.py
-------------------
Two-Stage Training Protocol for Task 3 Soft Mixture-of-Experts (MoE) Image Restoration.

Phase 1 (Warm-up, 5-10 epochs):
    - Specialist autoencoders frozen (requires_grad=False, eval mode).
    - Gate trained with Adam (LR 1e-3, weight_decay 1e-4).
    - Multi-objective loss: lambda_1*L1 + lambda_2*(1-SSIM) + lambda_3*CE + lambda_4*L_balance.
    - CosineAnnealingLR scheduler.

Phase 2 (Joint Fine-Tuning, 25-40 epochs):
    - Specialists unfrozen (train mode).
    - Differential parameter groups: Gate LR 1e-4, Specialists LR 2e-5.
    - Adam optimizer with gradient clipping (max_norm 1.0) and AMP autocast.
    - CosineAnnealingLR scheduler.

Validation & Health Audit:
    - Evaluated on manifests/val_manifest.json (736 images).
    - Tracks PSNR, SSIM, MAE, classification accuracy, and average routing weights.
    - Asserts no expert starvation (w_k >= 0.05).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from src.shared.config import get_settings
from src.shared.datasets.corrupted import CorruptedPetDataset, get_corrupted_pet_dataloader
from src.shared.tracking import ExperimentTracker
from src.task3.losses import compute_balance_loss, compute_total_loss
from src.task3.moe_model import SoftMoE, build_soft_moe
from src.task3.training_utils import evaluate_validation, train_one_epoch

__all__ = [
    "compute_balance_loss",
    "compute_total_loss",
    "evaluate_validation",
    "parse_args",
    "train_one_epoch",
    "train_two_stage",
]


def _execute_phase(
    model: SoftMoE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    phase: str,
    num_epochs: int,
    global_ep: int,
    lambdas: Tuple[float, float, float, float],
    bal_var: str,
    tau: float,
    tracker: ExperimentTracker,
    save_path: Path,
    latest_path: Path,
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> int:
    """Execute training epochs for a specified phase (warmup or joint)."""
    print(f"\n=== Phase {phase.capitalize()} ({num_epochs} epochs) ===")
    for ep in range(1, num_epochs + 1):
        global_ep += 1
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, device, phase, lambdas, bal_var, tau=tau
        )
        scheduler.step()
        val = evaluate_validation(model, val_loader, device, lambdas, bal_var, tau=tau)
        dur = time.time() - t0

        history.append({"epoch": global_ep, "phase": phase, "train": tr, "val": val})
        m_log = {
            "train_loss": tr["loss"],
            **{f"val_{k}": v for k, v in val.items() if isinstance(v, (int, float))},
        }
        tracker.log_metrics(m_log, step=global_ep)

        print(
            f"[{phase.capitalize()} {ep:02d}/{num_epochs:02d}] Loss: {tr['loss']:.4f} | "
            f"PSNR: {val['val_psnr']} dB | SSIM: {val['val_ssim']:.4f} | "
            f"Routing: {val['avg_routing_vector']} | {dur:.1f}s"
        )
        v_loss = float(val.get("val_loss", 0.0))
        if val["val_ssim"] > state["best_ssim"]:
            state["best_ssim"] = val["val_ssim"]
            state["best_epoch"] = global_ep
            state["best_val"] = val
            model.save_checkpoint(
                save_path, epoch=global_ep, val_loss=v_loss, metrics=val, optimizer=optimizer
            )
        model.save_checkpoint(
            latest_path, epoch=global_ep, val_loss=v_loss, metrics=val, optimizer=optimizer
        )
    return global_ep


def train_two_stage(
    model: SoftMoE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[SoftMoE, Dict[str, Any]]:
    """Execute two-stage training protocol: Warm-up followed by Joint Fine-Tuning."""
    lambdas = (args.lambda_l1, args.lambda_ssim, args.lambda_ce, args.lambda_bal)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {"best_ssim": -1.0, "best_epoch": 0, "best_val": {}}
    default_save = (
        "checkpoints/task3/smoke_best.pth"
        if getattr(args, "smoke_test", False)
        else "checkpoints/task3/baseline_best.pth"
    )
    default_metrics = (
        "results/task3/smoke_train_metrics.json"
        if getattr(args, "smoke_test", False)
        else "results/task3/baseline_train_metrics.json"
    )
    save_str = getattr(args, "save_path", None) or default_save
    metrics_str = getattr(args, "metrics_path", None) or default_metrics
    args.save_path = save_str
    args.metrics_path = metrics_str

    save_path = Path(save_str)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    latest_name = "smoke_latest.pth" if getattr(args, "smoke_test", False) else "latest.pth"
    latest_path = save_path.parent / latest_name
    metrics_path = Path(metrics_str)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker()
    run_name = "task3-moe-smoke" if getattr(args, "smoke_test", False) else "task3-moe-baseline"

    with tracker.run(run_name=run_name, experiment_name="genai-task3-soft-moe"):
        tracker.log_params(vars(args))

        # --- Phase 1: Warm-up ---
        gate_p = [p for p in model.gate.parameters() if p.requires_grad]
        opt_p1 = torch.optim.Adam(gate_p, lr=args.warmup_lr, weight_decay=args.weight_decay)
        sched_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_p1, T_max=max(1, args.warmup_epochs)
        )
        global_ep = _execute_phase(
            model, train_loader, val_loader, opt_p1, sched_p1, scaler, device, "warmup",
            args.warmup_epochs, 0, lambdas, args.balance_variant, args.tau, tracker,
            save_path, latest_path, state, history
        )

        # --- Phase 2: Joint Fine-Tuning ---
        model.set_experts_frozen(False)
        g_params = [p for p in model.gate.parameters() if p.requires_grad]
        s_params = [
            p for s in (model.specialist_salt, model.specialist_blur, model.specialist_occlusion)
            for p in s.parameters() if p.requires_grad
        ]
        opt_p2 = torch.optim.Adam([
            {"params": g_params, "lr": args.gate_lr, "weight_decay": args.weight_decay},
            {"params": s_params, "lr": args.specialist_lr, "weight_decay": args.weight_decay},
        ])
        sched_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_p2, T_max=max(1, args.joint_epochs)
        )
        _execute_phase(
            model, train_loader, val_loader, opt_p2, sched_p2, scaler, device, "joint",
            args.joint_epochs, global_ep, lambdas, args.balance_variant, args.tau, tracker,
            save_path, latest_path, state, history
        )

        summary = {
            "best_epoch": state["best_epoch"],
            "best_val_ssim": state["best_ssim"],
            "best_metrics": state["best_val"],
            "history": history,
            "config": vars(args),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if save_path.is_file():
            tracker.log_artifact(str(save_path))
        if metrics_path.is_file():
            tracker.log_artifact(str(metrics_path))

    print(f"\nTraining complete. Best SSIM: {state['best_ssim']:.4f} (Ep {state['best_epoch']}).")
    print(f"Checkpoints: {save_path} & {latest_path} | Metrics JSON: {metrics_path}")
    return model, summary


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Construct training and validation DataLoaders."""
    cfg = get_settings()
    if args.smoke_test:
        train_ds = CorruptedPetDataset(split="train", settings=cfg)
        val_ds = CorruptedPetDataset(split="val", settings=cfg)
        train_loader = DataLoader(Subset(train_ds, list(range(64))), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(val_ds, list(range(32))), batch_size=16, shuffle=False)
        return train_loader, val_loader

    train_loader = get_corrupted_pet_dataloader(
        split="train", batch_size=args.batch_size, num_workers=args.num_workers, settings=cfg
    )
    val_loader = get_corrupted_pet_dataloader(
        split="val", batch_size=args.batch_size, num_workers=args.num_workers, settings=cfg
    )
    return train_loader, val_loader


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI training arguments with decoupled smoke-test and baseline paths."""
    p = argparse.ArgumentParser(description="Task 3 Soft MoE Two-Stage Training")
    p.add_argument("--warmup-epochs", type=int, default=5, help="Warm-up epochs (5-10)")
    p.add_argument("--joint-epochs", type=int, default=25, help="Joint epochs (25-40)")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    p.add_argument("--warmup-lr", type=float, default=1e-3, help="Phase 1 Gate LR")
    p.add_argument("--gate-lr", type=float, default=1e-4, help="Phase 2 Gate LR")
    p.add_argument("--specialist-lr", type=float, default=2e-5, help="Phase 2 Specialists LR")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    p.add_argument("--tau", type=float, default=1.0, help="Routing temperature")
    p.add_argument("--lambda-l1", type=float, default=0.8, help="L1 loss coefficient")
    p.add_argument("--lambda-ssim", type=float, default=0.2, help="SSIM loss coefficient")
    p.add_argument("--lambda-ce", type=float, default=0.1, help="Auxiliary CE loss coefficient")
    p.add_argument("--lambda-bal", type=float, default=0.01, help="Balance regularizer coefficient")
    p.add_argument(
        "--balance-variant", type=str, default="l2_deviation",
        choices=["l2_deviation", "entropy", "switch"], help="Balance variant"
    )
    p.add_argument(
        "--save-path", type=str, default=None,
        help="Path for best checkpoint (defaults to baseline_best.pth or smoke_best.pth)",
    )
    p.add_argument(
        "--metrics-path", type=str, default=None,
        help="Path for metrics JSON (defaults to baseline or smoke JSON)",
    )
    p.add_argument("--device", type=str, default=None, help="Target device")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--smoke-test", action="store_true", help="Fast 1+1 epoch test on subsets")

    parsed = p.parse_args(args)
    if parsed.save_path is None:
        parsed.save_path = (
            "checkpoints/task3/smoke_best.pth"
            if parsed.smoke_test
            else "checkpoints/task3/baseline_best.pth"
        )
    if parsed.metrics_path is None:
        parsed.metrics_path = (
            "results/task3/smoke_train_metrics.json"
            if parsed.smoke_test
            else "results/task3/baseline_train_metrics.json"
        )
    return parsed


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else get_settings().torch_device
    print(f"Task 3 Soft MoE Training running on: {device}")

    model = build_soft_moe(
        classifier_path="checkpoints/task2/classifier_best.pt",
        salt_path="checkpoints/task2/specialist_salt_best.pt",
        blur_path="checkpoints/task2/specialist_blur_best.pt",
        occlusion_path="checkpoints/task2/specialist_occlusion_best.pt",
    ).to(device)

    train_loader, val_loader = build_dataloaders(args)
    train_two_stage(model, train_loader, val_loader, args, device)


if __name__ == "__main__":
    main()
