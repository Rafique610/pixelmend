"""src/task3/training_utils.py
----------------------------
Epoch training and validation routines for Task 3 Soft Mixture-of-Experts (MoE).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytorch_msssim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.task3.losses import compute_total_loss
from src.task3.moe_model import SoftMoE


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    lambdas: Tuple[float, float, float, float] = (0.8, 0.2, 0.1, 0.01),
    balance_variant: str = "l2_deviation",
    tau: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate SoftMoE on validation dataset and audit routing distributions."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_psnr, all_ssim, all_mae = [], [], []
    all_weights: List[torch.Tensor] = []

    with torch.no_grad():
        for corr, clean, lbl in val_loader:
            corr, clean, lbl = corr.to(device), clean.to(device), lbl.to(device)
            recon, w, logits = model(corr, tau=tau)
            loss, _ = compute_total_loss(recon, clean, logits, lbl, w, lambdas, balance_variant)
            total_loss += loss.item() * corr.size(0)

            mse = torch.mean((recon - clean) ** 2, dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            ssim = pytorch_msssim.ssim(recon, clean, data_range=1.0, size_average=False)
            mae = torch.mean(torch.abs(recon - clean), dim=[1, 2, 3])

            all_psnr.extend(psnr.cpu().tolist())
            all_ssim.extend(ssim.cpu().tolist())
            all_mae.extend(mae.cpu().tolist())
            total_correct += int((torch.argmax(logits, dim=-1) == lbl).sum().item())
            total_samples += corr.size(0)
            all_weights.append(w.cpu())

    avg_w = torch.cat(all_weights, dim=0).mean(dim=0).tolist()
    min_w, max_w = float(min(avg_w)), float(max(avg_w))
    collapse = bool(max_w > 0.90 or min_w < 0.05)

    return {
        "val_loss": round(total_loss / max(1, total_samples), 4),
        "val_psnr": round(float(sum(all_psnr) / max(1, len(all_psnr))), 2),
        "val_ssim": round(float(sum(all_ssim) / max(1, len(all_ssim))), 4),
        "val_mae": round(float(sum(all_mae) / max(1, len(all_mae))), 4),
        "val_accuracy": round(total_correct / max(1, total_samples), 4),
        "avg_routing_vector": [round(x, 4) for x in avg_w],
        "min_utilization": round(min_w, 4),
        "max_utilization": round(max_w, 4),
        "collapse_warning": collapse,
    }


def train_one_epoch(
    model: SoftMoE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    phase: str,
    lambdas: Tuple[float, float, float, float],
    balance_variant: str,
    tau: float = 1.0,
) -> Dict[str, float]:
    """Train model for one epoch in either 'warmup' or 'joint' phase."""
    if phase == "warmup":
        model.set_experts_frozen(True)
        model.gate.train()
        model.specialist_salt.eval()
        model.specialist_blur.eval()
        model.specialist_occlusion.eval()
    else:
        model.set_experts_frozen(False)
        model.train()

    t_loss, t_l1, t_ssim, t_ce, t_bal = 0.0, 0.0, 0.0, 0.0, 0.0
    use_amp = scaler is not None and device.type == "cuda"

    for corr, clean, lbl in dataloader:
        corr, clean, lbl = corr.to(device), clean.to(device), lbl.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                recon, w, logits = model(corr, tau=tau)
                loss, ld = compute_total_loss(
                    recon, clean, logits, lbl, w, lambdas, balance_variant
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            recon, w, logits = model(corr, tau=tau)
            loss, ld = compute_total_loss(recon, clean, logits, lbl, w, lambdas, balance_variant)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        t_loss += ld["loss_total"]
        t_l1 += ld["loss_l1"]
        t_ssim += ld["loss_ssim"]
        t_ce += ld["loss_ce"]
        t_bal += ld["loss_balance"]

    denom = max(1, len(dataloader))
    return {
        "loss": round(t_loss / denom, 4),
        "loss_l1": round(t_l1 / denom, 4),
        "loss_ssim": round(t_ssim / denom, 4),
        "loss_ce": round(t_ce / denom, 4),
        "loss_balance": round(t_bal / denom, 4),
    }
