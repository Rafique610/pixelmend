"""src/task3/losses.py
--------------------
Loss functions and balance regularizers for Task 3 Soft Mixture-of-Experts (MoE).

Multi-objective composite loss:
    L_tot = lambda_1 * L1 + lambda_2 * (1 - SSIM) + lambda_3 * L_CE + lambda_4 * L_balance

Candidate Balance Regularizers:
    1. L2 Deviation: sum((mean_w - 0.25) ** 2)
    2. Negative Entropy: sum(mean_w * log(mean_w + 1e-8))
    3. Switch Transformer: 4 * sum(f_k * mean_w_k)
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytorch_msssim
import torch
import torch.nn.functional as F


def compute_balance_loss(
    routing_weights: torch.Tensor, variant: str = "l2_deviation"
) -> torch.Tensor:
    """Compute regularizer penalizing expert load imbalance over batch routing weights (B, 4)."""
    mean_w = routing_weights.mean(dim=0)
    if variant == "l2_deviation":
        return torch.sum((mean_w - 0.25) ** 2)
    elif variant == "entropy":
        return torch.sum(mean_w * torch.log(mean_w + 1e-8))
    elif variant == "switch":
        hard_assign = torch.argmax(routing_weights.detach(), dim=-1)
        f = torch.zeros(4, device=routing_weights.device, dtype=routing_weights.dtype)
        for k in range(4):
            f[k] = (hard_assign == k).to(routing_weights.dtype).mean()
        return 4.0 * torch.sum(f * mean_w)
    raise ValueError(f"Unknown regularizer variant: {variant}")


def compute_total_loss(
    recon: torch.Tensor,
    clean: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    routing_weights: torch.Tensor,
    lambdas: Tuple[float, float, float, float] = (0.8, 0.2, 0.1, 0.01),
    balance_variant: str = "l2_deviation",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-objective composite loss: L1 + (1-SSIM) + CE + Balance."""
    l1_loss = F.l1_loss(recon, clean)
    ssim_val = pytorch_msssim.ssim(recon, clean, data_range=1.0)
    ssim_loss = 1.0 - ssim_val
    ce_loss = F.cross_entropy(logits, labels)
    bal_loss = compute_balance_loss(routing_weights, variant=balance_variant)

    l1_w, ssim_w, ce_w, bal_w = lambdas
    total = l1_w * l1_loss + ssim_w * ssim_loss + ce_w * ce_loss + bal_w * bal_loss
    loss_dict = {
        "loss_total": float(total.item()),
        "loss_l1": float(l1_loss.item()),
        "loss_ssim": float(ssim_loss.item()),
        "loss_ce": float(ce_loss.item()),
        "loss_balance": float(bal_loss.item()),
    }
    return total, loss_dict
