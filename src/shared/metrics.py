"""
src/shared/metrics.py
---------------------
Standardized computer vision evaluation metrics for Tasks 1, 2, 3, and 4:
- PSNR: Peak Signal-to-Noise Ratio (dB)
- SSIM: Structural Similarity Index Measure
- MAE / L1: Mean Absolute Error
- MSE: Mean Squared Error
"""

from __future__ import annotations

import math

import pytorch_msssim
import torch


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Parameters:
    -----------
    pred: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    target: torch.Tensor of same shape as pred
    max_val: float maximum possible pixel value (1.0 for [0, 1] normalized images)
    eps: small epsilon to avoid division by zero
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 100.0  # Perfect reconstruction ceiling

    psnr = 10.0 * math.log10((max_val**2) / mse)
    return float(psnr)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
) -> float:
    """
    Compute mean Structural Similarity Index (SSIM).

    Parameters:
    -----------
    pred: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    target: torch.Tensor of same shape as pred
    data_range: dynamic range of pixel values (1.0 for [0, 1] images)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    p = pred if pred.ndim == 4 else pred.unsqueeze(0)
    t = target if target.ndim == 4 else target.unsqueeze(0)

    val = pytorch_msssim.ssim(
        p,
        t,
        data_range=data_range,
        size_average=True,
        win_size=win_size,
    )
    return float(val.item())


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error (L1 pixel error)."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Squared Error (L2 pixel error)."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    return float(torch.mean((pred - target) ** 2).item())


def evaluate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> dict[str, float]:
    """
    Convenience wrapper computing PSNR, SSIM, MAE, and MSE in a single call.

    Returns:
    --------
    dict with keys 'psnr', 'ssim', 'mae', 'mse'
    """
    return {
        "psnr": compute_psnr(pred, target, max_val=data_range),
        "ssim": compute_ssim(pred, target, data_range=data_range),
        "mae": compute_mae(pred, target),
        "mse": compute_mse(pred, target),
    }
