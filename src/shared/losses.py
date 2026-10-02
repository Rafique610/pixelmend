"""
src/shared/losses.py
--------------------
Numerically stable, differentiable PyTorch loss functions for image restoration
and generative synthesis (Tasks 1, 2, 3, and 4).

Components:
1. L1Loss: Mean Absolute Error reconstruction loss.
2. SSIMLoss: Structural dissimilarity loss (1 - SSIM) via pytorch_msssim.
3. CombinedReconstructionLoss: alpha * L1 + (1 - alpha) * SSIMLoss.
4. GANLoss: Adversarial loss supporting BCEWithLogits (vanilla) and MSE (LSGAN).
"""

from __future__ import annotations

from typing import Union

import pytorch_msssim
import torch
import torch.nn as nn


class SSIMLoss(nn.Module):
    """
    Structural Dissimilarity (1 - SSIM) loss.
    Fully differentiable with backpropagation support.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        size_average: bool = True,
        channel: int = 3,
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.size_average = size_average
        self.channel = channel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute 1.0 - SSIM(pred, target).

        Parameters:
        -----------
        pred: torch.Tensor of shape (B, C, H, W)
        target: torch.Tensor of shape (B, C, H, W)
        """
        ssim_val = pytorch_msssim.ssim(
            pred,
            target,
            data_range=self.data_range,
            size_average=self.size_average,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
        )
        return torch.clamp(1.0 - ssim_val, min=0.0)


class CombinedReconstructionLoss(nn.Module):
    """
    Weighted combination of L1 and SSIM losses:
        L_rec = alpha * L1 + (1 - alpha) * (1 - SSIM)

    Default alpha = 0.84 is the landmark optimum established in:
    Zhao et al., 'Loss Functions for Image Restoration with Neural Networks',
    IEEE Transactions on Computational Imaging, 2017.
    """

    def __init__(
        self,
        alpha: float = 0.84,
        data_range: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(
            data_range=data_range,
            win_size=win_size,
            win_sigma=win_sigma,
            size_average=True,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        """Compute the combined reconstruction loss."""
        l1 = self.l1_loss(pred, target)
        ssim_l = self.ssim_loss(pred, target)
        total = self.alpha * l1 + (1.0 - self.alpha) * ssim_l

        if return_components:
            components = {
                "loss_total": float(total.detach().item()),
                "loss_l1": float(l1.detach().item()),
                "loss_ssim": float(ssim_l.detach().item()),
            }
            return total, components

        return total


class GANLoss(nn.Module):
    """
    Adversarial loss for Conditional GAN (Task 4).
    Supports 'vanilla' (BCE with logits) and 'lsgan' (MSE).
    """

    def __init__(
        self,
        gan_mode: str = "vanilla",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.gan_mode = gan_mode.lower()

        if self.gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported gan_mode: {gan_mode}. Choose 'vanilla' or 'lsgan'.")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_value = self.target_real_label if target_is_real else self.target_fake_label
        return prediction.new_full(prediction.size(), target_value)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
