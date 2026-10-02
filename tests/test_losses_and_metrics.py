"""
tests/test_losses_and_metrics.py
--------------------------------
Unit tests for shared losses, evaluation metrics, and visualization helpers.
"""

import matplotlib.pyplot as plt
import torch

from src.shared.losses import CombinedReconstructionLoss, GANLoss, SSIMLoss
from src.shared.metrics import (
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_ssim,
    evaluate_metrics,
)
from src.shared.visualization import (
    make_error_heatmap,
    make_reconstruction_grid,
    plot_training_curves,
)


def test_ssim_and_combined_loss_identical_tensors():
    x = torch.rand((2, 3, 128, 128))
    y = x.clone()

    ssim_loss_fn = SSIMLoss(data_range=1.0)
    comb_loss_fn = CombinedReconstructionLoss(alpha=0.84, data_range=1.0)

    l_ssim = ssim_loss_fn(x, y)
    l_comb, comps = comb_loss_fn(x, y, return_components=True)

    assert torch.isclose(l_ssim, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(l_comb, torch.tensor(0.0), atol=1e-5)
    assert comps["loss_l1"] == 0.0
    assert comps["loss_ssim"] == 0.0


def test_losses_gradient_backprop():
    pred = torch.rand((2, 3, 64, 64), requires_grad=True)
    target = torch.rand((2, 3, 64, 64))

    loss_fn = CombinedReconstructionLoss(alpha=0.84, data_range=1.0)
    loss = loss_fn(pred, target)

    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_gan_loss():
    gan_loss = GANLoss(gan_mode="vanilla")
    pred_real = torch.tensor([[2.5, 3.0]])
    pred_fake = torch.tensor([[-2.5, -3.0]])

    l_real = gan_loss(pred_real, target_is_real=True)
    l_fake = gan_loss(pred_fake, target_is_real=False)

    assert l_real.item() > 0.0
    assert l_fake.item() > 0.0
    # Confident correct predictions should yield low loss (< 0.15)
    assert l_real.item() < 0.15
    assert l_fake.item() < 0.15


def test_metrics_perfect_reconstruction():
    x = torch.rand((4, 3, 128, 128))
    y = x.clone()

    metrics = evaluate_metrics(x, y)
    assert metrics["psnr"] >= 80.0
    assert metrics["ssim"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0


def test_metrics_imperfect_reconstruction():
    black = torch.zeros((1, 3, 64, 64))
    white = torch.ones((1, 3, 64, 64))

    psnr = compute_psnr(black, white)
    ssim = compute_ssim(black, white)
    mae = compute_mae(black, white)
    mse = compute_mse(black, white)

    assert psnr <= 0.01
    assert ssim < 0.01
    assert mae == 1.0
    assert mse == 1.0


def test_visualization_reconstruction_grid():
    c = torch.rand((6, 3, 128, 128))
    r = torch.rand((6, 3, 128, 128))
    t = torch.rand((6, 3, 128, 128))

    grid = make_reconstruction_grid(c, r, t, n_samples=4)
    # 3 rows (c, r, t) x 4 columns with padding
    assert grid.ndim == 3
    assert grid.shape[0] == 3
    assert 0.0 <= grid.min() and grid.max() <= 1.0


def test_visualization_error_heatmap():
    target = torch.ones((3, 128, 128))
    restored = torch.zeros((3, 128, 128))

    heatmap = make_error_heatmap(restored, target, cmap="inferno")
    assert heatmap.shape == (3, 128, 128)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0


def test_plot_training_curves():
    history = {
        "train_loss": [0.5, 0.4, 0.3],
        "val_loss": [0.55, 0.42, 0.33],
        "val_psnr": [22.0, 24.5, 26.8],
        "val_ssim": [0.75, 0.81, 0.88],
    }
    fig = plot_training_curves(history, title="Test Run")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
