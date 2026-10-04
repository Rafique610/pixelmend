"""tests/test_task3_train.py
-------------------------
Unit tests for Task 3 Two-Stage Training Protocol, Loss Formulations, and Validation Engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.task2.specialist import SpecialistAutoencoder
from src.task3.gate import GatingNetwork
from src.task3.losses import compute_balance_loss, compute_total_loss
from src.task3.moe_model import SoftMoE
from src.task3.train import build_dataloaders, parse_args, train_two_stage
from src.task3.training_utils import evaluate_validation, train_one_epoch


@pytest.fixture
def mock_moe() -> SoftMoE:
    """Lightweight SoftMoE for fast unit testing."""
    torch.manual_seed(42)
    gate = GatingNetwork(channels=(16, 32, 64, 128), dropout=0.0)
    s_salt = SpecialistAutoencoder(
        corruption_type="salt_and_pepper", channels=(16, 32, 64, 128), bottleneck_dim=128
    )
    s_blur = SpecialistAutoencoder(
        corruption_type="gaussian_blur", channels=(16, 32, 64, 128), bottleneck_dim=128
    )
    s_occ = SpecialistAutoencoder(
        corruption_type="occlusion", channels=(16, 32, 64, 128), bottleneck_dim=128
    )
    return SoftMoE(
        gate=gate, specialist_salt=s_salt, specialist_blur=s_blur, specialist_occlusion=s_occ
    )


@pytest.fixture
def mock_loader() -> DataLoader:
    """Mock dataloader with 8 synthetic (corrupted, clean, label) samples."""
    torch.manual_seed(42)
    corr = torch.rand(8, 3, 128, 128)
    clean = torch.rand(8, 3, 128, 128)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    ds = TensorDataset(corr, clean, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


def test_balance_losses_uniform_routing() -> None:
    """L2 balance loss should be exactly 0 for perfectly uniform routing."""
    w_uniform = torch.full((16, 4), 0.25, requires_grad=True)
    l2_bal = compute_balance_loss(w_uniform, variant="l2_deviation")
    assert pytest.approx(l2_bal.item(), abs=1e-7) == 0.0

    # Entropy of uniform distribution is -log(4) ~ -1.3863
    ent_bal = compute_balance_loss(w_uniform, variant="entropy")
    assert ent_bal.item() < 0.0

    # Switch transformer balance loss for uniform assignments is 4 * sum(0.25 * 0.25) = 1.0
    switch_bal = compute_balance_loss(w_uniform, variant="switch")
    assert pytest.approx(switch_bal.item(), rel=1e-3) == 1.0


def test_balance_losses_invalid_variant() -> None:
    """Assert ValueError when invalid regularizer variant is provided."""
    w = torch.full((4, 4), 0.25)
    with pytest.raises(ValueError, match="Unknown regularizer variant"):
        compute_balance_loss(w, variant="nonexistent_variant")


def test_compute_total_loss_gradients() -> None:
    """Verify compute_total_loss preserves backward gradients to all inputs."""
    recon = torch.rand(4, 3, 128, 128, requires_grad=True)
    clean = torch.rand(4, 3, 128, 128)
    logits = torch.randn(4, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    weights = torch.softmax(logits, dim=-1)

    total_loss, loss_dict = compute_total_loss(
        recon, clean, logits, labels, weights, lambdas=(0.8, 0.2, 0.1, 0.01)
    )

    assert total_loss.ndim == 0
    assert "loss_total" in loss_dict
    assert "loss_l1" in loss_dict
    assert "loss_ssim" in loss_dict
    assert "loss_ce" in loss_dict
    assert "loss_balance" in loss_dict

    total_loss.backward()
    assert recon.grad is not None and not torch.isnan(recon.grad).any()
    assert logits.grad is not None and not torch.isnan(logits.grad).any()


def test_evaluate_validation_metrics(mock_moe: SoftMoE, mock_loader: DataLoader) -> None:
    """Verify evaluate_validation returns all required metrics and routing distributions."""
    device = torch.device("cpu")
    mock_moe.to(device)
    metrics = evaluate_validation(mock_moe, mock_loader, device=device)

    required_keys = [
        "val_loss", "val_psnr", "val_ssim", "val_mae", "val_accuracy",
        "avg_routing_vector", "min_utilization", "max_utilization", "collapse_warning",
    ]
    for k in required_keys:
        assert k in metrics, f"Missing key: {k}"

    assert len(metrics["avg_routing_vector"]) == 4
    assert pytest.approx(sum(metrics["avg_routing_vector"]), abs=1e-3) == 1.0
    assert isinstance(metrics["collapse_warning"], bool)


def test_train_one_epoch_warmup_freezes_experts(
    mock_moe: SoftMoE, mock_loader: DataLoader
) -> None:
    """Verify train_one_epoch in 'warmup' phase freezes specialists."""
    device = torch.device("cpu")
    mock_moe.to(device)
    gate_params = [p for p in mock_moe.gate.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(gate_params, lr=1e-3)

    metrics = train_one_epoch(
        mock_moe, mock_loader, optimizer, scaler=None, device=device, phase="warmup",
        lambdas=(0.8, 0.2, 0.1, 0.01), balance_variant="l2_deviation"
    )

    assert "loss" in metrics
    assert mock_moe.experts_frozen
    specialists = (
        mock_moe.specialist_salt, mock_moe.specialist_blur, mock_moe.specialist_occlusion
    )
    for specialist in specialists:
        for p in specialist.parameters():
            assert not p.requires_grad


def test_train_one_epoch_joint_unfreezes_experts(
    mock_moe: SoftMoE, mock_loader: DataLoader
) -> None:
    """Verify train_one_epoch in 'joint' phase unfreezes specialists."""
    device = torch.device("cpu")
    mock_moe.to(device)
    optimizer = torch.optim.Adam(mock_moe.parameters(), lr=1e-4)

    metrics = train_one_epoch(
        mock_moe, mock_loader, optimizer, scaler=None, device=device, phase="joint",
        lambdas=(0.8, 0.2, 0.1, 0.01), balance_variant="l2_deviation"
    )

    assert "loss" in metrics
    assert not mock_moe.experts_frozen
    specialists = (
        mock_moe.specialist_salt, mock_moe.specialist_blur, mock_moe.specialist_occlusion
    )
    for specialist in specialists:
        for p in specialist.parameters():
            assert p.requires_grad


def test_two_stage_training_end_to_end(
    mock_moe: SoftMoE, mock_loader: DataLoader, tmp_path: Path
) -> None:
    """Verify complete two-stage training loop saves valid checkpoints and history JSON."""
    device = torch.device("cpu")
    mock_moe.to(device)
    save_path = tmp_path / "best.pth"
    metrics_path = tmp_path / "metrics.json"

    args = argparse.Namespace(
        warmup_epochs=1,
        joint_epochs=1,
        batch_size=4,
        warmup_lr=1e-3,
        gate_lr=1e-4,
        specialist_lr=2e-5,
        weight_decay=1e-4,
        tau=1.0,
        lambda_l1=0.8,
        lambda_ssim=0.2,
        lambda_ce=0.1,
        lambda_bal=0.01,
        balance_variant="l2_deviation",
        save_path=str(save_path),
        metrics_path=str(metrics_path),
        smoke_test=True,
    )

    model, summary = train_two_stage(mock_moe, mock_loader, mock_loader, args, device)
    assert save_path.is_file()
    assert metrics_path.is_file()
    assert summary["best_epoch"] in [1, 2]
    assert len(summary["history"]) == 2

    # Load saved checkpoint
    loaded_ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in loaded_ckpt
    assert "optimizer_state_dict" in loaded_ckpt
    assert loaded_ckpt["optimizer_state_dict"] is not None
    assert "metrics" in loaded_ckpt
    assert "val_ssim" in loaded_ckpt["metrics"]


def test_build_dataloaders_smoke_test() -> None:
    """Verify build_dataloaders returns valid small loaders in smoke_test mode."""
    args = argparse.Namespace(smoke_test=True, batch_size=16, num_workers=0)
    tr_loader, val_loader = build_dataloaders(args)
    assert len(tr_loader.dataset) == 64
    assert len(val_loader.dataset) == 32


def test_checkpoint_includes_optimizer_state(mock_moe: SoftMoE, tmp_path: Path) -> None:
    """Verify SoftMoE.save_checkpoint serializes optimizer_state_dict."""
    opt = torch.optim.Adam(mock_moe.parameters(), lr=1e-3)
    save_file = tmp_path / "ckpt_opt.pth"
    mock_moe.save_checkpoint(save_file, epoch=3, val_loss=0.045, optimizer=opt)
    assert save_file.is_file()

    payload = torch.load(save_file, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in payload
    assert payload["optimizer_state_dict"] is not None
    assert "param_groups" in payload["optimizer_state_dict"]
    assert "state" in payload["optimizer_state_dict"]
    assert payload["epoch"] == 3
    assert payload["val_loss"] == 0.045


def test_smoke_test_default_paths_isolation() -> None:
    """Verify smoke test defaults decouple from baseline paths and do not touch baseline."""
    args_smoke = parse_args(["--smoke-test"])
    assert "smoke_best.pth" in args_smoke.save_path
    assert "smoke_train_metrics.json" in args_smoke.metrics_path
    assert "baseline_best.pth" not in args_smoke.save_path

    args_prod = parse_args([])
    assert "baseline_best.pth" in args_prod.save_path
    assert "baseline_train_metrics.json" in args_prod.metrics_path
    assert "smoke_best.pth" not in args_prod.save_path

    # Custom override verification
    args_custom = parse_args(["--smoke-test", "--save-path", "custom_best.pth"])
    assert args_custom.save_path == "custom_best.pth"
