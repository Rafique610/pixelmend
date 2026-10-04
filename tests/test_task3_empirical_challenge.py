"""tests/test_task3_empirical_challenge.py
---------------------------------------
Adversarial stress tests and empirical verification harness for Milestone 2:
1. Differential parameter groups: Gate LR 1e-4 vs Specialist LR 2e-5.
2. Phase 1 Warm-up: Specialist gradients are None and weights remain strictly immutable.
3. CUDA AMP autocast and GradScaler scaling/unscaling/clipping/step behavior.
4. Balance regularizer edge cases (uniform routing, collapse, differentiability).
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.task2.specialist import SpecialistAutoencoder
from src.task3.gate import GatingNetwork
from src.task3.losses import compute_balance_loss, compute_total_loss
from src.task3.moe_model import SoftMoE
from src.task3.train import train_two_stage
from src.task3.training_utils import train_one_epoch


def _create_toy_moe() -> SoftMoE:
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


def test_differential_parameter_groups_structure_and_lrs() -> None:
    """Empirically verify optimizer parameter groups, learning rates, and gradient updates."""
    model = _create_toy_moe()
    model.set_experts_frozen(False)

    g_params = [p for p in model.gate.parameters() if p.requires_grad]
    s_params = [
        p for s in (model.specialist_salt, model.specialist_blur, model.specialist_occlusion)
        for p in s.parameters() if p.requires_grad
    ]

    opt = torch.optim.Adam([
        {"params": g_params, "lr": 1e-4, "weight_decay": 1e-4},
        {"params": s_params, "lr": 2e-5, "weight_decay": 1e-4},
    ])

    assert len(opt.param_groups) == 2, "Expected exactly 2 parameter groups"
    assert opt.param_groups[0]["lr"] == 1e-4, "Gate LR must be 1e-4"
    assert opt.param_groups[1]["lr"] == 2e-5, "Specialist LR must be 2e-5"

    gate_param_ids = {id(p) for p in g_params}
    spec_param_ids = {id(p) for p in s_params}
    group0_ids = {id(p) for p in opt.param_groups[0]["params"]}
    group1_ids = {id(p) for p in opt.param_groups[1]["params"]}

    assert group0_ids == gate_param_ids, "Group 0 must strictly contain gate parameters"
    assert group1_ids == spec_param_ids, "Group 1 must strictly contain specialist parameters"
    assert len(group0_ids.intersection(group1_ids)) == 0, "Parameter groups must be disjoint"

    # Verify updates take place in both groups
    g_initial = [p.clone().detach() for p in g_params]
    s_initial = [p.clone().detach() for p in s_params]

    x = torch.rand(2, 3, 128, 128)
    clean = torch.rand(2, 3, 128, 128)
    lbl = torch.tensor([0, 1], dtype=torch.long)

    recon, w, logits = model(x)
    loss, _ = compute_total_loss(recon, clean, logits, lbl, w)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Check both groups updated
    g_changed = any(not torch.equal(p, p_init) for p, p_init in zip(g_params, g_initial))
    s_changed = any(not torch.equal(p, p_init) for p, p_init in zip(s_params, s_initial))
    assert g_changed, "Gate parameters must be updated by optimizer step"
    assert s_changed, "Specialist parameters must be updated by optimizer step"


def test_phase1_warmup_specialist_gradient_and_weight_immutability() -> None:
    """Verify that during Phase 1, specialist gradients are None and weights remain constant."""
    model = _create_toy_moe()
    model.set_experts_frozen(True)
    model.gate.train()
    model.specialist_salt.eval()
    model.specialist_blur.eval()
    model.specialist_occlusion.eval()

    gate_p = [p for p in model.gate.parameters() if p.requires_grad]
    opt_p1 = torch.optim.Adam(gate_p, lr=1e-3, weight_decay=1e-4)

    specialists = (model.specialist_salt, model.specialist_blur, model.specialist_occlusion)
    initial_spec_weights = [
        [p.clone().detach() for p in specialist.parameters()]
        for specialist in specialists
    ]
    initial_bn_means = [
        [m.running_mean.clone().detach() for m in specialist.modules() if isinstance(m, nn.BatchNorm2d)]
        for specialist in specialists
    ]

    x = torch.rand(4, 3, 128, 128)
    clean = torch.rand(4, 3, 128, 128)
    lbl = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    opt_p1.zero_grad()
    recon, w, logits = model(x)
    loss, _ = compute_total_loss(recon, clean, logits, lbl, w)
    loss.backward()

    # Gate parameters must have valid non-zero gradients
    assert all(p.grad is not None for p in gate_p), "All gate parameters must have gradients"
    assert any((p.grad != 0).any() for p in gate_p), "Gate gradients must be non-zero"

    # Specialist parameters must have requires_grad=False and grad=None
    for specialist in specialists:
        for p in specialist.parameters():
            assert not p.requires_grad, "Specialist param requires_grad must be False"
            assert p.grad is None, "Specialist param grad must be None during Phase 1"

    opt_p1.step()

    # Assert 100% strict numerical equality of specialist weights after optimizer step
    for specialist, init_weights in zip(specialists, initial_spec_weights):
        for p, init_p in zip(specialist.parameters(), init_weights):
            assert torch.equal(p, init_p), "Specialist weights drifted during Phase 1 step!"
            assert (p - init_p).abs().max().item() == 0.0

    # Assert BatchNorm running statistics did not drift in eval mode
    for specialist, init_means in zip(specialists, initial_bn_means):
        current_means = [
            m.running_mean for m in specialist.modules() if isinstance(m, nn.BatchNorm2d)
        ]
        for curr, init_m in zip(current_means, init_means):
            assert torch.equal(curr, init_m), "Specialist BatchNorm running mean drifted!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP verification")
def test_cuda_amp_autocast_and_grad_scaler_step() -> None:
    """Empirically test AMP autocast, gradient scaling, unscaling, clipping, and optimizer step on CUDA."""
    device = torch.device("cuda")
    model = _create_toy_moe().to(device)
    model.set_experts_frozen(False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    x = torch.rand(2, 3, 128, 128, device=device)
    clean = torch.rand(2, 3, 128, 128, device=device)
    lbl = torch.tensor([0, 1], dtype=torch.long, device=device)

    opt.zero_grad()
    with torch.amp.autocast("cuda"):
        recon, w, logits = model(x)
        assert recon.device.type == "cuda"
        loss, ld = compute_total_loss(recon, clean, logits, lbl, w)
        assert torch.isfinite(loss), "Loss must be finite under AMP"

    init_scale = scaler.get_scale()
    scaler.scale(loss).backward()

    # Check gradients exist and are scaled
    scaled_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(scaled_grads) > 0

    # Unscale before clipping
    scaler.unscale_(opt)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert not torch.isnan(total_norm)

    scaler.step(opt)
    scaler.update()

    # Scaler must complete without error and scale factor remains positive
    assert scaler.get_scale() > 0.0


def test_balance_regularizers_mathematical_bounds() -> None:
    """Empirically stress-test balance regularizers on uniform, partial, and collapsed distributions."""
    # 1. Perfectly uniform
    w_uni = torch.full((10, 4), 0.25, requires_grad=True)
    l2_uni = compute_balance_loss(w_uni, variant="l2_deviation")
    assert pytest.approx(l2_uni.item(), abs=1e-7) == 0.0

    switch_uni = compute_balance_loss(w_uni, variant="switch")
    assert pytest.approx(switch_uni.item(), rel=1e-3) == 1.0

    # 2. Total collapse to expert 0
    w_col = torch.zeros(10, 4, requires_grad=True)
    with torch.no_grad():
        w_col[:, 0] = 1.0
    l2_col = compute_balance_loss(w_col, variant="l2_deviation")
    # Expected L2: (1.0 - 0.25)^2 + 3 * (0.0 - 0.25)^2 = 0.5625 + 0.1875 = 0.75
    assert pytest.approx(l2_col.item(), rel=1e-5) == 0.75

    switch_col = compute_balance_loss(w_col, variant="switch")
    # Expected switch: 4.0 * (1.0 * 1.0) = 4.0
    assert pytest.approx(switch_col.item(), rel=1e-5) == 4.0
    assert switch_col.item() > switch_uni.item(), "Switch loss must penalize collapse"
    assert l2_col.item() > l2_uni.item(), "L2 loss must penalize collapse"


def test_smoke_test_execution_clean(tmp_path: Path) -> None:
    """Verify train_two_stage runs end-to-end with smoke_test=True on small dataloaders without errors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_toy_moe().to(device)

    # Synthetic dataloaders
    corr = torch.rand(16, 3, 128, 128)
    clean = torch.rand(16, 3, 128, 128)
    lbl = torch.randint(0, 4, (16,))
    ds = TensorDataset(corr, clean, lbl)
    train_loader = DataLoader(ds, batch_size=4)
    val_loader = DataLoader(ds, batch_size=4)

    import argparse
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
        save_path=str(tmp_path / "smoke_best.pth"),
        metrics_path=str(tmp_path / "smoke_metrics.json"),
        smoke_test=True,
    )

    trained_model, summary = train_two_stage(model, train_loader, val_loader, args, device)
    assert Path(args.save_path).is_file()
    assert Path(args.metrics_path).is_file()
    assert len(summary["history"]) == 2
