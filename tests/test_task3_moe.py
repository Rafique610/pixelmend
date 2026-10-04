"""tests/test_task3_moe.py
------------------------
Unit tests for Task 3 Soft Mixture-of-Experts (MoE) Architecture and Differentiable Blending.

Covers:
- GatingNetwork architecture, parameter count, feature extraction
- Temperature scaling effect on routing entropy and sharpness
- Temperature clamping safeguard (tau >= 0.05) preventing NaN/Inf overflow
- Probability simplex constraint: sum(w_k) = 1.0 +/- 1e-6 and w_k >= 0
- SoftMoE container initialization and forward pass shapes (batch sizes 1 and 4)
- Non-zero gradient flow to gate and specialist parameters through blending
- Expert parameter freezing and unfreezing utilities
- Exact identity branch pass-through preservation when w_1 = 1.0
- Mathematical verification of convex combination blending
- Task 2 pretrained component checkpoint loading
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task2.specialist import SpecialistAutoencoder
from src.task3.gate import GatingNetwork, build_gating_network
from src.task3.moe_model import SoftMoE, build_soft_moe

CHECKPOINT_DIR = Path("checkpoints/task2")
CLASSIFIER_CKPT = CHECKPOINT_DIR / "classifier_best.pt"
SALT_CKPT = CHECKPOINT_DIR / "specialist_salt_best.pt"
BLUR_CKPT = CHECKPOINT_DIR / "specialist_blur_best.pt"
OCCLUSION_CKPT = CHECKPOINT_DIR / "specialist_occlusion_best.pt"


class MockConstantSpecialist(nn.Module):
    """Deterministic specialist returning a constant value for mathematical verification."""

    def __init__(self, fill_value: float) -> None:
        super().__init__()
        self.fill_value = nn.Parameter(torch.tensor(fill_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * self.fill_value


class MockTargetGate(nn.Module):
    """Deterministic gate returning fixed routing weights and logits."""

    def __init__(self, target_weights: list[float]) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(target_weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0] if x.ndim == 4 else 1
        w = self.weights.unsqueeze(0).expand(b, -1)
        logits = torch.log(w + 1e-9)
        return w, logits


# =========================================================================
# GatingNetwork Tests
# =========================================================================

def test_gating_network_init() -> None:
    """Verify GatingNetwork architectural layers and parameter counting."""
    gate = GatingNetwork(in_channels=3, channels=(32, 64, 128, 256), num_classes=4, dropout=0.2)
    assert len(gate.features) == 16  # 4 stages * 4 layers (Conv, BN, LeakyReLU, MaxPool)
    assert isinstance(gate.gap, nn.AdaptiveAvgPool2d)
    assert isinstance(gate.dropout, nn.Dropout)
    assert gate.dropout.p == 0.2
    assert gate.fc.in_features == 256
    assert gate.fc.out_features == 4

    total_params = gate.count_parameters(only_trainable=False)
    trainable_params = gate.count_parameters(only_trainable=True)
    assert total_params == trainable_params
    assert total_params > 380_000


def test_gating_network_forward_shapes() -> None:
    """Verify GatingNetwork outputs correct tensor shapes for batch sizes 1 and 4."""
    gate = GatingNetwork()

    # Batch size 1
    x1 = torch.rand(1, 3, 128, 128)
    w1, z1 = gate(x1, tau=1.0)
    assert w1.shape == (1, 4)
    assert z1.shape == (1, 4)

    # Batch size 4
    x4 = torch.rand(4, 3, 128, 128)
    w4, z4 = gate(x4, tau=1.0)
    assert w4.shape == (4, 4)
    assert z4.shape == (4, 4)

    # 3D unbatched input (3, 128, 128)
    x3d = torch.rand(3, 128, 128)
    w3d, z3d = gate(x3d, tau=1.0)
    assert w3d.shape == (1, 4)
    assert z3d.shape == (1, 4)


def test_gating_network_temperature_scaling_effect() -> None:
    """Verify temperature scaling modulates routing entropy and sharpness."""
    gate = GatingNetwork()
    x = torch.rand(4, 3, 128, 128)

    # Low temperature (sharper distribution -> lower entropy)
    w_cold, _ = gate(x, tau=0.2)
    entropy_cold = -(w_cold * torch.log(w_cold + 1e-9)).sum(dim=-1).mean()

    # High temperature (softer distribution -> higher entropy)
    w_hot, _ = gate(x, tau=5.0)
    entropy_hot = -(w_hot * torch.log(w_hot + 1e-9)).sum(dim=-1).mean()

    # Max probability under cold should exceed max probability under hot
    assert w_cold.max(dim=-1).values.mean() > w_hot.max(dim=-1).values.mean()
    assert entropy_hot.item() > entropy_cold.item()


def test_gating_network_temperature_clamping() -> None:
    """Verify temperature clamping at 0.05 prevents division-by-zero and NaN/Inf."""
    gate = GatingNetwork()
    gate.eval()
    x = torch.rand(2, 3, 128, 128)

    # Extreme low temperatures
    for invalid_tau in [0.001, 0.00001, 0.0, -1.0]:
        w, z = gate(x, tau=invalid_tau)
        assert not torch.isnan(w).any()
        assert not torch.isinf(w).any()
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()

    # Clamping at 0.05 must yield exact same result as passing 0.05
    w_clamped, _ = gate(x, tau=0.0001)
    w_ref, _ = gate(x, tau=0.05)
    assert torch.allclose(w_clamped, w_ref, atol=1e-6)


def test_softmax_probability_constraint() -> None:
    """Verify routing weights form a strictly valid probability simplex."""
    gate = GatingNetwork()
    x = torch.rand(8, 3, 128, 128)

    for tau in [0.1, 0.5, 1.0, 2.0]:
        w, _ = gate(x, tau=tau)
        # Check non-negativity
        assert (w >= 0.0).all()
        assert (w <= 1.0).all()
        # Check sum to 1.0 +/- 1e-6
        row_sums = w.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_gating_network_checkpoint_loading() -> None:
    """Verify GatingNetwork loads pretrained weights from Task 2 classifier checkpoint."""
    if not CLASSIFIER_CKPT.is_file():
        pytest.skip(f"Checkpoint {CLASSIFIER_CKPT} not found")

    gate = build_gating_network(checkpoint_path=CLASSIFIER_CKPT)
    x = torch.rand(2, 3, 128, 128)
    w, z = gate(x)
    assert w.shape == (2, 4)
    assert z.shape == (2, 4)

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        gate.load_from_classifier_checkpoint("non_existent_classifier.pt")


# =========================================================================
# SoftMoE Tests
# =========================================================================

def test_soft_moe_init() -> None:
    """Verify SoftMoE module hierarchy and specialist properties."""
    model = SoftMoE()
    assert isinstance(model.gate, GatingNetwork)
    assert isinstance(model.specialist_salt, SpecialistAutoencoder)
    assert isinstance(model.specialist_blur, SpecialistAutoencoder)
    assert isinstance(model.specialist_occlusion, SpecialistAutoencoder)

    # Aliases
    assert model.salt_expert is model.specialist_salt
    assert model.blur_expert is model.specialist_blur
    assert model.occlusion_expert is model.specialist_occlusion


def test_soft_moe_forward_shapes() -> None:
    """Verify SoftMoE forward pass output shapes for batch sizes 1 and 4."""
    model = SoftMoE()

    # Batch size 1
    x1 = torch.rand(1, 3, 128, 128)
    restored1, w1, z1 = model(x1, tau=1.0)
    assert restored1.shape == (1, 3, 128, 128)
    assert w1.shape == (1, 4)
    assert z1.shape == (1, 4)
    assert torch.allclose(w1.sum(dim=-1), torch.ones(1), atol=1e-6)

    # Batch size 4
    x4 = torch.rand(4, 3, 128, 128)
    restored4, w4, z4 = model(x4, tau=1.0)
    assert restored4.shape == (4, 3, 128, 128)
    assert w4.shape == (4, 4)
    assert z4.shape == (4, 4)
    assert torch.allclose(w4.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_differentiable_gradient_flow() -> None:
    """Verify non-zero gradients flow through convex blending to gate and specialists."""
    model = SoftMoE()
    model.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128, requires_grad=True)
    restored, routing_weights, logits = model(x, tau=1.0)

    # Multi-component loss
    loss = restored.mean() + logits.mean()
    loss.backward()

    # 1. Gate gradients must be populated and non-zero
    assert model.gate.fc.weight.grad is not None
    assert model.gate.fc.weight.grad.abs().sum().item() > 0.0

    # 2. Specialist gradients must be populated and non-zero
    assert model.specialist_salt.decoder.output_head[0].weight.grad is not None
    assert model.specialist_salt.decoder.output_head[0].weight.grad.abs().sum().item() > 0.0

    assert model.specialist_blur.decoder.output_head[0].weight.grad is not None
    assert model.specialist_blur.decoder.output_head[0].weight.grad.abs().sum().item() > 0.0

    assert model.specialist_occlusion.decoder.output_head[0].weight.grad is not None
    assert model.specialist_occlusion.decoder.output_head[0].weight.grad.abs().sum().item() > 0.0


def test_expert_freezing_and_unfreezing() -> None:
    """Verify specialist freezing toggles requires_grad while leaving gate active."""
    model = SoftMoE()

    # Freeze specialists (Warm-up phase)
    model.set_experts_frozen(True)
    assert model.experts_frozen is True

    for specialist in (model.specialist_salt, model.specialist_blur, model.specialist_occlusion):
        for p in specialist.parameters():
            assert not p.requires_grad

    # Gate parameters must remain trainable
    assert any(p.requires_grad for p in model.gate.parameters())

    # Forward + backward with frozen experts
    x = torch.rand(2, 3, 128, 128)
    restored, _, logits = model(x)
    loss = restored.mean() + logits.mean()
    loss.backward()

    assert model.gate.fc.weight.grad is not None
    assert model.specialist_salt.decoder.output_head[0].weight.grad is None
    assert model.specialist_blur.decoder.output_head[0].weight.grad is None
    assert model.specialist_occlusion.decoder.output_head[0].weight.grad is None

    # Unfreeze specialists (Joint Fine-tuning phase)
    model.set_experts_frozen(False)
    assert model.experts_frozen is False

    for specialist in (model.specialist_salt, model.specialist_blur, model.specialist_occlusion):
        for p in specialist.parameters():
            assert p.requires_grad


def test_identity_branch_preservation() -> None:
    """Verify bit-exact clean image preservation when routing weight w_1 = 1.0."""
    model = SoftMoE()
    # Replace gate with deterministic mock routing 100% to clean branch
    model.gate = MockTargetGate(target_weights=[1.0, 0.0, 0.0, 0.0])

    x = torch.rand(2, 3, 128, 128)
    restored, w, _ = model(x)

    assert torch.allclose(w, torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    # With w_1 = 1.0, restored must be bit-exact identical to input x
    assert torch.allclose(restored, x, atol=1e-7)


def test_convex_combination_mathematical_blending() -> None:
    """Mathematically verify that convex blending performs sum_k w_k * b_k accurately."""
    # Use mock specialists with known constant outputs
    s_salt = MockConstantSpecialist(2.0)
    s_blur = MockConstantSpecialist(3.0)
    s_occ = MockConstantSpecialist(4.0)

    # Input tensor filled with 1.0 (Identity branch = 1.0)
    x = torch.full((1, 3, 128, 128), 1.0)

    # Weights: [0.1, 0.2, 0.3, 0.4] -> Expected: 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 = 3.0
    mock_gate = MockTargetGate(target_weights=[0.1, 0.2, 0.3, 0.4])

    model = SoftMoE(
        gate=mock_gate,  # type: ignore[arg-type]
        specialist_salt=s_salt,  # type: ignore[arg-type]
        specialist_blur=s_blur,  # type: ignore[arg-type]
        specialist_occlusion=s_occ,  # type: ignore[arg-type]
    )

    restored, w, _ = model(x)
    expected = torch.full_like(x, 3.0)
    assert torch.allclose(restored, expected, atol=1e-6)


def test_load_pretrained_components_all() -> None:
    """Verify loading real Task 2 checkpoints into SoftMoE without shape mismatch."""
    for ckpt in [CLASSIFIER_CKPT, SALT_CKPT, BLUR_CKPT, OCCLUSION_CKPT]:
        if not ckpt.is_file():
            pytest.skip(f"Checkpoint {ckpt} not found on disk")

    model = build_soft_moe(
        classifier_path=CLASSIFIER_CKPT,
        salt_path=SALT_CKPT,
        blur_path=BLUR_CKPT,
        occlusion_path=OCCLUSION_CKPT,
    )

    x = torch.rand(2, 3, 128, 128)
    restored, w, z = model(x)
    assert restored.shape == (2, 3, 128, 128)
    assert w.shape == (2, 4)
    assert z.shape == (2, 4)
    assert (restored >= -1.0).all()  # Specialist output valid numerical range


def test_checkpoint_saving_and_metadata(tmp_path: Path) -> None:
    """Verify SoftMoE checkpoint saving and metadata serialization."""
    model = SoftMoE()
    save_file = tmp_path / "moe_test.pth"
    model.save_checkpoint(
        save_file,
        epoch=5,
        val_loss=0.123,
        metrics={"psnr": 24.5, "ssim": 0.65},
        extra={"note": "unit-test"},
    )

    assert save_file.is_file()
    payload = torch.load(save_file, weights_only=False)
    assert "model_state_dict" in payload
    assert payload["epoch"] == 5
    assert payload["val_loss"] == 0.123
    assert payload["metrics"]["psnr"] == 24.5
    assert payload["extra"]["note"] == "unit-test"


def test_parameter_counts_frozen_vs_unfrozen() -> None:
    """Verify trainable parameter counts match expected values when frozen vs unfrozen."""
    model = SoftMoE()
    total_all = model.count_parameters(only_trainable=False)
    total_unfrozen = model.count_parameters(only_trainable=True)
    assert total_all == total_unfrozen
    assert total_all > 15_000_000  # Gate + 3 specialists ~ 15.13M parameters

    # When frozen, only gate parameters remain trainable (~390k)
    model.set_experts_frozen(True)
    total_frozen = model.count_parameters(only_trainable=True)
    gate_trainable = model.gate.count_parameters(only_trainable=True)
    assert total_frozen == gate_trainable
    assert total_frozen < 500_000

    # Unfreeze restores all parameters
    model.set_experts_frozen(False)
    assert model.count_parameters(only_trainable=True) == total_all


def test_soft_moe_3d_unbatched_input() -> None:
    """Verify SoftMoE handles single unbatched 3D tensor (3, 128, 128)."""
    model = SoftMoE()
    x = torch.rand(3, 128, 128)
    restored, w, z = model(x, tau=1.0)
    assert restored.shape == (1, 3, 128, 128)
    assert w.shape == (1, 4)
    assert z.shape == (1, 4)


def test_soft_moe_pretrained_missing_file_raises() -> None:
    """Verify loading non-existent specialist checkpoint raises FileNotFoundError."""
    model = SoftMoE()
    with pytest.raises(FileNotFoundError):
        model.load_pretrained_components(
            classifier_path=CLASSIFIER_CKPT if CLASSIFIER_CKPT.is_file() else "fake.pt",
            salt_path="non_existent_salt.pt",
        )


def test_soft_moe_l1_reconstruction_loss_gradients() -> None:
    """Verify realistic L1 reconstruction loss computes valid gradients across all branches."""
    model = SoftMoE()
    model.set_experts_frozen(False)

    x_corrupt = torch.rand(2, 3, 128, 128, requires_grad=True)
    x_clean = torch.rand(2, 3, 128, 128)

    restored, w, z = model(x_corrupt, tau=1.0)
    l1_loss = F.l1_loss(restored, x_clean)
    l1_loss.backward()

    # Gate feature and head gradients
    assert model.gate.features[0].weight.grad is not None
    assert model.gate.fc.weight.grad is not None

    # Specialist decoders gradients
    assert model.specialist_salt.decoder.output_head[0].weight.grad is not None
    assert model.specialist_blur.decoder.output_head[0].weight.grad is not None
    assert model.specialist_occlusion.decoder.output_head[0].weight.grad is not None
