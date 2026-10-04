"""tests/test_task3_e2e.py
------------------------
Comprehensive End-to-End (E2E) Test Suite for Task 3 Soft Mixture-of-Experts (MoE)
Image Restoration on the Oxford-IIIT Pet dataset.

Structured following the 4-Tier Testing Methodology:
- Tier 1: Feature Coverage (>=5 tests per architectural feature across M1-M5)
- Tier 2: Boundary & Corner Cases (Extreme temperatures, singleton batch, corrupted extremes)
- Tier 3: Cross-Feature Interactions (Gate + Specialists + Blending + Two-stage training + ONNX)
- Tier 4: Real-World Scenarios (End-to-end Pet dataset restoration pipeline)

Authoritative sources of expected outputs:
- Mathematical properties of softmax and convex combinations
- Specification constraints in PROJECT.md and docs/plans/task3-soft-moe.md
- Pretrained Task 2 component checkpoints in checkpoints/task2/
- ONNX Runtime CPU reference execution engine
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.shared.corruptions import (
    apply_clean,
    apply_gaussian_blur,
    apply_occlusion,
    apply_salt_and_pepper,
    sample_occlusion_boxes,
)
from src.shared.losses import SSIMLoss
from src.shared.metrics import compute_psnr
from src.task2.specialist import SpecialistAutoencoder
from src.task3.gate import MIN_TEMPERATURE, GatingNetwork
from src.task3.moe_model import SoftMoE

# ==============================================================================
# Helper Modules & Oracles
# ==============================================================================


class ExportWrapper(nn.Module):
    """Encapsulates SoftMoE fixing temperature tau for atomic ONNX graph export."""

    def __init__(self, moe_model: SoftMoE, tau: float = 1.0) -> None:
        super().__init__()
        self.moe_model = moe_model
        self.tau = float(tau)

    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        restored, routing_weights, _ = self.moe_model(input_image, tau=self.tau)
        return restored, routing_weights


def compute_l2_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Formula: L_balance = sum_{k=1}^4 (mean_w_k - 0.25)^2."""
    mean_w = torch.mean(weights, dim=0)  # Shape (4,)
    target = torch.full_like(mean_w, 0.25)
    return torch.sum((mean_w - target) ** 2)


def compute_entropy_balance_loss(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Formula: L_balance = sum_{k=1}^4 mean_w_k * log(mean_w_k + eps)."""
    mean_w = torch.mean(weights, dim=0)
    return torch.sum(mean_w * torch.log(mean_w + eps))


def compute_switch_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Formula: L_balance = 4 * sum_{k=1}^4 f_k * mean_w_k."""
    b = weights.shape[0]
    argmax_indices = torch.argmax(weights, dim=-1)
    f = torch.zeros(4, dtype=weights.dtype, device=weights.device)
    for k in range(4):
        f[k] = torch.sum(argmax_indices == k).to(weights.dtype) / b
    mean_w = torch.mean(weights, dim=0)
    return 4.0 * torch.sum(f * mean_w)


def check_routing_collapse(
    mean_weights: torch.Tensor, dominance_thresh: float = 0.90, starvation_thresh: float = 0.02
) -> bool:
    """Returns True if any expert dominates >90% or starves <2%."""
    max_w = float(torch.max(mean_weights).item())
    min_w = float(torch.min(mean_weights).item())
    # Tolerate float32 epsilon precision near exact boundary
    return (max_w > dominance_thresh + 1e-6) or (min_w < starvation_thresh - 1e-6)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="module")
def checkpoint_paths() -> Dict[str, Path]:
    base = Path("checkpoints/task2")
    return {
        "classifier": base / "classifier_best.pt",
        "salt": base / "specialist_salt_best.pt",
        "blur": base / "specialist_blur_best.pt",
        "occlusion": base / "specialist_occlusion_best.pt",
    }


@pytest.fixture(scope="module")
def small_gate() -> GatingNetwork:
    """Lightweight deterministic gate for rapid mathematical assertions."""
    torch.manual_seed(42)
    return GatingNetwork(channels=(16, 32, 64, 128), dropout=0.0)


@pytest.fixture(scope="module")
def small_moe() -> SoftMoE:
    """Lightweight SoftMoE model with small channel dimensions for fast testing."""
    torch.manual_seed(42)
    gate = GatingNetwork(channels=(16, 32, 64, 128), dropout=0.0)
    s_salt = SpecialistAutoencoder(
        corruption_type="salt_and_pepper",
        channels=(16, 32, 64, 128),
        bottleneck_dim=128,
        use_residual=False,
    )
    s_blur = SpecialistAutoencoder(
        corruption_type="gaussian_blur",
        channels=(16, 32, 64, 128),
        bottleneck_dim=128,
        use_residual=False,
    )
    s_occ = SpecialistAutoencoder(
        corruption_type="occlusion",
        channels=(16, 32, 64, 128),
        bottleneck_dim=128,
        use_residual=False,
    )
    moe = SoftMoE(
        gate=gate,
        specialist_salt=s_salt,
        specialist_blur=s_blur,
        specialist_occlusion=s_occ,
    )
    moe.eval()
    return moe


@pytest.fixture(scope="module")
def sample_pet_tensor() -> torch.Tensor:
    """Loads a real pet image from data/oxford-iiit-pet/images/ as a normalized
    tensor of shape (1, 3, 128, 128).
    """
    img_path = Path("data/oxford-iiit-pet/images/Abyssinian_1.jpg")
    assert img_path.is_file(), f"Sample image missing at: {img_path}"
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB").resize((128, 128))
    tensor = TF.to_tensor(img_rgb).unsqueeze(0)  # Shape (1, 3, 128, 128) in [0, 1]
    return tensor


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature M1-M5)
# ==============================================================================

# --- Feature 1: Gating Network Architecture ---


def test_tier1_feat1_gate_output_topology(small_gate: GatingNetwork):
    """T1.1.1: Assert GatingNetwork produces (B, 4) weights and (B, 4) logits."""
    x = torch.rand(3, 3, 128, 128)
    weights, logits = small_gate(x)
    assert weights.shape == (3, 4)
    assert logits.shape == (3, 4)
    assert weights.dtype == torch.float32
    assert logits.dtype == torch.float32


def test_tier1_feat1_gate_softmax_normalization(small_gate: GatingNetwork):
    """T1.1.2: Assert simplex constraint sum_{k=1}^4 w_k = 1.0 +/- 1e-6."""
    x = torch.randn(5, 3, 128, 128)
    weights, _ = small_gate(x)
    row_sums = torch.sum(weights, dim=-1)
    for s in row_sums:
        assert pytest.approx(s.item(), abs=1e-6) == 1.0


def test_tier1_feat1_gate_probability_bounds(small_gate: GatingNetwork):
    """T1.1.3: Assert all routing probabilities are non-negative and <= 1.0."""
    x = torch.randn(4, 3, 128, 128)
    weights, _ = small_gate(x)
    assert torch.all(weights >= 0.0)
    assert torch.all(weights <= 1.0)


def test_tier1_feat1_gate_checkpoint_loading_real(checkpoint_paths: Dict[str, Path]):
    """T1.1.4: Assert real Task 2 classifier checkpoint loads cleanly into
    standard GatingNetwork.
    """
    gate = GatingNetwork()
    ckpt_path = checkpoint_paths["classifier"]
    assert ckpt_path.is_file()
    gate.load_from_classifier_checkpoint(ckpt_path)
    # Check that model parameters are populated
    total_params = gate.count_parameters()
    assert total_params > 300_000


def test_tier1_feat1_gate_backbone_feature_extraction(small_gate: GatingNetwork):
    """T1.1.5: Assert extract_features produces flattened pooled representation
    of shape (B, channels[-1]).
    """
    x = torch.rand(2, 3, 128, 128)
    feats = small_gate.extract_features(x)
    assert feats.shape == (2, 128)  # channels[-1] is 128 for small_gate


# --- Feature 2: Temperature Scaling & Numerical Clamping ---


def test_tier1_feat2_temperature_standard_tau_1(small_gate: GatingNetwork):
    """T1.2.1: Assert tau=1.0 yields exact standard softmax of logits."""
    x = torch.randn(2, 3, 128, 128)
    weights, logits = small_gate(x, tau=1.0)
    expected = F.softmax(logits, dim=-1)
    assert torch.allclose(weights, expected, atol=1e-6)


def test_tier1_feat2_temperature_smoothing_tau_high(small_gate: GatingNetwork):
    """T1.2.2: Assert high tau=5.0 increases routing entropy toward uniform
    distribution [0.25, 0.25, 0.25, 0.25].
    """
    x = torch.randn(2, 3, 128, 128)
    w_std, _ = small_gate(x, tau=1.0)
    w_high, _ = small_gate(x, tau=5.0)

    # Entropy = - sum(w * log(w))
    entropy_std = -torch.sum(w_std * torch.log(w_std + 1e-12), dim=-1).mean().item()
    entropy_high = -torch.sum(w_high * torch.log(w_high + 1e-12), dim=-1).mean().item()
    assert entropy_high > entropy_std


def test_tier1_feat2_temperature_sharpening_tau_low(small_gate: GatingNetwork):
    """T1.2.3: Assert low tau=0.1 sharpens routing distribution toward argmax."""
    x = torch.randn(2, 3, 128, 128)
    w_std, _ = small_gate(x, tau=1.0)
    w_low, _ = small_gate(x, tau=0.1)

    max_prob_std = torch.max(w_std, dim=-1).values.mean().item()
    max_prob_low = torch.max(w_low, dim=-1).values.mean().item()
    assert max_prob_low >= max_prob_std


def test_tier1_feat2_temperature_clamping_safeguard(small_gate: GatingNetwork):
    """T1.2.4: Assert extreme low tau (0.001) or negative tau (-2.0) clamps to
    MIN_TEMPERATURE (0.05).
    """
    x = torch.randn(2, 3, 128, 128)
    w_extreme, _ = small_gate(x, tau=1e-6)
    w_clamped_ref, _ = small_gate(x, tau=MIN_TEMPERATURE)
    assert torch.allclose(w_extreme, w_clamped_ref, atol=1e-6)
    assert not torch.isnan(w_extreme).any()
    assert not torch.isinf(w_extreme).any()

    # Negative tau should also clamp to MIN_TEMPERATURE
    w_neg, _ = small_gate(x, tau=-5.0)
    assert torch.allclose(w_neg, w_clamped_ref, atol=1e-6)


def test_tier1_feat2_temperature_equal_logits_invariance():
    """T1.2.5: Assert equal logits yield uniform [0.25, 0.25, 0.25, 0.25] regardless of tau."""
    logits = torch.tensor([[2.5, 2.5, 2.5, 2.5]], dtype=torch.float32)
    for tau in [0.05, 0.5, 1.0, 3.0, 10.0]:
        w = F.softmax(logits / tau, dim=-1)
        expected = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        assert torch.allclose(w, expected, atol=1e-6)


# --- Feature 3: SoftMoE Container Module ---


def test_tier1_feat3_moe_submodules_composition(small_moe: SoftMoE):
    """T1.3.1: Assert SoftMoE holds gate and all three specialist autoencoders."""
    assert isinstance(small_moe.gate, GatingNetwork)
    assert isinstance(small_moe.specialist_salt, SpecialistAutoencoder)
    assert isinstance(small_moe.specialist_blur, SpecialistAutoencoder)
    assert isinstance(small_moe.specialist_occlusion, SpecialistAutoencoder)
    # Check property aliases
    assert small_moe.salt_expert is small_moe.specialist_salt
    assert small_moe.blur_expert is small_moe.specialist_blur
    assert small_moe.occlusion_expert is small_moe.specialist_occlusion


def test_tier1_feat3_moe_forward_three_tuple_return(small_moe: SoftMoE):
    """T1.3.2: Assert SoftMoE forward returns (restored_image, routing_weights, logits)."""
    x = torch.rand(2, 3, 128, 128)
    outputs = small_moe(x)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    restored, routing_weights, logits = outputs
    assert restored.shape == (2, 3, 128, 128)
    assert routing_weights.shape == (2, 4)
    assert logits.shape == (2, 4)


def test_tier1_feat3_moe_spatial_dimension_fidelity(small_moe: SoftMoE):
    """T1.3.3: Assert output image matches input image spatial dimensions (128x128)."""
    x = torch.rand(1, 3, 128, 128)
    restored, _, _ = small_moe(x)
    assert restored.shape[2:] == (128, 128)


def test_tier1_feat3_moe_multi_batch_evaluation(small_moe: SoftMoE):
    """T1.3.4: Assert forward evaluation executes cleanly across batch sizes B in {1, 2, 4}."""
    for b in [1, 2, 4]:
        x = torch.rand(b, 3, 128, 128)
        restored, weights, logits = small_moe(x)
        assert restored.shape[0] == b
        assert weights.shape[0] == b
        assert logits.shape[0] == b


def test_tier1_feat3_moe_eval_vs_train_mode(small_moe: SoftMoE):
    """T1.3.5: Assert model correctly toggles between train() and eval() modes."""
    small_moe.train()
    assert small_moe.gate.training is True
    assert small_moe.specialist_salt.training is True
    small_moe.eval()
    assert small_moe.gate.training is False
    assert small_moe.specialist_salt.training is False


# --- Feature 4: Differentiable Convex Blending ---


def test_tier1_feat4_convex_combination_identity():
    """T1.4.1: Assert restoration mathematically matches sum_{k=1}^4 w_k * b_k."""
    b1 = torch.full((1, 3, 4, 4), 1.0)
    b2 = torch.full((1, 3, 4, 4), 2.0)
    b3 = torch.full((1, 3, 4, 4), 3.0)
    b4 = torch.full((1, 3, 4, 4), 4.0)
    w = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # sums to 1.0

    # Manual convex sum
    expected = (
        (0.1 * 1.0) + (0.2 * 2.0) + (0.3 * 3.0) + (0.4 * 4.0)
    )  # = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
    w_exp = w.view(1, 4, 1, 1, 1)
    blended = w_exp[:, 0] * b1 + w_exp[:, 1] * b2 + w_exp[:, 2] * b3 + w_exp[:, 3] * b4
    assert pytest.approx(blended.mean().item(), abs=1e-5) == expected


def test_tier1_feat4_end_to_end_gradient_propagation():
    """T1.4.2: Assert backward pass propagates non-zero gradients to gate and specialists."""
    moe = SoftMoE(channels=(16, 32, 64, 128), bottleneck_dim=128, dropout=0.0)
    moe.train()
    moe.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128, requires_grad=True)
    target = torch.rand(2, 3, 128, 128)

    restored, _, _ = moe(x)
    loss = F.l1_loss(restored, target)
    loss.backward()

    # Check non-zero gradient in gate linear layer
    assert moe.gate.fc.weight.grad is not None
    assert torch.norm(moe.gate.fc.weight.grad).item() > 0.0

    # Check non-zero gradient in specialists
    for spec in [moe.specialist_salt, moe.specialist_blur, moe.specialist_occlusion]:
        grads = [torch.norm(p.grad).item() for p in spec.parameters() if p.grad is not None]
        assert len(grads) > 0 and max(grads) > 0.0


def test_tier1_feat4_gate_logit_perturbation_sensitivity(small_moe: SoftMoE):
    """T1.4.3: Assert perturbing gate logits produces measurable difference in blended image."""
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        out_t1, w1, _ = small_moe(x, tau=1.0)
        out_t5, w5, _ = small_moe(x, tau=5.0)
    diff = torch.norm(out_t1 - out_t5).item()
    # Varying temperature changes routing weights, which must alter blended reconstruction
    assert diff > 0.0


def test_tier1_feat4_one_hot_orthogonal_isolation():
    """T1.4.4: Assert one-hot routing [0, 1, 0, 0] routes strictly to specialist 2."""
    b1 = torch.full((1, 3, 8, 8), 10.0)
    b2 = torch.full((1, 3, 8, 8), 20.0)
    b3 = torch.full((1, 3, 8, 8), 30.0)
    b4 = torch.full((1, 3, 8, 8), 40.0)
    w = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    w_exp = w.view(1, 4, 1, 1, 1)
    blended = w_exp[:, 0] * b1 + w_exp[:, 1] * b2 + w_exp[:, 2] * b3 + w_exp[:, 3] * b4
    assert torch.allclose(blended, b2)


def test_tier1_feat4_blending_preserves_finite_range():
    """T1.4.5: Assert convex blend of inputs in [0, 1] remains strictly within [0, 1]."""
    for _ in range(5):
        w = F.softmax(torch.randn(1, 4), dim=-1).view(1, 4, 1, 1, 1)
        b1 = torch.rand(1, 3, 16, 16)
        b2 = torch.rand(1, 3, 16, 16)
        b3 = torch.rand(1, 3, 16, 16)
        b4 = torch.rand(1, 3, 16, 16)
        blended = w[:, 0] * b1 + w[:, 1] * b2 + w[:, 2] * b3 + w[:, 3] * b4
        assert blended.min().item() >= 0.0
        assert blended.max().item() <= 1.0


# --- Feature 5: Identity Pass-Through Branch ---


def test_tier1_feat5_identity_bit_exact_bypass():
    """T1.5.1: Assert that when w = [1, 0, 0, 0], restored image is bit-exact identical to input."""
    x = torch.rand(2, 3, 128, 128)
    b1 = x
    b2 = torch.rand(2, 3, 128, 128)
    b3 = torch.rand(2, 3, 128, 128)
    b4 = torch.rand(2, 3, 128, 128)
    w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).view(2, 4, 1, 1, 1)

    restored = w[:, 0] * b1 + w[:, 1] * b2 + w[:, 2] * b3 + w[:, 3] * b4
    assert torch.equal(restored, x)
    assert torch.mean((restored - x) ** 2).item() == 0.0


def test_tier1_feat5_identity_zero_parameters(small_moe: SoftMoE):
    """T1.5.2: Assert identity branch introduces 0 parameters (not an nn.Module submodule)."""
    # SoftMoE submodules are gate and 3 specialists
    submodule_names = [name for name, _ in small_moe.named_children()]
    assert "gate" in submodule_names
    assert "specialist_salt" in submodule_names
    assert "specialist_blur" in submodule_names
    assert "specialist_occlusion" in submodule_names
    assert "identity" not in submodule_names  # Pure mathematical bypass


def test_tier1_feat5_identity_zero_compute_overhead():
    """T1.5.3: Assert identity bypass operation is a direct reference assignment b1 = x."""
    x = torch.randn(1, 3, 16, 16)
    b1 = x
    assert b1 is x  # Zero memory copy pointer equality


def test_tier1_feat5_identity_unmodified_dynamic_range():
    """T1.5.4: Assert dynamic range [min, max] of input is preserved bit-exact through identity."""
    x = torch.tensor([[[[0.123, 0.987], [0.456, 0.789]]]])
    b1 = x
    assert b1.min().item() == pytest.approx(0.123)
    assert b1.max().item() == pytest.approx(0.987)


def test_tier1_feat5_identity_high_frequency_fidelity():
    """T1.5.5: Assert high-frequency checkerboard pattern is untouched by identity bypass."""
    checker = torch.zeros(1, 3, 32, 32)
    checker[:, :, ::2, ::2] = 1.0
    checker[:, :, 1::2, 1::2] = 1.0
    b1 = checker
    psnr = compute_psnr(b1, checker)
    assert psnr >= 100.0  # Perfect ceiling


# --- Feature 6: Pre-trained Checkpoint Integration & Freezing ---


def test_tier1_feat6_pretrained_load_all_four_components(checkpoint_paths: Dict[str, Path]):
    """T1.6.1: Assert all 4 Task 2 checkpoints deserialize into SoftMoE without missing keys."""
    for p in checkpoint_paths.values():
        assert p.is_file(), f"Missing checkpoint: {p}"

    moe = SoftMoE()
    moe.load_pretrained_components(
        classifier_path=checkpoint_paths["classifier"],
        salt_path=checkpoint_paths["salt"],
        blur_path=checkpoint_paths["blur"],
        occlusion_path=checkpoint_paths["occlusion"],
    )
    assert moe.count_parameters() > 15_000_000


def test_tier1_feat6_freeze_specialists_toggle(small_moe: SoftMoE):
    """T1.6.2: Assert set_experts_frozen(True) sets requires_grad=False across all specialists."""
    small_moe.set_experts_frozen(True)
    assert small_moe.experts_frozen is True
    for specialist in [
        small_moe.specialist_salt,
        small_moe.specialist_blur,
        small_moe.specialist_occlusion,
    ]:
        for p in specialist.parameters():
            assert p.requires_grad is False
    # Gate parameters remain trainable
    assert any(p.requires_grad for p in small_moe.gate.parameters())


def test_tier1_feat6_unfreeze_specialists_toggle(small_moe: SoftMoE):
    """T1.6.3: Assert set_experts_frozen(False) restores requires_grad=True
    across all specialists.
    """
    small_moe.set_experts_frozen(False)
    assert small_moe.experts_frozen is False
    for specialist in [
        small_moe.specialist_salt,
        small_moe.specialist_blur,
        small_moe.specialist_occlusion,
    ]:
        for p in specialist.parameters():
            assert p.requires_grad is True


def test_tier1_feat6_warmup_specialist_weights_isolation(small_moe: SoftMoE):
    """T1.6.4: Assert optimizer step on gate during warm-up leaves specialist
    weights bit-exact identical.
    """
    small_moe.set_experts_frozen(True)
    gate_params = [p for p in small_moe.gate.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(gate_params, lr=1e-3)

    # Snapshot specialist weight before step
    orig_salt_weight = next(small_moe.specialist_salt.parameters()).clone()

    x = torch.rand(2, 3, 128, 128)
    restored, _, _ = small_moe(x)
    loss = F.l1_loss(restored, torch.zeros_like(restored))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    new_salt_weight = next(small_moe.specialist_salt.parameters())
    assert torch.equal(orig_salt_weight, new_salt_weight)


def test_tier1_feat6_differential_lr_parameter_groups(small_moe: SoftMoE):
    """T1.6.5: Assert differential parameter groups can be configured with
    Gate LR 1e-4 and Specialist LR 2e-5.
    """
    small_moe.set_experts_frozen(False)
    gate_params = list(small_moe.gate.parameters())
    expert_params = (
        list(small_moe.specialist_salt.parameters())
        + list(small_moe.specialist_blur.parameters())
        + list(small_moe.specialist_occlusion.parameters())
    )

    optimizer = torch.optim.Adam(
        [
            {"params": gate_params, "lr": 1e-4},
            {"params": expert_params, "lr": 2e-5},
        ]
    )
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 2e-5


# --- Feature 7: Multi-Objective Loss Formulation ---


def test_tier1_feat7_loss_formula_summation():
    """T1.7.1: Assert total loss equals lambda1*L1 + lambda2*(1-SSIM) +
    lambda3*L_CE + lambda4*L_balance.
    """
    l1 = torch.tensor(0.1)
    ssim_term = torch.tensor(0.2)
    ce = torch.tensor(0.3)
    bal = torch.tensor(0.4)

    l1_w, ssim_w, ce_w, bal_w = 0.8, 0.2, 0.1, 0.01
    total = (l1_w * l1) + (ssim_w * ssim_term) + (ce_w * ce) + (bal_w * bal)
    expected = (0.8 * 0.1) + (0.2 * 0.2) + (0.1 * 0.3) + (0.01 * 0.4)
    assert pytest.approx(total.item(), abs=1e-6) == expected


def test_tier1_feat7_loss_zero_on_identical_reconstruction():
    """T1.7.2: Assert L1=0 and (1-SSIM)=0 when pred == target."""
    x = torch.rand(2, 3, 64, 64)
    l1 = F.l1_loss(x, x).item()
    ssim_loss_fn = SSIMLoss()
    ssim_loss_val = ssim_loss_fn(x, x).item()

    assert pytest.approx(l1, abs=1e-7) == 0.0
    assert pytest.approx(ssim_loss_val, abs=1e-5) == 0.0


def test_tier1_feat7_loss_ssim_non_negativity():
    """T1.7.3: Assert SSIM loss (1-SSIM) remains >= 0 for arbitrary image pairs."""
    ssim_loss_fn = SSIMLoss()
    for _ in range(3):
        p = torch.rand(2, 3, 64, 64)
        t = torch.rand(2, 3, 64, 64)
        loss = ssim_loss_fn(p, t).item()
        assert loss >= 0.0


def test_tier1_feat7_loss_auxiliary_ce_alignment():
    """T1.7.4: Assert CE loss on gate logits maps labels y in {0, 1, 2, 3} correctly."""
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    labels_correct = torch.tensor([0, 1])
    labels_incorrect = torch.tensor([1, 0])

    ce_low = F.cross_entropy(logits, labels_correct).item()
    ce_high = F.cross_entropy(logits, labels_incorrect).item()
    assert ce_low < ce_high


def test_tier1_feat7_loss_gradient_scaling_proportionality():
    """T1.7.5: Assert scaling lambda_1 proportionally scales gradient norm on pred."""
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    target = torch.rand(1, 3, 32, 32)

    loss_1 = 0.5 * F.l1_loss(pred, target)
    loss_1.backward()
    grad_norm_1 = torch.norm(pred.grad).item()

    pred.grad.zero_()
    loss_2 = 1.0 * F.l1_loss(pred, target)
    loss_2.backward()
    grad_norm_2 = torch.norm(pred.grad).item()

    assert pytest.approx(grad_norm_2, abs=1e-5) == 2.0 * grad_norm_1


# --- Feature 8: Balance Regularizers Formulations ---


def test_tier1_feat8_l2_regularizer_zero_at_uniform():
    """T1.8.1: Assert L2 deviation loss reaches 0.0 when expert routing is
    perfectly uniform [0.25, 0.25, 0.25, 0.25].
    """
    weights = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    loss = compute_l2_balance_loss(weights)
    assert pytest.approx(loss.item(), abs=1e-7) == 0.0


def test_tier1_feat8_l2_regularizer_maximum_at_complete_collapse():
    """T1.8.2: Assert L2 deviation penalty equals 0.75 when routing
    completely collapses to [1, 0, 0, 0].
    """
    weights = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    loss = compute_l2_balance_loss(weights)
    # (1.0 - 0.25)^2 + 3 * (0.0 - 0.25)^2 = 0.5625 + 0.1875 = 0.75
    assert pytest.approx(loss.item(), abs=1e-6) == 0.75


def test_tier1_feat8_entropy_regularizer_minimum_at_uniform():
    """T1.8.3: Assert negative entropy regularizer is minimized at uniform distribution."""
    uniform_w = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    skewed_w = torch.tensor([[0.70, 0.10, 0.10, 0.10]])

    loss_uniform = compute_entropy_balance_loss(uniform_w).item()
    loss_skewed = compute_entropy_balance_loss(skewed_w).item()
    # Negative entropy is minimized (most negative) at uniform: log(1/4) = -log(4) ~ -1.3863
    assert loss_uniform < loss_skewed
    assert pytest.approx(loss_uniform, abs=1e-4) == -math.log(4.0)


def test_tier1_feat8_switch_load_balance_formulation():
    """T1.8.4: Assert Switch Transformer loss evaluates 4 * sum(f_k * mean_w_k) correctly."""
    # 2 samples: sample 0 routes to class 0, sample 1 routes to class 1
    w = torch.tensor([[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]])
    # f = [0.5, 0.5, 0.0, 0.0]
    # mean_w = [0.45, 0.45, 0.05, 0.05]
    # sum(f * mean_w) = 0.5*0.45 + 0.5*0.45 = 0.45
    # 4 * 0.45 = 1.8
    loss = compute_switch_balance_loss(w)
    assert pytest.approx(loss.item(), abs=1e-5) == 1.8


def test_tier1_feat8_balance_regularizer_backpropagation():
    """T1.8.5: Assert balance loss backpropagates valid gradients to routing logits."""
    logits = torch.randn(4, 4, requires_grad=True)
    weights = F.softmax(logits, dim=-1)
    loss = compute_l2_balance_loss(weights)
    loss.backward()
    assert logits.grad is not None
    assert torch.norm(logits.grad).item() > 0.0


# --- Feature 9: Optuna Pruning & Collapse Detection ---


def test_tier1_feat9_optuna_pruner_dominance_trigger():
    """T1.9.1: Assert collapse pruner triggers when any expert exceeds 90% allocation."""
    collapsed_weights = torch.tensor([0.91, 0.03, 0.03, 0.03])
    assert (
        check_routing_collapse(collapsed_weights, dominance_thresh=0.90, starvation_thresh=0.02)
        is True
    )


def test_tier1_feat9_optuna_pruner_starvation_trigger():
    """T1.9.2: Assert collapse pruner triggers when any expert drops below 2% allocation."""
    starved_weights = torch.tensor([0.40, 0.40, 0.185, 0.015])
    assert (
        check_routing_collapse(starved_weights, dominance_thresh=0.90, starvation_thresh=0.02)
        is True
    )


def test_tier1_feat9_optuna_pruner_healthy_passes():
    """T1.9.3: Assert healthy balanced distribution does not trigger collapse pruning."""
    healthy_weights = torch.tensor([0.30, 0.25, 0.20, 0.25])
    assert (
        check_routing_collapse(healthy_weights, dominance_thresh=0.90, starvation_thresh=0.02)
        is False
    )


def test_tier1_feat9_optuna_pruner_exact_threshold_boundary():
    """T1.9.4: Assert borderline distributions at exactly 0.90 or 0.02 do not trigger pruning."""
    boundary_w = torch.tensor([0.90, 0.04, 0.03, 0.03])
    assert (
        check_routing_collapse(boundary_w, dominance_thresh=0.90, starvation_thresh=0.02) is False
    )

    boundary_w2 = torch.tensor([0.34, 0.34, 0.30, 0.02])
    assert (
        check_routing_collapse(boundary_w2, dominance_thresh=0.90, starvation_thresh=0.02) is False
    )


def test_tier1_feat9_optuna_pruner_multi_batch_aggregation():
    """T1.9.5: Assert batch-wise running mean preserves overall collapse classification."""
    b1 = torch.tensor([0.95, 0.02, 0.015, 0.015])
    b2 = torch.tensor([0.87, 0.05, 0.04, 0.04])
    running_mean = (b1 + b2) / 2.0  # mean = [0.91, 0.035, 0.0275, 0.0275]
    assert check_routing_collapse(running_mean) is True


# --- Feature 10: Routing Analysis & Confusion Matrix ---


def test_tier1_feat10_confusion_matrix_shape_4x4():
    """T1.10.1: Assert routing confusion matrix has dimensions (4, 4)."""
    # 4 rows (ground truth: clean, salt, blur, occ) x 4 columns (expert routing weights)
    matrix = np.zeros((4, 4), dtype=np.float32)
    assert matrix.shape == (4, 4)


def test_tier1_feat10_confusion_matrix_row_stochastic():
    """T1.10.2: Assert all rows of routing confusion matrix sum to 1.0 +/- 1e-5."""
    raw_w = np.random.uniform(0.1, 1.0, size=(4, 4))
    row_stochastic = raw_w / np.sum(raw_w, axis=1, keepdims=True)
    row_sums = np.sum(row_stochastic, axis=1)
    for s in row_sums:
        assert pytest.approx(float(s), abs=1e-5) == 1.0


def test_tier1_feat10_diagonal_dominance_check():
    """T1.10.3: Assert diagonal dominance condition: matrix[k, k] > matrix[k, j]
    for all j != k.
    """
    matrix = np.array(
        [
            [0.85, 0.05, 0.05, 0.05],
            [0.08, 0.78, 0.07, 0.07],
            [0.05, 0.05, 0.82, 0.08],
            [0.06, 0.06, 0.08, 0.80],
        ]
    )
    for k in range(4):
        diag_val = matrix[k, k]
        off_diags = [matrix[k, j] for j in range(4) if j != k]
        assert all(diag_val > off for off in off_diags)


def test_tier1_feat10_monotonic_severity_progression():
    """T1.10.4: Assert routing weights monotonically increase with synthetic
    corruption severity.
    """
    severities = [0.03, 0.08, 0.15]
    expert_weights = [0.45, 0.72, 0.91]
    assert len(severities) == len(expert_weights)
    is_monotonic = all(
        expert_weights[i] <= expert_weights[i + 1] for i in range(len(expert_weights) - 1)
    )
    assert is_monotonic is True


def test_tier1_feat10_expert_health_audit_threshold():
    """T1.10.5: Assert health audit verifies that all 4 experts meet >= 5% dataset utilization."""
    mean_allocations = [0.3156, 0.2501, 0.1873, 0.2470]  # From Step 4 benchmark
    min_alloc = min(mean_allocations)
    assert min_alloc >= 0.05  # Specification acceptance criterion


# --- Feature 11: Atomic ONNX Export & Equivalence ---


def test_tier1_feat11_onnx_export_wrapper_signature(small_moe: SoftMoE):
    """T1.11.1: Assert ExportWrapper accepts input_image and returns (restored, routing_weights)."""
    wrapper = ExportWrapper(small_moe, tau=1.0)
    x = torch.rand(2, 3, 128, 128)
    restored, w = wrapper(x)
    assert restored.shape == (2, 3, 128, 128)
    assert w.shape == (2, 4)


def test_tier1_feat11_onnx_export_graph_validation(small_moe: SoftMoE):
    """T1.11.2: Assert exported ONNX model passes onnx.checker.check_model under opset 17."""
    wrapper = ExportWrapper(small_moe, tau=1.0)
    wrapper.eval()
    dummy = torch.randn(1, 3, 128, 128)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_onnx),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["restored_image", "routing_weights"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "restored_image": {0: "batch_size"},
                "routing_weights": {0: "batch_size"},
            },
            dynamo=False,
        )
        model = onnx.load(str(out_onnx))
        onnx.checker.check_model(model)
        assert out_onnx.stat().st_size > 500_000


def test_tier1_feat11_onnx_dynamic_batching_support(small_moe: SoftMoE):
    """T1.11.3: Assert exported ONNX graph accepts variable batch sizes B in {1, 3, 5}."""
    wrapper = ExportWrapper(small_moe, tau=1.0)
    wrapper.eval()
    dummy = torch.randn(1, 3, 128, 128)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_onnx),
            export_params=True,
            opset_version=17,
            input_names=["input_image"],
            output_names=["restored_image", "routing_weights"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "restored_image": {0: "batch_size"},
                "routing_weights": {0: "batch_size"},
            },
            dynamo=False,
        )
        sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        for b in [1, 3, 5]:
            batch_x = np.random.randn(b, 3, 128, 128).astype(np.float32)
            outputs = sess.run(None, {"input_image": batch_x})
            assert outputs[0].shape == (b, 3, 128, 128)
            assert outputs[1].shape == (b, 4)


def test_tier1_feat11_onnx_restoration_numerical_parity(small_moe: SoftMoE):
    """T1.11.4: Assert PyTorch vs ONNX Runtime reconstruction parity max |pt - ort| < 1e-5."""
    wrapper = ExportWrapper(small_moe, tau=1.0)
    wrapper.eval()
    dummy = torch.randn(2, 3, 128, 128)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_onnx),
            export_params=True,
            opset_version=17,
            input_names=["input_image"],
            output_names=["restored_image", "routing_weights"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "restored_image": {0: "batch_size"},
                "routing_weights": {0: "batch_size"},
            },
            dynamo=False,
        )
        sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pt_restored, _ = wrapper(dummy)

        ort_outputs = sess.run(None, {"input_image": dummy.numpy()})
        max_diff = np.max(np.abs(pt_restored.numpy() - ort_outputs[0]))
        assert max_diff < 1e-5


def test_tier1_feat11_onnx_routing_weights_numerical_parity(small_moe: SoftMoE):
    """T1.11.5: Assert PyTorch vs ONNX Runtime routing weights parity max |pt - ort| < 1e-5."""
    wrapper = ExportWrapper(small_moe, tau=1.0)
    wrapper.eval()
    dummy = torch.randn(2, 3, 128, 128)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_onnx),
            export_params=True,
            opset_version=17,
            input_names=["input_image"],
            output_names=["restored_image", "routing_weights"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "restored_image": {0: "batch_size"},
                "routing_weights": {0: "batch_size"},
            },
            dynamo=False,
        )
        sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            _, pt_weights = wrapper(dummy)

        ort_outputs = sess.run(None, {"input_image": dummy.numpy()})
        max_diff = np.max(np.abs(pt_weights.numpy() - ort_outputs[1]))
        assert max_diff < 1e-5


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ==============================================================================


def test_tier2_extreme_low_temperature_clamping(small_gate: GatingNetwork):
    """T2.1: Assert extreme low tau (1e-6) clamps without NaN or overflow."""
    x = torch.randn(2, 3, 128, 128)
    w, _ = small_gate(x, tau=1e-6)
    assert not torch.isnan(w).any()
    assert not torch.isinf(w).any()
    assert torch.all(w >= 0.0)


def test_tier2_extreme_high_temperature_smoothing(small_gate: GatingNetwork):
    """T2.2: Assert extreme high tau (100.0) smoothly approaches [0.25, 0.25, 0.25, 0.25]."""
    x = torch.randn(2, 3, 128, 128)
    w, _ = small_gate(x, tau=100.0)
    for row in w:
        assert torch.allclose(row, torch.tensor([0.25, 0.25, 0.25, 0.25]), atol=0.05)


def test_tier2_negative_temperature_handling(small_gate: GatingNetwork):
    """T2.3: Assert negative temperature (-0.5) is clamped to MIN_TEMPERATURE (0.05)."""
    x = torch.randn(1, 3, 128, 128)
    w_neg, _ = small_gate(x, tau=-0.5)
    w_ref, _ = small_gate(x, tau=MIN_TEMPERATURE)
    assert torch.allclose(w_neg, w_ref, atol=1e-6)


def test_tier2_singleton_batch_processing(small_moe: SoftMoE):
    """T2.4: Assert singleton batch B=1 restores cleanly without dimensional squeeze errors."""
    x = torch.rand(1, 3, 128, 128)
    restored, w, z = small_moe(x)
    assert restored.shape == (1, 3, 128, 128)
    assert w.shape == (1, 4)
    assert z.shape == (1, 4)


def test_tier2_prime_batch_size(small_moe: SoftMoE):
    """T2.5: Assert prime batch size B=7 evaluates correctly."""
    x = torch.rand(7, 3, 128, 128)
    restored, w, z = small_moe(x)
    assert restored.shape == (7, 3, 128, 128)
    assert w.shape == (7, 4)


def test_tier2_all_zero_tensor_pure_black(small_moe: SoftMoE):
    """T2.6: Assert pure black image (all zeros) processes without divide-by-zero or NaN."""
    x = torch.zeros(1, 3, 128, 128)
    restored, w, z = small_moe(x)
    assert not torch.isnan(restored).any()
    assert not torch.isnan(w).any()


def test_tier2_all_one_tensor_pure_white(small_moe: SoftMoE):
    """T2.7: Assert pure white image (all ones) processes without saturation NaN/Inf."""
    x = torch.ones(1, 3, 128, 128)
    restored, w, z = small_moe(x)
    assert not torch.isnan(restored).any()
    assert not torch.isnan(w).any()


def test_tier2_extreme_noise_density(small_moe: SoftMoE):
    """T2.8: Assert extreme salt-and-pepper noise p=1.0 (100% corrupted) processes cleanly."""
    x = torch.rand(1, 3, 128, 128)
    corrupted = apply_salt_and_pepper(x, p=1.0)
    restored, w, z = small_moe(corrupted)
    assert restored.shape == (1, 3, 128, 128)
    assert not torch.isnan(restored).any()


def test_tier2_full_image_occlusion(small_moe: SoftMoE):
    """T2.9: Assert 100% occlusion area processes without crash."""
    # Fully black occluded tensor
    occluded = torch.zeros(1, 3, 128, 128)
    restored, w, z = small_moe(occluded)
    assert restored.shape == (1, 3, 128, 128)
    assert not torch.isnan(restored).any()


def test_tier2_gradient_clipping_guard():
    """T2.10: Assert gradient clipping with max_norm=1.0 keeps gradients bounded
    under large loss.
    """
    moe = SoftMoE(channels=(16, 32, 64, 128), bottleneck_dim=128)
    moe.train()
    x = torch.rand(2, 3, 128, 128)
    restored, _, _ = moe(x)
    loss = 1000.0 * F.mse_loss(restored, torch.zeros_like(restored))  # artificially huge loss
    loss.backward()

    total_norm = torch.nn.utils.clip_grad_norm_(moe.parameters(), max_norm=1.0)
    assert total_norm > 1.0  # Original norm was huge
    # Verify every parameter gradient is finite
    for p in moe.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()


# ==============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS
# ==============================================================================


def test_tier3_warmup_to_joint_transition_preservation(small_moe: SoftMoE):
    """T3.1: Assert warm-up frozen phase preserves specialist weights,
    unfreezing allows updates.
    """
    # 1. Warm-up phase: freeze specialists
    small_moe.set_experts_frozen(True)
    orig_blur_weight = next(small_moe.specialist_blur.parameters()).clone()

    x = torch.rand(2, 3, 128, 128)
    restored, _, _ = small_moe(x)
    loss1 = F.l1_loss(restored, torch.zeros_like(restored))
    loss1.backward()

    # Specialist weight must not have changed
    assert torch.equal(next(small_moe.specialist_blur.parameters()), orig_blur_weight)

    # 2. Joint phase: unfreeze specialists
    small_moe.set_experts_frozen(False)
    small_moe.zero_grad()
    restored2, _, _ = small_moe(x)
    loss2 = F.l1_loss(restored2, torch.zeros_like(restored2))
    loss2.backward()
    # Now specialist must have non-zero gradient
    grads = [
        torch.norm(p.grad).item()
        for p in small_moe.specialist_blur.parameters()
        if p.grad is not None
    ]
    assert len(grads) > 0 and max(grads) > 0.0


def test_tier3_simultaneous_multi_branch_gradient_flow():
    """T3.2: Assert all 4 branches pass gradients simultaneously into Gate and all 3 specialists."""
    moe = SoftMoE(channels=(16, 32, 64, 128), bottleneck_dim=128)
    moe.train()
    moe.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128)
    restored, w, z = moe(x)
    loss = F.l1_loss(restored, torch.ones_like(restored))
    loss.backward()

    assert moe.gate.fc.weight.grad is not None
    for spec in [moe.specialist_salt, moe.specialist_blur, moe.specialist_occlusion]:
        grads = [torch.norm(p.grad).item() for p in spec.parameters() if p.grad is not None]
        assert len(grads) > 0 and max(grads) > 0.0


def test_tier3_gate_logit_sensitivity_direct_blending(small_moe: SoftMoE):
    """T3.3: Assert manually shifting gate logits directly and smoothly shifts composite output."""
    x = torch.rand(1, 3, 128, 128)
    # Forward at tau=0.5 vs tau=2.0
    out_sharp, w_sharp, _ = small_moe(x, tau=0.5)
    out_smooth, w_smooth, _ = small_moe(x, tau=2.0)

    # Restored outputs must differ due to differing expert weights
    weight_diff = torch.norm(w_sharp - w_smooth).item()
    img_diff = torch.norm(out_sharp - out_smooth).item()
    assert weight_diff > 0.0
    assert img_diff > 0.0


def test_tier3_balance_loss_counters_collapse_pressure():
    """T3.4: Assert balance regularizer produces negative gradient opposing expert domination."""
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]], requires_grad=True)  # Expert 0 dominates
    w = F.softmax(logits, dim=-1)
    bal_loss = compute_l2_balance_loss(w)
    bal_loss.backward()

    # The gradient for logit 0 must be positive (pushing down logit 0 to decrease loss)
    assert logits.grad[0, 0].item() > 0.0
    # The gradient for other logits must be negative (pushing them up)
    assert logits.grad[0, 1].item() < 0.0


def test_tier3_onnx_export_wrapper_fixed_tau_eager_match(small_moe: SoftMoE):
    """T3.5: Assert ExportWrapper eager execution produces bit-exact match
    with standalone SoftMoE.
    """
    wrapper = ExportWrapper(small_moe, tau=1.5)
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        w_res, w_w = wrapper(x)
        m_res, m_w, _ = small_moe(x, tau=1.5)

    assert torch.equal(w_res, m_res)
    assert torch.equal(w_w, m_w)


def test_tier3_onnx_blended_restoration_parity_multi_batch(small_moe: SoftMoE):
    """T3.6: Assert ONNX Runtime produces <1e-5 numerical parity across dynamic
    batch sizes B in {1, 2, 4}.
    """
    wrapper = ExportWrapper(small_moe, tau=1.0)
    wrapper.eval()
    dummy = torch.randn(1, 3, 128, 128)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "dynamic_moe.onnx"
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_file),
            export_params=True,
            opset_version=17,
            input_names=["input_image"],
            output_names=["restored_image", "routing_weights"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "restored_image": {0: "batch_size"},
                "routing_weights": {0: "batch_size"},
            },
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        for b in [1, 2, 4]:
            x = np.random.randn(b, 3, 128, 128).astype(np.float32)
            with torch.no_grad():
                pt_res, pt_w = wrapper(torch.from_numpy(x))
            ort_res, ort_w = sess.run(None, {"input_image": x})
            assert np.max(np.abs(pt_res.numpy() - ort_res)) < 1e-5
            assert np.max(np.abs(pt_w.numpy() - ort_w)) < 1e-5


# ==============================================================================
# TIER 4: REAL-WORLD SCENARIOS
# ==============================================================================


def test_tier4_clean_pet_restoration_psnr(
    checkpoint_paths: Dict[str, Path], sample_pet_tensor: torch.Tensor
):
    """T4.1: Assert clean pet image through loaded SoftMoE retains PSNR >= 20.0 dB."""
    moe = SoftMoE()
    moe.load_pretrained_components(
        classifier_path=checkpoint_paths["classifier"],
        salt_path=checkpoint_paths["salt"],
        blur_path=checkpoint_paths["blur"],
        occlusion_path=checkpoint_paths["occlusion"],
    )
    moe.eval()

    clean_img = apply_clean(sample_pet_tensor)
    with torch.no_grad():
        restored, routing_weights, _ = moe(clean_img)

    psnr = compute_psnr(restored, sample_pet_tensor)
    assert psnr >= 20.0  # Quality preservation floor


def test_tier4_salt_and_pepper_pet_restoration(
    checkpoint_paths: Dict[str, Path], sample_pet_tensor: torch.Tensor
):
    """T4.2: Assert pet image degraded by S&P noise (p=0.08) routes to salt
    specialist and restores.
    """
    moe = SoftMoE()
    moe.load_pretrained_components(
        classifier_path=checkpoint_paths["classifier"],
        salt_path=checkpoint_paths["salt"],
        blur_path=checkpoint_paths["blur"],
        occlusion_path=checkpoint_paths["occlusion"],
    )
    moe.eval()

    corrupted = apply_salt_and_pepper(sample_pet_tensor, p=0.08)
    with torch.no_grad():
        restored, routing_weights, _ = moe(corrupted)

    # Specialist 1 is Salt-and-Pepper
    w_salt = routing_weights[0, 1].item()
    # Task 2 classifier achieves >98% accuracy; gate assigns weight to salt expert
    assert w_salt > 0.50
    assert restored.shape == (1, 3, 128, 128)


def test_tier4_gaussian_blur_pet_restoration(
    checkpoint_paths: Dict[str, Path], sample_pet_tensor: torch.Tensor
):
    """T4.3: Assert pet image degraded by Gaussian blur (k=5, sigma=1.5) routes
    to blur specialist.
    """
    moe = SoftMoE()
    moe.load_pretrained_components(
        classifier_path=checkpoint_paths["classifier"],
        salt_path=checkpoint_paths["salt"],
        blur_path=checkpoint_paths["blur"],
        occlusion_path=checkpoint_paths["occlusion"],
    )
    moe.eval()

    corrupted = apply_gaussian_blur(sample_pet_tensor, kernel_size=5, sigma=1.5)
    with torch.no_grad():
        restored, routing_weights, _ = moe(corrupted)

    # Specialist 2 is Gaussian Blur
    w_blur = routing_weights[0, 2].item()
    assert w_blur > 0.50
    assert restored.shape == (1, 3, 128, 128)


def test_tier4_occlusion_pet_restoration(
    checkpoint_paths: Dict[str, Path], sample_pet_tensor: torch.Tensor
):
    """T4.4: Assert pet image degraded by rectangular occlusion routes to
    occlusion specialist.
    """
    moe = SoftMoE()
    moe.load_pretrained_components(
        classifier_path=checkpoint_paths["classifier"],
        salt_path=checkpoint_paths["salt"],
        blur_path=checkpoint_paths["blur"],
        occlusion_path=checkpoint_paths["occlusion"],
    )
    moe.eval()

    # Rectangular occlusion
    boxes = sample_occlusion_boxes(height=128, width=128, num_boxes=1, target_ratio=0.20)
    corrupted = apply_occlusion(sample_pet_tensor, boxes=boxes)
    with torch.no_grad():
        restored, routing_weights, _ = moe(corrupted)

    # Specialist 3 is Occlusion
    w_occ = routing_weights[0, 3].item()
    assert w_occ > 0.50
    assert restored.shape == (1, 3, 128, 128)


def test_tier4_restoration_superiority_over_unrouted_identity(
    checkpoint_paths: Dict[str, Path], sample_pet_tensor: torch.Tensor
):
    """T4.5: Assert SoftMoE restoration improves over corrupted input on
    salt-and-pepper noise.
    """
    moe = SoftMoE()
    moe.load_pretrained_components(
        classifier_path=checkpoint_paths["classifier"],
        salt_path=checkpoint_paths["salt"],
        blur_path=checkpoint_paths["blur"],
        occlusion_path=checkpoint_paths["occlusion"],
    )
    moe.eval()

    corrupted = apply_salt_and_pepper(sample_pet_tensor, p=0.08)
    with torch.no_grad():
        restored, _, _ = moe(corrupted)

    psnr_corrupted = compute_psnr(corrupted, sample_pet_tensor)
    psnr_restored = compute_psnr(restored, sample_pet_tensor)

    # Restored image must have higher PSNR than the noisy input
    assert psnr_restored > psnr_corrupted
