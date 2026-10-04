"""tests/test_task3_adversarial_stress.py
---------------------------------------
Adversarial Stress Testing Suite for Task 3 Soft Mixture-of-Experts (MoE).

Milestone 1 Verification:
1. Extreme temperature values: tau in {1e-6, 0.001, 0.05, 1.0, 10.0, 1000.0}
2. Extreme input values:
   - all-zeros (pure black)
   - all-ones (pure white)
   - uniform noise [-10.0, 10.0]
   - NaN / Inf tensor handling
3. Differentiability under multiple loss functions:
   - L1 reconstruction loss
   - MSE reconstruction loss
   - SSIM dissimilarity loss
   - Multi-objective combined loss (L1 + SSIM + Auxiliary CE)
   Asserts non-zero, finite gradients on both gate and specialist parameters.
4. Two-stage freeze/unfreeze gradient flow isolation.
5. CPU vs CUDA numerical parity and device stability.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.losses import CombinedReconstructionLoss, SSIMLoss
from src.task3.gate import GatingNetwork, MIN_TEMPERATURE
from src.task3.moe_model import SoftMoE

EXTREME_TEMPERATURES: List[float] = [1e-6, 0.001, 0.05, 1.0, 10.0, 1000.0]
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


# ==============================================================================
# Helper Factories
# ==============================================================================

def create_deterministic_moe(device: str = "cpu") -> SoftMoE:
    """Create a lightweight deterministic SoftMoE for rapid, reproducible testing."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = SoftMoE(channels=(16, 32, 64, 128), bottleneck_dim=128, dropout=0.0)
    model.to(device)
    return model


# ==============================================================================
# 1. Extreme Temperature Stress Tests
# ==============================================================================

@pytest.mark.parametrize("tau", EXTREME_TEMPERATURES)
@pytest.mark.parametrize("device_name", DEVICES)
def test_adversarial_extreme_temperatures_forward(tau: float, device_name: str) -> None:
    """Verify GatingNetwork and SoftMoE forward pass remains strictly finite and valid under extreme tau."""
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    model.eval()

    x = torch.rand(4, 3, 128, 128, device=device)

    # 1. GatingNetwork check
    w, z = model.gate(x, tau=tau)
    assert not torch.isnan(w).any(), f"NaN detected in routing weights at tau={tau}"
    assert not torch.isinf(w).any(), f"Inf detected in routing weights at tau={tau}"
    assert not torch.isnan(z).any(), f"NaN detected in logits at tau={tau}"
    assert not torch.isinf(z).any(), f"Inf detected in logits at tau={tau}"
    assert (w >= 0.0).all(), f"Negative probability at tau={tau}"
    assert (w <= 1.0).all(), f"Probability > 1.0 at tau={tau}"

    # Simplex sum constraint
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Simplex sum violated at tau={tau}: {row_sums}"
    )

    # 2. SoftMoE check
    restored, routing_w, logits = model(x, tau=tau)
    assert not torch.isnan(restored).any(), f"NaN detected in restored image at tau={tau}"
    assert not torch.isinf(restored).any(), f"Inf detected in restored image at tau={tau}"
    assert restored.shape == (4, 3, 128, 128)
    assert torch.allclose(routing_w, w, atol=1e-6)


@pytest.mark.parametrize("tau", EXTREME_TEMPERATURES)
def test_adversarial_extreme_temperatures_backward(tau: float) -> None:
    """Verify backward pass computes finite gradients for all extreme temperatures."""
    model = create_deterministic_moe()
    model.train()
    model.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128, requires_grad=True)
    target = torch.rand(2, 3, 128, 128)

    restored, routing_weights, logits = model(x, tau=tau)
    loss = F.l1_loss(restored, target)
    loss.backward()

    # Gate gradients must be non-zero and finite
    assert model.gate.fc.weight.grad is not None
    assert not torch.isnan(model.gate.fc.weight.grad).any()
    assert not torch.isinf(model.gate.fc.weight.grad).any()
    assert model.gate.fc.weight.grad.abs().sum().item() > 0.0, (
        f"Gate gradient vanished at tau={tau}"
    )

    # Specialist gradients must be non-zero and finite
    for name, specialist in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        head_grad = specialist.decoder.output_head[0].weight.grad
        assert head_grad is not None, f"Specialist {name} head grad is None at tau={tau}"
        assert not torch.isnan(head_grad).any(), f"Specialist {name} head grad has NaN at tau={tau}"
        assert not torch.isinf(head_grad).any(), f"Specialist {name} head grad has Inf at tau={tau}"
        assert head_grad.abs().sum().item() > 0.0, f"Specialist {name} head grad vanished at tau={tau}"


def test_adversarial_temperature_clamping_exactness() -> None:
    """Verify that tau < MIN_TEMPERATURE (0.05) is clamped and produces identical outputs."""
    model = create_deterministic_moe()
    model.eval()
    x = torch.rand(3, 3, 128, 128)

    w_ref, _ = model.gate(x, tau=MIN_TEMPERATURE)

    for sub_tau in [1e-6, 1e-4, 0.001, 0.01, 0.0499]:
        w_sub, _ = model.gate(x, tau=sub_tau)
        assert torch.allclose(w_sub, w_ref, atol=1e-7), (
            f"Clamping inconsistency between tau={sub_tau} and MIN_TEMPERATURE={MIN_TEMPERATURE}"
        )


def test_adversarial_temperature_high_uniformity() -> None:
    """Verify that tau = 1000.0 converges to uniform distribution [0.25, 0.25, 0.25, 0.25]."""
    model = create_deterministic_moe()
    model.eval()
    x = torch.randn(4, 3, 128, 128)

    w_high, _ = model.gate(x, tau=1000.0)
    expected_uniform = torch.full_like(w_high, 0.25)
    max_deviation = (w_high - expected_uniform).abs().max().item()
    assert max_deviation < 0.01, f"High temperature failed to converge to uniform: max diff={max_deviation}"


def test_adversarial_temperature_entropy_monotonicity() -> None:
    """Verify that routing entropy is non-decreasing with increasing tau above clamping threshold."""
    model = create_deterministic_moe()
    model.eval()
    x = torch.randn(8, 3, 128, 128)

    taus = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]
    entropies: List[float] = []

    for t in taus:
        w, _ = model.gate(x, tau=t)
        # Shannon entropy H(w) = - sum(w * log(w + 1e-12))
        ent = -(w * torch.log(w + 1e-12)).sum(dim=-1).mean().item()
        entropies.append(ent)

    # Entropies should be non-decreasing
    for i in range(len(entropies) - 1):
        assert entropies[i] <= entropies[i + 1] + 1e-5, (
            f"Entropy decreased from tau={taus[i]} ({entropies[i]:.4f}) to tau={taus[i+1]} ({entropies[i+1]:.4f})"
        )


# ==============================================================================
# 2. Extreme Input Values Stress Tests
# ==============================================================================

@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_adversarial_extreme_inputs_all_zeros(device_name: str, mode: str) -> None:
    """Verify model behavior on all-zeros (pure black) inputs."""
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    if mode == "train":
        model.train()
    else:
        model.eval()

    x_zero = torch.zeros(2, 3, 128, 128, device=device)

    # Forward
    restored, w, z = model(x_zero)
    assert not torch.isnan(restored).any(), f"NaN in restored image for all-zeros ({mode}, {device_name})"
    assert not torch.isinf(restored).any(), f"Inf in restored image for all-zeros ({mode}, {device_name})"
    assert not torch.isnan(w).any(), f"NaN in routing weights for all-zeros ({mode}, {device_name})"
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, device=device), atol=1e-5)

    # Backward
    if mode == "train":
        model.set_experts_frozen(False)
        target = torch.rand(2, 3, 128, 128, device=device)
        loss = F.l1_loss(restored, target)
        loss.backward()
        assert model.gate.fc.weight.grad is not None
        assert not torch.isnan(model.gate.fc.weight.grad).any()
        assert not torch.isinf(model.gate.fc.weight.grad).any()


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_adversarial_extreme_inputs_all_ones(device_name: str, mode: str) -> None:
    """Verify model behavior on all-ones (pure white) inputs."""
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    if mode == "train":
        model.train()
    else:
        model.eval()

    x_ones = torch.ones(2, 3, 128, 128, device=device)

    # Forward
    restored, w, z = model(x_ones)
    assert not torch.isnan(restored).any(), f"NaN in restored image for all-ones ({mode}, {device_name})"
    assert not torch.isinf(restored).any(), f"Inf in restored image for all-ones ({mode}, {device_name})"
    assert not torch.isnan(w).any(), f"NaN in routing weights for all-ones ({mode}, {device_name})"
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, device=device), atol=1e-5)

    # Backward
    if mode == "train":
        model.set_experts_frozen(False)
        target = torch.rand(2, 3, 128, 128, device=device)
        loss = F.l1_loss(restored, target)
        loss.backward()
        assert model.gate.fc.weight.grad is not None
        assert not torch.isnan(model.gate.fc.weight.grad).any()
        assert not torch.isinf(model.gate.fc.weight.grad).any()


@pytest.mark.parametrize("device_name", DEVICES)
def test_adversarial_extreme_inputs_out_of_bounds_noise(device_name: str) -> None:
    """Verify model behavior on extreme out-of-bounds uniform noise [-10.0, 10.0]."""
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    model.train()
    model.set_experts_frozen(False)

    x_noise = torch.empty(2, 3, 128, 128, device=device).uniform_(-10.0, 10.0)

    # Forward
    restored, w, z = model(x_noise)
    assert not torch.isnan(restored).any(), "NaN detected under [-10, 10] noise"
    assert not torch.isinf(restored).any(), "Inf detected under [-10, 10] noise"
    assert not torch.isnan(w).any(), "NaN in routing weights under [-10, 10] noise"
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, device=device), atol=1e-5)

    # Backward
    loss = F.mse_loss(restored, torch.zeros_like(restored))
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name} under extreme noise"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name} under extreme noise"


def test_adversarial_nan_and_inf_tensor_propagation() -> None:
    """Verify model behavior when fed corrupted tensors containing NaN or Inf."""
    model = create_deterministic_moe()
    model.eval()

    # 1. NaN propagation: PyTorch should propagate NaN through forward pass without unhandled crashes
    x_nan = torch.rand(2, 3, 128, 128)
    x_nan[0, 1, 30, 40] = float("nan")

    restored_nan, w_nan, z_nan = model(x_nan)
    # The corrupted sample (index 0) must reflect NaN
    assert torch.isnan(restored_nan[0]).any(), "NaN should propagate through restoration for sample 0"
    assert torch.isnan(w_nan[0]).any(), "NaN should propagate through gate routing weights for sample 0"
    # The clean sample (index 1) should remain valid
    assert not torch.isnan(restored_nan[1]).any(), "Sample 1 was contaminated by sample 0 NaNs!"
    assert not torch.isnan(w_nan[1]).any(), "Sample 1 weights were contaminated by sample 0 NaNs!"

    # 2. Inf propagation
    x_inf = torch.rand(2, 3, 128, 128)
    x_inf[0, 0, 10, 10] = float("inf")

    restored_inf, w_inf, z_inf = model(x_inf)
    assert torch.isnan(restored_inf[0]).any() or torch.isinf(restored_inf[0]).any(), (
        "Inf should produce Inf or NaN in restoration output"
    )
    # Clean sample 1 remains valid
    assert not torch.isnan(restored_inf[1]).any()
    assert not torch.isinf(restored_inf[1]).any()


# ==============================================================================
# 3. Differentiability with Multiple Loss Functions
# ==============================================================================

@pytest.mark.parametrize("device_name", DEVICES)
def test_adversarial_differentiability_l1_loss(device_name: str) -> None:
    """Assert non-zero, finite gradients on gate and all specialists under L1 reconstruction loss."""
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    model.train()
    model.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128, device=device)
    target = torch.rand(2, 3, 128, 128, device=device)

    restored, w, z = model(x)
    loss = F.l1_loss(restored, target)
    loss.backward()

    # Check Gate
    gate_grad = model.gate.fc.weight.grad
    assert gate_grad is not None
    assert not torch.isnan(gate_grad).any()
    assert not torch.isinf(gate_grad).any()
    assert gate_grad.abs().sum().item() > 0.0

    # Check Specialists
    for name, specialist in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        p_grad = specialist.decoder.output_head[0].weight.grad
        assert p_grad is not None, f"Specialist {name} grad is None"
        assert not torch.isnan(p_grad).any(), f"Specialist {name} grad is NaN"
        assert not torch.isinf(p_grad).any(), f"Specialist {name} grad is Inf"
        assert p_grad.abs().sum().item() > 0.0, f"Specialist {name} grad vanished"


@pytest.mark.parametrize("device_name", DEVICES)
def test_adversarial_differentiability_mse_loss(device_name: str) -> None:
    """Assert non-zero, finite gradients on gate and all specialists under MSE reconstruction loss."""
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    model.train()
    model.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128, device=device)
    target = torch.rand(2, 3, 128, 128, device=device)

    restored, w, z = model(x)
    loss = F.mse_loss(restored, target)
    loss.backward()

    # Gate
    assert model.gate.fc.weight.grad is not None
    assert not torch.isnan(model.gate.fc.weight.grad).any()
    assert model.gate.fc.weight.grad.abs().sum().item() > 0.0

    # Specialists
    for name, specialist in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        p_grad = specialist.decoder.output_head[0].weight.grad
        assert p_grad is not None
        assert not torch.isnan(p_grad).any()
        assert p_grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("device_name", DEVICES)
def test_adversarial_differentiability_ssim_loss(device_name: str) -> None:
    """Assert non-zero, finite gradients on gate and all specialists under SSIM dissimilarity loss."""
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    model.train()
    model.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128, device=device)
    target = torch.rand(2, 3, 128, 128, device=device)

    restored, w, z = model(x)
    ssim_loss_fn = SSIMLoss()
    loss = ssim_loss_fn(restored, target)
    loss.backward()

    # Gate
    assert model.gate.fc.weight.grad is not None
    assert not torch.isnan(model.gate.fc.weight.grad).any()
    assert model.gate.fc.weight.grad.abs().sum().item() > 0.0

    # Specialists
    for name, specialist in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        p_grad = specialist.decoder.output_head[0].weight.grad
        assert p_grad is not None
        assert not torch.isnan(p_grad).any()
        assert p_grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("device_name", DEVICES)
def test_adversarial_differentiability_combined_multi_objective_loss(device_name: str) -> None:
    """Assert non-zero, finite gradients under total multi-objective loss:
    L_tot = 0.8 * L1 + 0.2 * (1 - SSIM) + 0.1 * L_CE.
    """
    device = torch.device(device_name)
    model = create_deterministic_moe(device=device_name)
    model.train()
    model.set_experts_frozen(False)

    x = torch.rand(2, 3, 128, 128, device=device)
    target = torch.rand(2, 3, 128, 128, device=device)
    labels = torch.tensor([1, 2], device=device)  # Class labels for auxiliary CE

    restored, w, z = model(x)
    comb_loss_fn = CombinedReconstructionLoss(alpha=0.80)
    rec_loss = comb_loss_fn(restored, target)
    ce_loss = F.cross_entropy(z, labels)
    total_loss = rec_loss + 0.1 * ce_loss

    total_loss.backward()

    # Gate
    assert model.gate.fc.weight.grad is not None
    assert not torch.isnan(model.gate.fc.weight.grad).any()
    assert model.gate.fc.weight.grad.abs().sum().item() > 0.0

    # Specialists
    for name, specialist in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        p_grad = specialist.decoder.output_head[0].weight.grad
        assert p_grad is not None
        assert not torch.isnan(p_grad).any()
        assert p_grad.abs().sum().item() > 0.0


# ==============================================================================
# 4. Freeze/Unfreeze Gradient Flow Isolation Tests
# ==============================================================================

def test_adversarial_freeze_isolation_guarantee() -> None:
    """Verify that when specialists are frozen, their parameter gradients are strictly None."""
    model = create_deterministic_moe()
    model.train()

    # 1. Warm-up phase: freeze specialists
    model.set_experts_frozen(True)
    assert model.experts_frozen is True

    x = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)
    restored, _, z = model(x)
    loss = F.l1_loss(restored, target) + F.cross_entropy(z, torch.tensor([0, 1]))
    loss.backward()

    # Gate MUST receive gradients
    assert model.gate.fc.weight.grad is not None
    assert model.gate.fc.weight.grad.abs().sum().item() > 0.0

    # Specialists MUST NOT receive gradients
    for name, specialist in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        for param_name, param in specialist.named_parameters():
            assert param.grad is None, (
                f"Frozen specialist {name} parameter {param_name} received unexpected gradient!"
            )


def test_adversarial_unfreeze_flow_guarantee() -> None:
    """Verify that unfreezing restores gradient flow across all specialist parameters."""
    model = create_deterministic_moe()
    model.train()

    # Unfreeze
    model.set_experts_frozen(False)
    assert model.experts_frozen is False

    x = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)
    restored, _, _ = model(x)
    loss = F.l1_loss(restored, target)
    loss.backward()

    # Every specialist parameter must have grad
    for name, specialist in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        for param_name, param in specialist.named_parameters():
            assert param.grad is not None, (
                f"Unfrozen specialist {name} parameter {param_name} did not receive gradient!"
            )
            assert not torch.isnan(param.grad).any()


# ==============================================================================
# 5. Device Parity (CPU vs CUDA)
# ==============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adversarial_cpu_vs_cuda_numerical_parity() -> None:
    """Assert CPU and CUDA produce numerically identical outputs within float32 tolerance."""
    model_cpu = create_deterministic_moe(device="cpu")
    model_cuda = create_deterministic_moe(device="cuda")
    model_cuda.load_state_dict(model_cpu.state_dict())

    model_cpu.eval()
    model_cuda.eval()

    torch.manual_seed(99)
    x_cpu = torch.rand(2, 3, 128, 128)
    x_cuda = x_cpu.clone().to("cuda")

    with torch.no_grad():
        restored_cpu, w_cpu, z_cpu = model_cpu(x_cpu, tau=1.0)
        restored_cuda, w_cuda, z_cuda = model_cuda(x_cuda, tau=1.0)

    # Compare restored image
    diff_img = (restored_cpu - restored_cuda.cpu()).abs().max().item()
    diff_w = (w_cpu - w_cuda.cpu()).abs().max().item()
    diff_z = (z_cpu - z_cuda.cpu()).abs().max().item()

    assert diff_img < 1e-4, f"CPU vs CUDA restored image discrepancy too high: {diff_img}"
    assert diff_w < 1e-5, f"CPU vs CUDA routing weights discrepancy too high: {diff_w}"
    assert diff_z < 1e-4, f"CPU vs CUDA logits discrepancy too high: {diff_z}"


# ==============================================================================
# Standalone Execution Runner
# ==============================================================================

def run_all_adversarial_stress_benchmarks() -> Dict[str, Any]:
    """Execute all adversarial benchmarks and collect detailed quantitative summary."""
    print("=" * 80)
    print("STARTING EMPIRICAL ADVERSARIAL STRESS SUITE (MILESTONE 1)")
    print("=" * 80)

    summary: Dict[str, Any] = {
        "temperature_checks": {},
        "extreme_inputs": {},
        "differentiability": {},
        "freeze_isolation": {},
        "cuda_parity": {},
    }

    # 1. Temperature checks
    print("\n--- [1] Extreme Temperature Sweep ---")
    model = create_deterministic_moe()
    model.eval()
    x = torch.rand(4, 3, 128, 128)

    for tau in EXTREME_TEMPERATURES:
        w, z = model.gate(x, tau=tau)
        max_diff_simplex = (w.sum(dim=-1) - 1.0).abs().max().item()
        entropy = -(w * torch.log(w + 1e-12)).sum(dim=-1).mean().item()
        w_sample = [round(float(v), 4) for v in w[0].tolist()]

        summary["temperature_checks"][f"tau_{tau}"] = {
            "max_simplex_error": max_diff_simplex,
            "entropy": entropy,
            "sample_w": w_sample,
            "is_finite": bool(not torch.isnan(w).any() and not torch.isinf(w).any()),
        }
        print(f"tau={tau:8.6f} | Finite: {not torch.isnan(w).any()} | Simplex Err: {max_diff_simplex:.2e} | Entropy: {entropy:.4f} | Sample w: {w_sample}")

    # 2. Extreme inputs
    print("\n--- [2] Extreme Input Handling ---")
    inputs = {
        "all_zeros": torch.zeros(2, 3, 128, 128),
        "all_ones": torch.ones(2, 3, 128, 128),
        "noise_uniform_[-10,10]": torch.empty(2, 3, 128, 128).uniform_(-10.0, 10.0),
    }
    for name, inp in inputs.items():
        restored, w, z = model(inp)
        has_nan = bool(torch.isnan(restored).any() or torch.isnan(w).any())
        has_inf = bool(torch.isinf(restored).any() or torch.isinf(w).any())
        summary["extreme_inputs"][name] = {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "simplex_sum": [round(float(v), 5) for v in w.sum(dim=-1).tolist()],
        }
        print(f"Input: {name:22s} | NaN: {has_nan} | Inf: {has_inf} | Simplex: {w.sum(dim=-1).tolist()}")

    # 3. Differentiability
    print("\n--- [3] Differentiability & Multi-Loss Gradient Flow ---")
    model.train()
    model.set_experts_frozen(False)
    x = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)
    labels = torch.tensor([0, 1])

    losses = {
        "L1": lambda res, logits: F.l1_loss(res, target),
        "MSE": lambda res, logits: F.mse_loss(res, target),
        "SSIM": lambda res, logits: SSIMLoss()(res, target),
        "Combined": lambda res, logits: CombinedReconstructionLoss(alpha=0.8)(res, target) + 0.1 * F.cross_entropy(logits, labels),
    }

    for loss_name, loss_fn in losses.items():
        model.zero_grad()
        res, w, z = model(x)
        loss_val = loss_fn(res, z)
        loss_val.backward()

        gate_norm = torch.norm(model.gate.fc.weight.grad).item()
        salt_norm = torch.norm(model.specialist_salt.decoder.output_head[0].weight.grad).item()
        blur_norm = torch.norm(model.specialist_blur.decoder.output_head[0].weight.grad).item()
        occ_norm = torch.norm(model.specialist_occlusion.decoder.output_head[0].weight.grad).item()

        summary["differentiability"][loss_name] = {
            "loss_value": round(float(loss_val.item()), 5),
            "gate_fc_grad_norm": round(gate_norm, 6),
            "salt_head_grad_norm": round(salt_norm, 6),
            "blur_head_grad_norm": round(blur_norm, 6),
            "occ_head_grad_norm": round(occ_norm, 6),
        }
        print(f"Loss: {loss_name:8s} | Gate Norm: {gate_norm:.6f} | Salt: {salt_norm:.6f} | Blur: {blur_norm:.6f} | Occ: {occ_norm:.6f}")

    # 4. Freeze isolation
    print("\n--- [4] Freezing Isolation ---")
    model.set_experts_frozen(True)
    model.zero_grad()
    res, _, z = model(x)
    (F.l1_loss(res, target) + F.cross_entropy(z, labels)).backward()

    gate_grad = model.gate.fc.weight.grad is not None and torch.norm(model.gate.fc.weight.grad).item() > 0.0
    experts_none = all(
        p.grad is None
        for spec in (model.specialist_salt, model.specialist_blur, model.specialist_occlusion)
        for p in spec.parameters()
    )
    summary["freeze_isolation"] = {
        "gate_trainable": gate_grad,
        "specialists_isolated": experts_none,
    }
    print(f"Gate trainable: {gate_grad} | Specialists isolated (grad is None): {experts_none}")

    # 5. CUDA parity
    if torch.cuda.is_available():
        print("\n--- [5] CUDA Numerical Parity ---")
        model_cuda = create_deterministic_moe(device="cuda")
        model_cuda.load_state_dict(model.state_dict())
        model_cuda.eval()
        model.eval()

        with torch.no_grad():
            r_cpu, w_cpu, _ = model(x)
            r_cuda, w_cuda, _ = model_cuda(x.cuda())
            max_r_diff = (r_cpu - r_cuda.cpu()).abs().max().item()
            max_w_diff = (w_cpu - w_cuda.cpu()).abs().max().item()

        summary["cuda_parity"] = {
            "max_restored_diff": max_r_diff,
            "max_routing_diff": max_w_diff,
        }
        print(f"Max restored diff: {max_r_diff:.2e} | Max routing diff: {max_w_diff:.2e}")

    print("\n" + "=" * 80)
    print("ADVERSARIAL STRESS SUITE COMPLETE")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    run_all_adversarial_stress_benchmarks()
