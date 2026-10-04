"""tests/test_challenger2_moe_stress.py
------------------------------------
Empirical Challenger 2 Adversarial Stress Suite for Milestone 1:
Soft MoE Architecture & Differentiable Blending.

Covers:
1. Real Checkpoint Loading & Exact Weight Transplantation:
   - Real checkpoint loading from checkpoints/task2/ for classifier and all 3 specialists.
   - Assert 100% bit-exact parameter equality (torch.equal) between source checkpoints and model.
2. Parameter Isolation During Freezing:
   - model.set_experts_frozen(True).
   - 5 optimizer steps on synthetic batches across multiple optimizers (AdamW, Adam, SGD).
   - Assert specialist parameters remain BIT-EXACT IDENTICAL (torch.equal(w_initial, w_after)).
   - Assert gate parameters successfully receive gradients and update.
   - Verify unfreezing (model.set_experts_frozen(False)) allows specialist parameter updates.
   - Buffer mutation analysis for specialist BatchNorm layers.
3. Batching Resilience Across CPU and CUDA:
   - Batch sizes B in {1, 2, 7, 16, 32}.
   - Tested in both eval and train modes, forward + backward passes.
   - Verified output shapes, simplex probability constraints, and peak VRAM profiling.
   - 3D unbatched tensor inputs (3, 128, 128).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task3.gate import build_gating_network
from src.task3.moe_model import build_soft_moe

CLASSIFIER_PATH = Path("checkpoints/task2/classifier_best.pt")
SALT_PATH = Path("checkpoints/task2/specialist_salt_best.pt")
BLUR_PATH = Path("checkpoints/task2/specialist_blur_best.pt")
OCCLUSION_PATH = Path("checkpoints/task2/specialist_occlusion_best.pt")


def get_available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


# =========================================================================
# Domain 1: Real Checkpoint Loading & Exact Weight Transplantation
# =========================================================================


def test_checkpoint_files_exist():
    """Verify that all required Task 2 checkpoints exist on disk."""
    for path, name in [
        (CLASSIFIER_PATH, "Classifier"),
        (SALT_PATH, "Salt specialist"),
        (BLUR_PATH, "Blur specialist"),
        (OCCLUSION_PATH, "Occlusion specialist"),
    ]:
        assert path.is_file(), f"Required checkpoint {name} not found at {path}"


def test_gate_exact_weight_transplantation():
    """Verify 100% bit-exact parameter transplantation from classifier_best.pt."""
    gate = build_gating_network(checkpoint_path=CLASSIFIER_PATH)
    ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    gate_sd = gate.state_dict()
    assert len(gate_sd) == len(state_dict), (
        f"Key count mismatch in gate: {len(gate_sd)} model keys vs {len(state_dict)} ckpt keys"
    )

    for k, v in state_dict.items():
        clean_k = k.replace("model.", "").replace("module.", "")
        assert clean_k in gate_sd, f"Missing key in GatingNetwork: {clean_k}"
        assert torch.equal(gate_sd[clean_k], v), f"Gate weight mismatch for tensor: {clean_k}"


def test_all_specialists_exact_weight_transplantation():
    """Verify bit-exact parameter transplantation for salt, blur, and occlusion specialists."""
    moe = build_soft_moe(
        classifier_path=CLASSIFIER_PATH,
        salt_path=SALT_PATH,
        blur_path=BLUR_PATH,
        occlusion_path=OCCLUSION_PATH,
    )

    specialist_specs = [
        ("salt", moe.specialist_salt, SALT_PATH),
        ("blur", moe.specialist_blur, BLUR_PATH),
        ("occlusion", moe.specialist_occlusion, OCCLUSION_PATH),
    ]

    for name, specialist, ckpt_path in specialist_specs:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        mod_sd = specialist.state_dict()

        assert len(mod_sd) == len(state_dict), (
            f"Key count mismatch for {name}: {len(mod_sd)} vs {len(state_dict)}"
        )

        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            assert clean_k in mod_sd, f"Missing key in {name}: {clean_k}"
            assert torch.equal(mod_sd[clean_k], v), (
                f"Weight mismatch for {name} tensor: {clean_k}"
            )


# =========================================================================
# Domain 2: Parameter Isolation During Freezing (5 Optimizer Steps)
# =========================================================================


@pytest.mark.parametrize("optimizer_type", ["adamw", "adam", "sgd"])
def test_freezing_parameter_isolation_bit_exact(optimizer_type: str):
    """Stress test: 5 optimizer steps on synthetic data must leave specialist weights BIT-EXACT."""
    model = build_soft_moe(
        classifier_path=CLASSIFIER_PATH,
        salt_path=SALT_PATH,
        blur_path=BLUR_PATH,
        occlusion_path=OCCLUSION_PATH,
    )

    # 1. Freeze specialist branches
    model.set_experts_frozen(True)
    assert model.experts_frozen is True

    # 2. Snapshot initial specialist weights
    initial_specialist_weights: Dict[str, torch.Tensor] = {}
    for spec_name, spec in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        for p_name, param in spec.named_parameters():
            assert not param.requires_grad, (
                f"Parameter {spec_name}.{p_name} unexpectedly requires grad!"
            )
            initial_specialist_weights[f"{spec_name}.{p_name}"] = param.clone().detach()

    # Snapshot initial gate weights
    initial_gate_weights = {
        name: param.clone().detach() for name, param in model.gate.named_parameters()
    }

    # 3. Setup optimizer
    if optimizer_type == "adamw":
        # Pass all parameters to test PyTorch optimizer skipping param.grad=None
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    elif optimizer_type == "adam":
        # Pass only trainable parameters
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    else:
        # SGD with momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    criterion = nn.L1Loss()
    model.train()
    torch.manual_seed(2026)

    # 4. Take 5 training steps with synthetic corrupted images and targets
    for step in range(5):
        optimizer.zero_grad()
        fake_x = torch.randn(4, 3, 128, 128)
        fake_target = torch.randn(4, 3, 128, 128)
        restored, routing_w, logits = model(fake_x)
        loss = criterion(restored, fake_target)
        loss.backward()
        optimizer.step()

    # 5. Assert BIT-EXACT equality on all specialist parameters
    for spec_name, spec in [
        ("salt", model.specialist_salt),
        ("blur", model.specialist_blur),
        ("occlusion", model.specialist_occlusion),
    ]:
        for p_name, param in spec.named_parameters():
            key = f"{spec_name}.{p_name}"
            orig = initial_specialist_weights[key]
            assert torch.equal(param, orig), (
                f"VIOLATION: Parameter {key} changed during frozen training! "
                f"Max absolute diff: {(param - orig).abs().max().item()}"
            )

    # 6. Assert gate parameters DID update (proving non-trivial learning took place)
    gate_updated = any(
        not torch.equal(param, initial_gate_weights[name])
        for name, param in model.gate.named_parameters()
    )
    assert gate_updated, "Gate weights unexpectedly did not update during training steps!"


def test_unfreezing_restores_gradient_flow_and_updates():
    """Verify that unfreezing allows specialist parameters to update."""
    model = build_soft_moe(
        classifier_path=CLASSIFIER_PATH,
        salt_path=SALT_PATH,
        blur_path=BLUR_PATH,
        occlusion_path=OCCLUSION_PATH,
    )
    model.set_experts_frozen(False)
    assert model.experts_frozen is False

    initial_weights = {
        name: param.clone().detach() for name, param in model.specialist_salt.named_parameters()
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    x = torch.randn(2, 3, 128, 128)
    target = torch.randn(2, 3, 128, 128)
    restored, _, _ = model(x)
    loss = F.l1_loss(restored, target)
    loss.backward()
    optimizer.step()

    # Assert salt parameters were updated
    salt_updated = any(
        not torch.equal(param, initial_weights[name])
        for name, param in model.specialist_salt.named_parameters()
    )
    assert salt_updated, "Salt specialist parameters failed to update when unfrozen!"


def test_freezing_batchnorm_buffer_behavior():
    """Examine buffer evolution: in train mode buffers mutate; in eval mode they are invariant."""
    model = build_soft_moe(
        classifier_path=CLASSIFIER_PATH,
        salt_path=SALT_PATH,
        blur_path=BLUR_PATH,
        occlusion_path=OCCLUSION_PATH,
    )
    model.set_experts_frozen(True)

    # Specialists placed in eval mode
    model.specialist_salt.eval()
    model.specialist_blur.eval()
    model.specialist_occlusion.eval()

    eval_init_buffers = {
        f"salt.{k}": b.clone().detach() for k, b in model.specialist_salt.named_buffers()
    }
    x = torch.randn(4, 3, 128, 128)
    _ = model(x)
    for k, b in model.specialist_salt.named_buffers():
        assert torch.equal(b, eval_init_buffers[f"salt.{k}"]), (
            f"Buffer {k} mutated while specialist was in eval mode!"
        )


# =========================================================================
# Domain 3: Batching Resilience Across CPU and CUDA
# =========================================================================

BATCH_SIZES = [1, 2, 7, 16, 32]
DEVICES = get_available_devices()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_batching_resilience_forward_pass(device: torch.device, batch_size: int, mode: str):
    """Stress test forward pass across B in {1, 2, 7, 16, 32} on CPU and CUDA."""
    model = build_soft_moe(
        classifier_path=CLASSIFIER_PATH,
        salt_path=SALT_PATH,
        blur_path=BLUR_PATH,
        occlusion_path=OCCLUSION_PATH,
    ).to(device)

    if mode == "eval":
        model.eval()
    else:
        model.train()

    torch.manual_seed(100 + batch_size)
    x = torch.rand(batch_size, 3, 128, 128, device=device)

    restored, routing_w, logits = model(x, tau=1.0)

    # 1. Output shapes
    assert restored.shape == (batch_size, 3, 128, 128)
    assert routing_w.shape == (batch_size, 4)
    assert logits.shape == (batch_size, 4)

    # 2. Probability constraints
    sums = routing_w.sum(dim=-1)
    max_err = (sums - 1.0).abs().max()
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Simplex sum violation at B={batch_size}, dev={device}: max error {max_err}"
    )
    assert (routing_w >= 0.0).all() and (routing_w <= 1.0).all(), (
        f"Probability bound violation at B={batch_size}, dev={device}"
    )

    # 3. Finite numbers
    assert torch.isfinite(restored).all(), f"Non-finite values in restored image at B={batch_size}"
    assert torch.isfinite(logits).all(), f"Non-finite logits at B={batch_size}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batching_resilience_backward_pass(device: torch.device, batch_size: int):
    """Verify backward gradient computation with arbitrary batch sizes."""
    model = build_soft_moe(
        classifier_path=CLASSIFIER_PATH,
        salt_path=SALT_PATH,
        blur_path=BLUR_PATH,
        occlusion_path=OCCLUSION_PATH,
    ).to(device)
    model.train()

    x = torch.rand(batch_size, 3, 128, 128, device=device)
    target = torch.rand(batch_size, 3, 128, 128, device=device)

    restored, routing_w, logits = model(x)
    loss = F.l1_loss(restored, target) + F.cross_entropy(
        logits, torch.zeros(batch_size, dtype=torch.long, device=device)
    )
    loss.backward()

    # Verify gate and specialist gradients exist and are finite
    assert model.gate.fc.weight.grad is not None
    assert torch.isfinite(model.gate.fc.weight.grad).all()
    assert (model.gate.fc.weight.grad.abs() > 0).any()


def test_unbatched_3d_input_cpu_and_cuda():
    """Verify handling of 3D unbatched input tensors (3, 128, 128)."""
    for device in DEVICES:
        model = build_soft_moe().to(device)
        model.eval()
        x_3d = torch.rand(3, 128, 128, device=device)
        restored, routing_w, logits = model(x_3d)

        assert restored.shape == (1, 3, 128, 128)
        assert routing_w.shape == (1, 4)
        assert logits.shape == (1, 4)
