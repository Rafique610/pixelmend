"""
tests/test_task2_router.py
--------------------------
Unit tests for Task 2 HardRouter and inference pipeline.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.task2.classifier import build_classifier
from src.task2.router import HardRouter, load_hard_router
from src.task2.specialist import build_specialist
from src.task2.inference import restore_image, benchmark_router


class MockClassifier(nn.Module):
    """Deterministic mock classifier returning predetermined class logits."""

    def __init__(self, target_class: int = 1) -> None:
        super().__init__()
        self.target_class = target_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = torch.zeros((b, 4), device=x.device)
        logits[:, self.target_class] = 10.0
        return logits


class MockSpecialist(nn.Module):
    """Deterministic mock specialist that multiplies input by a constant factor."""

    def __init__(self, multiplier: float = 0.5) -> None:
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.multiplier


@pytest.fixture
def mock_router() -> HardRouter:
    """Fixture providing HardRouter with predictable mock sub-modules."""
    clf = MockClassifier(target_class=1)
    s_salt = MockSpecialist(multiplier=0.1)
    s_blur = MockSpecialist(multiplier=0.2)
    s_occ = MockSpecialist(multiplier=0.3)
    return HardRouter(
        classifier=clf,
        specialist_salt=s_salt,
        specialist_blur=s_blur,
        specialist_occlusion=s_occ,
    )


def test_router_initialization(mock_router: HardRouter) -> None:
    """Verify HardRouter stores components and expert maps."""
    assert len(mock_router.specialists) == 3
    assert 1 in mock_router.specialists
    assert 2 in mock_router.specialists
    assert 3 in mock_router.specialists


def test_router_single_image_routing(mock_router: HardRouter) -> None:
    """Verify inference on a single 3D image tensor."""
    x = torch.rand(3, 128, 128)
    res = mock_router(x)

    assert "reconstructed" in res
    assert res["reconstructed"].shape == (3, 128, 128)
    assert res["predicted_class"] == "salt_pepper"
    assert res["selected_expert"] == "specialist_salt"
    assert res["routing_decision"] == 1
    assert "probabilities" in res
    assert pytest.approx(sum(res["probabilities"].values()), abs=1e-3) == 1.0
    assert res["classifier_latency_ms"] >= 0.0
    assert res["restoration_latency_ms"] >= 0.0
    assert res["total_latency_ms"] >= 0.0
    assert not res["oracle_routing"]


def test_router_identity_bypass_exact(mock_router: HardRouter) -> None:
    """Verify identity bypass produces bit-exact identical tensor with 0 numerical error."""
    x = torch.rand(2, 3, 128, 128)
    # Force clean routing via oracle_labels=0
    res = mock_router(x, oracle_labels=0)

    assert res["selected_expert"] == ["identity_bypass", "identity_bypass"]
    assert res["routing_decision"] == [0, 0]
    # Bit-exact check
    assert torch.all(res["reconstructed"] == x)
    mse = torch.mean((res["reconstructed"] - x) ** 2).item()
    assert mse == 0.0


def test_router_batched_mixed_routing(mock_router: HardRouter) -> None:
    """Verify batch with heterogeneous routing decisions reassembles in original order."""
    x = torch.ones(4, 3, 128, 128)
    # Oracle dispatch: sample 0 -> clean, 1 -> salt (*0.1), 2 -> blur (*0.2), 3 -> occ (*0.3)
    res = mock_router(x, oracle_labels=[0, 1, 2, 3])

    recon = res["reconstructed"]
    assert recon.shape == (4, 3, 128, 128)
    assert res["selected_expert"] == [
        "identity_bypass",
        "specialist_salt",
        "specialist_blur",
        "specialist_occlusion",
    ]
    # Check transformed values
    assert pytest.approx(recon[0].mean().item(), abs=1e-5) == 1.0  # clean bypass
    assert pytest.approx(recon[1].mean().item(), abs=1e-5) == 0.1  # salt
    assert pytest.approx(recon[2].mean().item(), abs=1e-5) == 0.2  # blur
    assert pytest.approx(recon[3].mean().item(), abs=1e-5) == 0.3  # occlusion


def test_router_oracle_tensor_mode(mock_router: HardRouter) -> None:
    """Verify oracle routing with torch.Tensor input."""
    x = torch.rand(2, 3, 128, 128)
    oracle_tensor = torch.tensor([2, 3])
    res = mock_router(x, oracle_labels=oracle_tensor)

    assert res["oracle_routing"] is True
    assert res["routing_decision"] == [2, 3]
    assert res["selected_expert"] == ["specialist_blur", "specialist_occlusion"]


def test_restore_image_pil_input(mock_router: HardRouter) -> None:
    """Verify restore_image accepts PIL images and returns restored PIL images."""
    pil_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    res = restore_image(mock_router, pil_img)

    assert "restored_image" in res
    assert isinstance(res["restored_image"], Image.Image)
    assert res["restored_image"].size == (128, 128)
    assert res["reconstructed"].shape == (3, 128, 128)


def test_load_hard_router_production_checkpoints() -> None:
    """Verify load_hard_router successfully binds real checkpoints from checkpoints/task2/."""
    clf_path = Path("checkpoints/task2/classifier_best.pt")
    salt_path = Path("checkpoints/task2/specialist_salt_best.pt")
    blur_path = Path("checkpoints/task2/specialist_blur_best.pt")
    occ_path = Path("checkpoints/task2/specialist_occlusion_best.pt")

    if not (clf_path.is_file() and salt_path.is_file() and blur_path.is_file() and occ_path.is_file()):
        pytest.skip("Production checkpoints not found in checkpoints/task2/")

    router = load_hard_router(device="cpu")
    assert isinstance(router, HardRouter)

    # Test single real inference pass
    test_inp = torch.rand(1, 3, 128, 128)
    res = router(test_inp)

    assert res["reconstructed"].shape == (1, 3, 128, 128)
    assert res["predicted_class"][0] in ["clean", "salt_pepper", "blur", "occlusion"]
    assert (res["reconstructed"] >= 0.0).all() and (res["reconstructed"] <= 1.0).all()


def test_benchmark_router_metrics(mock_router: HardRouter) -> None:
    """Verify benchmark_router computes single, batched, and per-branch latencies."""
    fake_pairs = [(torch.rand(3, 128, 128), i % 4) for i in range(8)]
    bench = benchmark_router(
        router=mock_router,
        test_tensors=fake_pairs,
        batch_size=4,
        num_warmup=2,
        num_runs=5,
        device=torch.device("cpu"),
    )

    assert "single_image" in bench
    assert bench["single_image"]["mean_latency_ms"] >= 0.0
    assert bench["single_image"]["fps"] >= 0.0
    assert "batched" in bench
    assert bench["batched"]["batch_size"] == 4
    assert bench["batched"]["mean_latency_ms"] >= 0.0
    assert "per_branch_oracle_latency_ms" in bench
    assert len(bench["per_branch_oracle_latency_ms"]) == 4


def test_router_invalid_input_dimensions(mock_router: HardRouter) -> None:
    """Verify router raises ValueError when tensor is not 3D or 4D."""
    with pytest.raises(ValueError, match="3D .* or 4D"):
        mock_router(torch.rand(128, 128))

