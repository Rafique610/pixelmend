"""
tests/test_task1_onnx.py
------------------------
Unit tests for Task 1 ONNX export, checker validation, parity, and dynamic batching.
"""

from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from src.task1.autoencoder import UniversalAutoencoder
from src.task1.export_onnx import (
    export_model_to_onnx,
    verify_numerical_parity,
)


@pytest.fixture(scope="module")
def sample_model() -> UniversalAutoencoder:
    """Provide initialized lightweight model for fast ONNX testing."""
    checkpoint_path = Path("checkpoints/task1/best_model.pth")
    if checkpoint_path.exists():
        model, _ = UniversalAutoencoder.load_from_checkpoint(checkpoint_path, device="cpu")
    else:
        model = UniversalAutoencoder(channels=[32, 64], bottleneck_dim=64)
    model.eval()
    return model


def test_onnx_export_and_checker(sample_model: UniversalAutoencoder, tmp_path: Path) -> None:
    onnx_file = tmp_path / "test_model.onnx"
    export_model_to_onnx(sample_model, onnx_file, opset_version=17)

    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 0

    onnx_m = onnx.load(str(onnx_file))
    onnx.checker.check_model(onnx_m)


def test_onnx_numerical_parity(sample_model: UniversalAutoencoder, tmp_path: Path) -> None:
    onnx_file = tmp_path / "parity_model.onnx"
    export_model_to_onnx(sample_model, onnx_file, opset_version=17)

    parity_stats = verify_numerical_parity(
        sample_model,
        onnx_file,
        batch_sizes=(1, 2),
        atol=1e-4,
    )
    assert parity_stats["all_passed"] is True
    assert parity_stats["batch_1"]["max_abs_diff"] < 1e-4
    assert parity_stats["batch_2"]["max_abs_diff"] < 1e-4


def test_onnx_dynamic_batching(sample_model: UniversalAutoencoder, tmp_path: Path) -> None:
    onnx_file = tmp_path / "dynamic_model.onnx"
    export_model_to_onnx(sample_model, onnx_file, opset_version=17)

    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    for b in [1, 3, 5]:
        x = np.random.randn(b, 3, 128, 128).astype(np.float32)
        out = sess.run(None, {"input": x})[0]
        assert out.shape == (b, 3, 128, 128)
