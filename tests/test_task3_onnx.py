"""tests/test_task3_onnx.py
-------------------------
Unit tests for Task 3 Step 10: Single-Graph ONNX Export, Parity Verification, and Benchmarking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from src.task3.export_onnx import (
    OnnxSoftMoE,
    benchmark_model_latency,
    export_soft_moe_onnx,
    verify_numerical_parity,
)
from src.task3.moe_model import SoftMoE


@pytest.fixture
def test_moe() -> SoftMoE:
    """Fixture providing a lightweight SoftMoE for fast unit testing."""
    model = SoftMoE(
        channels=(16, 32, 64),
        bottleneck_dim=64,
    )
    model.eval()
    return model


def test_export_soft_moe_onnx_valid(test_moe: SoftMoE) -> None:
    """Verify export_soft_moe_onnx generates valid ONNX graph with correct nodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        result_path = export_soft_moe_onnx(test_moe, out_onnx, tau=1.5, opset_version=17)

        assert result_path.is_file()
        assert result_path.stat().st_size > 100_000

        model = onnx.load(str(result_path))
        onnx.checker.check_model(model)

        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]
        assert "input_image" in input_names
        assert "restored_image" in output_names
        assert "routing_weights" in output_names


def test_verify_numerical_parity_passes(test_moe: SoftMoE) -> None:
    """Verify numerical parity assertion passes with high precision on dynamic batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        export_soft_moe_onnx(test_moe, out_onnx, tau=1.5, opset_version=17)

        parity = verify_numerical_parity(
            test_moe, out_onnx, tau=1.5, batch_sizes=(1, 4), atol=1e-5
        )
        assert parity["all_passed"] is True
        for b in [1, 4]:
            assert parity[f"batch_{b}"]["passed"] is True
            assert parity[f"batch_{b}"]["max_restored_abs_diff"] < 1e-5
            assert parity[f"batch_{b}"]["max_weights_abs_diff"] < 1e-5


def test_benchmark_model_latency(test_moe: SoftMoE) -> None:
    """Verify benchmark_model_latency returns valid numbers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        export_soft_moe_onnx(test_moe, out_onnx, tau=1.5, opset_version=17)

        lat = benchmark_model_latency(
            test_moe, out_onnx, tau=1.5, num_warmup=2, num_runs=5
        )
        assert lat["pytorch_ms"] > 0.0
        assert lat["onnxruntime_ms"] > 0.0
        assert lat["speedup"] > 0.0
        assert lat["throughput_fps"] > 0.0


def test_onnx_soft_moe_inference(test_moe: SoftMoE) -> None:
    """Verify OnnxSoftMoE engine produces expected tensor shapes for single and batched inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_moe.onnx"
        export_soft_moe_onnx(test_moe, out_onnx, tau=1.5, opset_version=17)

        engine = OnnxSoftMoE(out_onnx)

        # Single 3D image
        img_3d = np.random.rand(3, 128, 128).astype(np.float32)
        restored_3d, weights_3d = engine.predict(img_3d)
        assert restored_3d.shape == (1, 3, 128, 128)
        assert weights_3d.shape == (1, 4)
        assert pytest.approx(float(np.sum(weights_3d)), abs=1e-5) == 1.0

        # Batched 4D tensor
        img_4d = np.random.rand(4, 3, 128, 128).astype(np.float32)
        restored_4d, weights_4d = engine.predict(img_4d)
        assert restored_4d.shape == (4, 3, 128, 128)
        assert weights_4d.shape == (4, 4)
        row_sums = np.sum(weights_4d, axis=-1)
        for s in row_sums:
            assert pytest.approx(float(s), abs=1e-5) == 1.0
