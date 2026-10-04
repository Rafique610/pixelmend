"""
tests/test_task4_eval_onnx.py
-----------------------------
Unit tests for Task 4 evaluation and ONNX export modules:
1. Tests quantitative evaluation pipeline functions.
2. Tests ONNX model export and structural validation.
3. Tests numerical parity verification between PyTorch and ONNX Runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from src.task4.generator import UNetGenerator
from src.task4.export_onnx import (
    export_generator_onnx,
    verify_numerical_parity,
    benchmark_inference_latency,
)


@pytest.fixture
def dummy_generator_checkpoint(tmp_path: Path) -> Path:
    """Create a temporary generator checkpoint."""
    net_g = UNetGenerator(base_channels=32, embed_dim=16)
    cp_path = tmp_path / "dummy_generator.pth"
    torch.save(net_g.state_dict(), cp_path)
    return cp_path


def test_export_generator_onnx(dummy_generator_checkpoint: Path, tmp_path: Path):
    """Verify that export_generator_onnx produces a valid ONNX graph."""
    out_onnx = tmp_path / "test_gen.onnx"
    # Temporarily monkeypatch base_channels inside export function or pass custom model
    net_g = UNetGenerator(base_channels=64, embed_dim=16)
    cp_64 = tmp_path / "gen_64.pth"
    torch.save(net_g.state_dict(), cp_64)

    exported_path = export_generator_onnx(checkpoint_path=cp_64, output_path=out_onnx)
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

    # ONNX check
    model = onnx.load(str(exported_path))
    onnx.checker.check_model(model)


def test_numerical_parity_check(tmp_path: Path):
    """Verify numerical parity assertion logic."""
    net_g = UNetGenerator(base_channels=64, embed_dim=16)
    cp_path = tmp_path / "gen_parity.pth"
    torch.save(net_g.state_dict(), cp_path)

    onnx_path = tmp_path / "gen_parity.onnx"
    export_generator_onnx(checkpoint_path=cp_path, output_path=onnx_path)

    parity_res = verify_numerical_parity(
        onnx_path=onnx_path,
        checkpoint_path=cp_path,
        batch_sizes=(1, 2),
        tolerance=1e-4,
    )
    assert parity_res["all_passed"] is True
    assert "batch_1" in parity_res
    assert parity_res["batch_1"]["max_abs_diff"] < 1e-4
