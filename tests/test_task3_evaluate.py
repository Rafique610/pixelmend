"""tests/test_task3_evaluate.py
------------------------------
Unit tests for Task 3 Step 9: Comparative Evaluation and Qualitative Visualization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.task1.autoencoder import UniversalAutoencoder
from src.task2.router import HardRouter, load_hard_router
from src.task3.evaluate import (
    benchmark_model_latencies,
    compute_batch_metrics,
    plot_failure_cases_4,
    plot_qualitative_grid_12,
)
from src.task3.moe_model import SoftMoE



def test_compute_batch_metrics() -> None:
    """Verify compute_batch_metrics computes correct PSNR, SSIM, and MAE bounds."""
    pred = torch.rand(4, 3, 128, 128)
    clean = pred.clone()
    psnr, ssim, mae = compute_batch_metrics(pred, clean)

    assert len(psnr) == 4
    assert len(ssim) == 4
    assert len(mae) == 4
    for i in range(4):
        assert psnr[i] >= 70.0  # Identical images
        assert pytest.approx(ssim[i], abs=1e-4) == 1.0
        assert pytest.approx(mae[i], abs=1e-6) == 0.0


def test_benchmark_model_latencies_cpu() -> None:
    """Verify latency benchmark returns valid positive millisecond numbers."""
    t1 = UniversalAutoencoder()
    t1.eval()
    t2 = load_hard_router()
    t2.eval()
    t3 = SoftMoE()
    t3.eval()

    dev = torch.device("cpu")
    lats = benchmark_model_latencies(t1, t2, t3, dev, tau=2.5, num_runs=2)

    assert "task1_universal" in lats
    assert "task2_predicted" in lats
    assert "task2_oracle" in lats
    assert "task3_soft_moe" in lats
    for k, v in lats.items():
        assert v > 0.0


def test_plot_qualitative_grid_12(tmp_path: Path) -> None:
    """Verify 12-sample qualitative grid exports valid PNG image."""
    candidates = {}
    for i in range(12):
        k = f"variant_{i}"
        candidates[k] = {
            "corr": torch.rand(3, 128, 128),
            "clean": torch.rand(3, 128, 128),
            "out_t1": torch.rand(3, 128, 128),
            "out_t2": torch.rand(3, 128, 128),
            "out_t3": torch.rand(3, 128, 128),
            "v_name": k,
            "p_t1": 20.0,
            "p_t2": 21.0,
            "p_t3": 22.0,
            "s_t1": 0.6,
            "s_t2": 0.65,
            "s_t3": 0.7,
        }

    out_file = tmp_path / "qual_test.png"
    res = plot_qualitative_grid_12(candidates, out_file)
    assert res.is_file()
    assert res.stat().st_size > 5000


def test_plot_failure_cases_4(tmp_path: Path) -> None:
    """Verify 4-sample failure case panel exports valid PNG image."""
    failures = [
        {
            "corr": torch.rand(3, 128, 128),
            "clean": torch.rand(3, 128, 128),
            "out_t1": torch.rand(3, 128, 128),
            "out_t2": torch.rand(3, 128, 128),
            "out_t3": torch.rand(3, 128, 128),
            "v_name": f"fail_{i}",
            "desc": f"Failure Case {i+1}",
            "p_t2": 18.0,
            "p_t3": 21.5,
        }
        for i in range(4)
    ]

    out_file = tmp_path / "fail_test.png"
    res = plot_failure_cases_4(failures, {}, out_file)
    assert res.is_file()
    assert res.stat().st_size > 5000
