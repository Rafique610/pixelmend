"""
tests/test_task2_evaluate.py
----------------------------
Unit tests for Task 2 test set evaluation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.task2.router import HardRouter
from src.task2.evaluate import (
    evaluate_test_set,
    format_tables_and_export,
    run_evaluation,
)


class MockEvalRouter(HardRouter):
    """Deterministic mock router for evaluation testing."""

    def __init__(self) -> None:
        super().__init__(
            classifier=nn.Identity(),
            specialist_salt=nn.Identity(),
            specialist_blur=nn.Identity(),
            specialist_occlusion=nn.Identity(),
        )

    def forward(self, x: torch.Tensor, oracle_labels=None):
        b = x.shape[0] if x.ndim == 4 else 1
        # Predict class 1 for all samples
        decisions = [1] * b if oracle_labels is None else (
            [oracle_labels] * b if isinstance(oracle_labels, int) else list(oracle_labels)
        )
        # Add slight transformation
        rest = x.clone() * 0.98
        return {
            "reconstructed": rest,
            "routing_decision": decisions,
            "predicted_class": ["salt_pepper"] * b,
            "probabilities": [{"clean": 0.0, "salt_pepper": 1.0, "blur": 0.0, "occlusion": 0.0}] * b,
            "selected_expert": ["specialist_salt"] * b,
            "classifier_latency_ms": 1.0,
            "restoration_latency_ms": 5.0,
            "total_latency_ms": 6.0,
        }


def test_evaluate_test_set_mock():
    """Verify evaluate_test_set produces valid summary metrics and failure audit lists."""
    router = MockEvalRouter()
    clean_cache = {f"img_{i}.jpg": torch.rand(3, 128, 128) for i in range(4)}
    entries = [
        {"image": f"img_{i}.jpg", "corruption_type": "clean", "corruption_label": 0, "variant_name": "clean", "params": {}}
        for i in range(2)
    ] + [
        {"image": f"img_{i}.jpg", "corruption_type": "salt_and_pepper", "corruption_label": 1, "variant_name": "sp_mild", "params": {"prob": 0.03}}
        for i in range(2, 4)
    ]

    summary, qual_samples, failures = evaluate_test_set(
        router, entries, clean_cache, batch_size=2, device="cpu"
    )

    assert "overall_predicted" in summary
    assert "overall_oracle" in summary
    assert "psnr" in summary["overall_predicted"]
    assert "ssim" in summary["overall_predicted"]
    assert summary["total_evaluated"] == 4
    assert len(qual_samples) >= 2


def test_format_tables_and_export(tmp_path: Path):
    """Verify markdown and CSV comparative tables generation."""
    summary = {
        "overall_predicted": {"psnr": 21.5, "ssim": 0.62, "mae": 0.065},
        "overall_oracle": {"psnr": 21.8, "ssim": 0.63, "mae": 0.063},
        "per_corruption_predicted": {
            "clean": {"psnr": 30.0, "ssim": 1.0, "mae": 0.0},
            "salt_and_pepper": {"psnr": 20.0, "ssim": 0.55, "mae": 0.07},
        },
        "per_corruption_oracle": {
            "clean": {"psnr": 30.0, "ssim": 1.0, "mae": 0.0},
            "salt_and_pepper": {"psnr": 20.5, "ssim": 0.56, "mae": 0.068},
        },
    }
    t1_path = tmp_path / "t1_metrics.json"
    with open(t1_path, "w") as f:
        json.dump({"overall": {"psnr": 20.16, "ssim": 0.5935, "mae": 0.0742}}, f)

    table_md = format_tables_and_export(summary, t1_path, tmp_path)
    assert "| Overall Mean |" in table_md
    assert "| PSNR (dB) |" in table_md
    assert (tmp_path / "metrics_summary.csv").is_file()


def test_run_evaluation_smoke_run(tmp_path: Path):
    """Verify run_evaluation executes on a small subset without errors."""
    summary = run_evaluation(
        manifest_path="manifests/test_manifest.json",
        limit=4,
        batch_size=2,
        device="cpu",
        output_dir=str(tmp_path),
        log_mlflow=False,
    )

    assert summary["total_evaluated"] == 4
    assert (tmp_path / "metrics_summary.json").is_file()
    assert (tmp_path / "metrics_summary.csv").is_file()
    assert (tmp_path / "test_summary_table.md").is_file()
    assert (tmp_path / "visuals" / "qualitative_comparison_grid.png").is_file()
