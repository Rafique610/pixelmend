"""
tests/test_task1_evaluate.py
----------------------------
Unit tests for Task 1 evaluation pipeline, table generation, and artifact exports.
"""

from pathlib import Path
import pytest
import torch

from src.task1.evaluate import (
    build_failure_cases_selection,
    build_qualitative_sample_selection,
    generate_markdown_table,
    run_evaluation,
)


def test_markdown_table_generation() -> None:
    mock_metrics = {
        "overall": {"psnr": 20.5, "ssim": 0.62, "mae": 0.07, "mse": 0.01},
        "per_corruption": {
            "clean": {"psnr": 21.0, "ssim": 0.64, "mae": 0.06, "mse": 0.008},
            "salt_and_pepper": {"psnr": 21.2, "ssim": 0.63, "mae": 0.065, "mse": 0.0085},
            "gaussian_blur": {"psnr": 20.8, "ssim": 0.62, "mae": 0.071, "mse": 0.009},
            "occlusion": {"psnr": 19.0, "ssim": 0.58, "mae": 0.084, "mse": 0.014},
        },
        "per_severity": {
            "clean": {"psnr": 21.0, "ssim": 0.64, "mae": 0.06},
            "sp_mild": {"psnr": 22.0, "ssim": 0.66, "mae": 0.05},
        },
    }
    table_str = generate_markdown_table(mock_metrics)
    assert "Per-Corruption Summary Table" in table_str
    assert "Per-Severity Breakdown Table" in table_str
    assert "20.50 dB" in table_str


def test_qualitative_and_failure_selection() -> None:
    dummy_samples = [
        {
            "image": f"pet_{i}.jpg",
            "variant_name": name,
            "ssim": 0.5 + i * 0.01,
            "psnr": 18.0 + i * 0.2,
        }
        for i, name in enumerate(["clean", "sp_mild", "blur_medium", "occl_severe", "sp_severe"])
    ]
    qual = build_qualitative_sample_selection(dummy_samples)
    assert len(qual) == 5
    assert "label_str" in qual[0]

    failures = build_failure_cases_selection(dummy_samples)
    assert len(failures) == 4
    assert "notes" in failures[0]


def test_run_evaluation_smoke(tmp_path: Path) -> None:
    results = run_evaluation(
        checkpoint_path="checkpoints/task1/best_model.pth",
        manifest_path="manifests/test_manifest.json",
        limit=10,
        batch_size=5,
        device="cpu",
        output_dir=str(tmp_path),
        log_mlflow=False,
    )
    assert results["total_evaluated"] == 10
    assert "overall" in results
    assert "psnr" in results["overall"]

    assert (tmp_path / "metrics" / "test_metrics.json").exists()
    assert (tmp_path / "metrics" / "test_summary_table.md").exists()
    assert (tmp_path / "visualizations" / "qualitative_comparison_grid.png").exists()
    assert (tmp_path / "visualizations" / "failure_cases_analysis.png").exists()
