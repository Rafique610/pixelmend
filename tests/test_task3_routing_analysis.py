"""tests/test_task3_routing_analysis.py
------------------------------------
Unit tests for Task 3 Step 8: Routing Behavior Analysis and Visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.task3.moe_model import SoftMoE
from src.task3.routing_analysis import (
    audit_expert_health,
    compute_routing_confusion_matrix,
    export_confusion_matrix_csv,
    extract_and_plot_routing_galleries,
    plot_routing_heatmap,
    plot_severity_routing_trends,
)


@pytest.fixture
def mock_loader() -> DataLoader:
    """Mock dataloader with 16 samples across all 4 corruption classes."""
    torch.manual_seed(42)
    corr = torch.rand(16, 3, 128, 128)
    clean = torch.rand(16, 3, 128, 128)
    lbl = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
    ds = TensorDataset(corr, clean, lbl)
    return DataLoader(ds, batch_size=4)


def test_compute_routing_confusion_matrix(mock_loader: DataLoader) -> None:
    """Verify routing confusion matrix is 4x4 and row sums equal 1.0 within numerical precision."""
    m = SoftMoE()
    m.eval()
    dev = torch.device("cpu")
    matrix = compute_routing_confusion_matrix(m, mock_loader, dev, tau=2.5)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 4)
    # Each row is a probability distribution averaging softmax outputs
    for r in range(4):
        assert pytest.approx(float(np.sum(matrix[r])), abs=1e-5) == 1.0
        assert np.all(matrix[r] >= 0.0)


def test_export_confusion_matrix_csv(tmp_path: Path) -> None:
    """Verify CSV exporter produces correct format and parses properly."""
    matrix = np.array([
        [0.70, 0.10, 0.10, 0.10],
        [0.15, 0.65, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
        [0.10, 0.10, 0.10, 0.70],
    ])
    csv_file = tmp_path / "cm_test.csv"
    res_path = export_confusion_matrix_csv(matrix, csv_file)
    assert res_path.is_file()

    lines = res_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    assert "True Corruption,w_Clean,w_Salt,w_Blur,w_Occlusion" in lines[0]
    clean_vals = [float(x) for x in lines[1].split(",")[1:]]
    assert clean_vals[0] == 0.70


def test_plot_routing_heatmap(tmp_path: Path) -> None:
    """Verify heatmap renderer saves valid PNG."""
    matrix = np.full((4, 4), 0.25)
    img_path = tmp_path / "heatmap_test.png"
    out = plot_routing_heatmap(matrix, img_path)
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_plot_severity_routing_trends(tmp_path: Path) -> None:
    """Verify severity routing trend plot generates 3-panel figure."""
    data = {
        "salt_and_pepper": {1: [0.3, 0.5, 0.1, 0.1], 2: [0.2, 0.6, 0.1, 0.1], 3: [0.1, 0.7, 0.1, 0.1]},
        "gaussian_blur": {1: [0.3, 0.1, 0.5, 0.1], 2: [0.2, 0.1, 0.6, 0.1], 3: [0.1, 0.1, 0.7, 0.1]},
        "occlusion": {1: [0.3, 0.1, 0.1, 0.5], 2: [0.2, 0.1, 0.1, 0.6], 3: [0.1, 0.1, 0.1, 0.7]},
    }
    img_path = tmp_path / "severity_test.png"
    out = plot_severity_routing_trends(data, img_path)
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_audit_expert_health() -> None:
    """Verify expert health check detects starvation correctly."""
    healthy_matrix = np.eye(4)
    res_healthy = audit_expert_health(healthy_matrix)
    assert res_healthy["health_status"] == "PASS"
    assert not res_healthy["starvation_detected"]

    # Starvation scenario (expert 2 gets ~0 allocation everywhere)
    starved_matrix = np.array([
        [0.60, 0.40, 0.00, 0.00],
        [0.40, 0.60, 0.00, 0.00],
        [0.50, 0.50, 0.00, 0.00],
        [0.50, 0.50, 0.00, 0.00],
    ])
    res_starved = audit_expert_health(starved_matrix)
    assert res_starved["health_status"] == "WARNING"
    assert res_starved["starvation_detected"]


def test_extract_and_plot_routing_galleries(mock_loader: DataLoader, tmp_path: Path) -> None:
    """Verify routing galleries generate multi-sample visualization."""
    m = SoftMoE()
    m.eval()
    dev = torch.device("cpu")
    img_path = tmp_path / "galleries_test.png"
    out = extract_and_plot_routing_galleries(m, mock_loader, dev, tau=2.0, output_path=img_path)
    assert out.is_file()
    assert out.stat().st_size > 5000
