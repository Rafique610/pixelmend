"""
tests/test_tracking.py
----------------------
Unit tests for the MLflow ExperimentTracker facade.
Verifies run lifecycles, metrics, parameters, and multi-format image logging.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from src.shared.tracking import ExperimentTracker, get_tracker


@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    """Fixture providing an isolated ExperimentTracker logging to a temporary SQLite database."""
    from src.shared.config import Settings

    db_path = tmp_path / "test_mlflow.db"
    custom_settings = Settings(mlflow_tracking_uri=f"sqlite:///{db_path.resolve()}")
    return ExperimentTracker(settings=custom_settings)


def test_tracker_singleton():
    t1 = get_tracker()
    t2 = get_tracker()
    assert t1 is t2


def test_run_context_and_scalar_logging(tracker: ExperimentTracker):
    exp_name = "test-scalar-experiment"
    with tracker.run(run_name="unit-test-run", experiment_name=exp_name) as t:
        t.log_params({"lr": 0.001, "batch_size": 32, "model": "test_net"})
        t.log_param("optimizer", "adamw")

        for epoch in range(3):
            t.log_metrics({"train_loss": 1.0 / (epoch + 1), "val_psnr": 20.0 + epoch}, step=epoch)
            t.log_metric("val_ssim", 0.85 + (epoch * 0.02), step=epoch)

    # Verify run exists and is recorded in MLflow
    client = tracker.client
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["lr"] == "0.001"
    assert run.data.params["optimizer"] == "adamw"
    assert "train_loss" in run.data.metrics


def test_image_and_figure_logging(tracker: ExperimentTracker):
    exp_name = "test-artifact-experiment"
    with tracker.run(run_name="artifact-test-run", experiment_name=exp_name) as t:
        # 1. PyTorch Tensor (C, H, W) normalized [0, 1]
        tensor_img = torch.rand(3, 64, 64)
        t.log_image(tensor_img, artifact_file="images/torch_sample.png")

        # 2. NumPy ndarray (H, W, C) uint8
        np_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        t.log_image(np_img, artifact_file="images/numpy_sample.png")

        # 3. PIL Image
        pil_img = Image.new("RGB", (64, 64), color="blue")
        t.log_image(pil_img, artifact_file="images/pil_sample.png")

        # 4. Matplotlib Figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_title("Test Curve")
        t.log_figure(fig, artifact_file="plots/test_curve.png")
        plt.close(fig)

    # Verify artifacts were logged
    client = tracker.client
    exp = client.get_experiment_by_name(exp_name)
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    artifacts = client.list_artifacts(runs[0].info.run_id)
    artifact_paths = [a.path for a in artifacts]
    assert "images" in artifact_paths or "plots" in artifact_paths
