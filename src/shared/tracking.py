"""
src/shared/tracking.py
----------------------
Decoupled experiment tracking facade backed by MLflow.
Handles run lifecycle, parameter logging, metric histories,
PyTorch tensor / PIL / NumPy image logging, and matplotlib figure artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Optional, Union

import matplotlib.figure
import mlflow
import numpy as np
import torch
from PIL import Image

from src.shared.config import Settings, get_settings


class ExperimentTracker:
    """Wrapper and facade around MLflow tracking for all project tasks."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._init_tracking_uri()

    def _init_tracking_uri(self) -> None:
        """Configure MLflow tracking URI based on project settings."""
        mlflow.set_tracking_uri(self.settings.mlflow_uri)

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        """Return an MLflowClient configured with the active tracking URI."""
        return mlflow.tracking.MlflowClient(tracking_uri=self.settings.mlflow_uri)

    def set_experiment(self, experiment_name: str) -> str:
        """Ensure experiment exists and set it as active, restoring if deleted. Return experiment ID."""
        client = self.client
        exp = client.get_experiment_by_name(experiment_name)
        if exp is not None and exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)
        active_exp = mlflow.set_experiment(experiment_name)
        return active_exp.experiment_id

    def start_run(
        self,
        run_name: str,
        experiment_name: str = "default",
        tags: Optional[dict[str, str]] = None,
        nested: bool = False,
    ) -> mlflow.ActiveRun:
        """Start a new tracked MLflow run."""
        self.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name, tags=tags, nested=nested)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the currently active MLflow run."""
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)

    @contextmanager
    def run(
        self,
        run_name: str,
        experiment_name: str = "default",
        tags: Optional[dict[str, str]] = None,
        nested: bool = False,
    ) -> Generator[ExperimentTracker, None, None]:
        """Context manager for scoping an active run cleanly."""
        self.start_run(
            run_name=run_name,
            experiment_name=experiment_name,
            tags=tags,
            nested=nested,
        )
        try:
            yield self
        except Exception:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters (converts complex types to str)."""
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                clean_params[k] = v
            else:
                clean_params[k] = str(v)
        mlflow.log_params(clean_params)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self.log_params({key: value})

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log scalar metrics at an optional epoch or iteration step."""
        clean_metrics = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar metric at an optional step."""
        self.log_metrics({key: value}, step=step)

    def log_image(
        self,
        image_or_tag: Union[torch.Tensor, np.ndarray, Image.Image, str],
        artifact_or_image: Optional[Union[torch.Tensor, np.ndarray, Image.Image, str]] = None,
        artifact_file: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log an image artifact (supports Tensor [C,H,W] or [H,W,C], ndarray, or PIL).

        Supports both calling conventions:
        - log_image(image, artifact_file="name.png", step=1)
        - log_image("tag_name", image, step=1)
        """
        if isinstance(image_or_tag, str):
            tag = image_or_tag
            img = artifact_or_image
            suffix = f"_step_{step}.png" if step is not None else ".png"
            filename = tag if tag.endswith(".png") else f"{tag}{suffix}"
        else:
            img = image_or_tag
            filename = str(artifact_or_image or artifact_file or "image.png")
            if step is not None and not filename.endswith(f"_{step}.png"):
                stem = Path(filename).stem
                filename = f"{stem}_step_{step}.png"
            elif not filename.endswith(".png"):
                filename = f"{filename}.png"

        assert img is not None, "Image must not be None"
        pil_img = self._to_pil(img)
        mlflow.log_image(pil_img, artifact_file=filename)

    def log_figure(
        self,
        fig: matplotlib.figure.Figure,
        artifact_file: str,
    ) -> None:
        """Log a matplotlib figure object directly as an artifact."""
        mlflow.log_figure(fig, artifact_file=artifact_file)

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log a file or directory as an artifact in MLflow."""
        path_str = str(local_path)
        mlflow.log_artifact(path_str, artifact_path=artifact_path)

    @staticmethod
    def _to_pil(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Image.Image:
        """Convert supported image formats to a normalized PIL Image."""
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.dim() == 3 and t.shape[0] in {1, 3}:
                # Channels-first (C, H, W) -> (H, W, C)
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 4:
                arr = arr.squeeze(0)
            if arr.ndim == 3 and arr.shape[0] in {1, 3}:
                arr = np.transpose(arr, (1, 2, 0))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Handle grayscale single channel
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(2)

        # Scale float values [0.0, 1.0] to [0, 255] uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = arr.astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return Image.fromarray(arr)


@lru_cache(maxsize=1)
def get_tracker() -> ExperimentTracker:
    """Return the cached singleton ExperimentTracker instance."""
    return ExperimentTracker()
