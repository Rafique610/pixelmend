"""
tests/test_config.py
--------------------
Verifies that configuration loading, path resolution, and package imports work properly.
"""

from pathlib import Path

import mlflow
import onnx
import onnxruntime
import torch

import optuna
from src.shared.config import Settings, get_settings


def test_settings_defaults():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.data_dir == Path("data")
    assert settings.checkpoints_dir == Path("checkpoints")
    assert settings.onnx_dir == Path("models/onnx")
    assert settings.manifests_dir == Path("manifests")
    assert settings.seed == 42
    assert settings.resolved_device in {"cuda", "cpu", "mps"}
    assert isinstance(settings.torch_device, torch.device)
    assert "http://localhost:3000" in settings.cors_origins_list


def test_core_dependencies_available():
    assert hasattr(torch, "__version__")
    assert hasattr(optuna, "__version__")
    assert hasattr(mlflow, "__version__")
    assert hasattr(onnx, "__version__")
    assert hasattr(onnxruntime, "__version__")
