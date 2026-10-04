"""tests/test_task3_retrain.py
---------------------------
Unit tests for Task 3 Step 7: Definitive Retraining with Best Hyperparameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.task3.moe_model import SoftMoE
from src.task3.optuna_search import compute_derived_lambdas, evaluate_per_corruption


def test_task3_best_params_json_structure() -> None:
    """Verify config/task3_best_params.json exists and contains all required parameters."""
    path = Path("config/task3_best_params.json")
    assert path.is_file(), f"Missing configuration file: {path}"
    cfg = json.loads(path.read_text(encoding="utf-8"))

    assert "best_params" in cfg
    params = cfg["best_params"]
    assert "fine_tune_lr" in params
    assert "temperature" in params
    assert "lambda_ce" in params
    assert "lambda_balance" in params
    assert "reconstruction_alpha" in params

    assert 1e-5 <= params["fine_tune_lr"] <= 1e-3
    assert 0.1 <= params["temperature"] <= 5.0
    assert 0.01 <= params["lambda_ce"] <= 0.5
    assert 0.001 <= params["lambda_balance"] <= 0.1
    assert 0.5 <= params["reconstruction_alpha"] <= 1.0


def test_compute_derived_lambdas_consistency() -> None:
    """Verify compute_derived_lambdas maps alpha, ce, and balance correctly."""
    params = {
        "fine_tune_lr": 1.75e-5,
        "temperature": 2.5,
        "lambda_ce": 0.02,
        "lambda_balance": 0.05,
        "reconstruction_alpha": 0.7,
    }
    l1, ssim, ce, bal = compute_derived_lambdas(params)
    assert pytest.approx(l1, abs=1e-5) == 0.7
    assert pytest.approx(ssim, abs=1e-5) == 0.3
    assert pytest.approx(ce, abs=1e-5) == 0.02
    assert pytest.approx(bal, abs=1e-5) == 0.05


def test_evaluate_per_corruption_mock() -> None:
    """Verify evaluate_per_corruption computes valid metrics across mock dataset."""
    torch.manual_seed(42)
    clean = torch.rand(8, 3, 128, 128)
    corr = (clean + 0.02 * torch.randn(8, 3, 128, 128)).clamp(0.0, 1.0)
    lbl = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    ds = TensorDataset(corr, clean, lbl)
    loader = DataLoader(ds, batch_size=4)

    m = SoftMoE()
    m.eval()
    dev = torch.device("cpu")
    res = evaluate_per_corruption(m, loader, dev, tau=2.0)

    assert "val_psnr" in res
    assert "val_ssim" in res
    assert "per_corruption" in res
    assert len(res["per_corruption"]) == 4
    for c_name in ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"]:
        assert c_name in res["per_corruption"]
        assert "val_psnr" in res["per_corruption"][c_name]
        assert "val_ssim" in res["per_corruption"][c_name]
