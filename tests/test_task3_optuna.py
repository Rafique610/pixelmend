"""tests/test_task3_optuna.py
--------------------------
Unit test suite for Task 3 Soft MoE Optuna Hyperparameter Optimization:
- Routing collapse check boundary tests
- Search space parameter ranges and derived loss multipliers
- Objective function instantiation and single-step trial execution
- Collapse pruning exception raising logic
- Best parameters export to JSON
- Per-corruption validation evaluation structure
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import optuna
from src.task3.moe_model import build_soft_moe
from src.task3.optuna_search import (
    check_routing_collapse,
    compute_derived_lambdas,
    create_moe_objective,
    evaluate_per_corruption,
    export_best_params,
    sample_hyperparameters,
)


def test_check_routing_collapse_normal() -> None:
    """Balanced distributions must not trigger collapse pruning."""
    assert not check_routing_collapse([0.25, 0.25, 0.25, 0.25])
    assert not check_routing_collapse([0.35, 0.25, 0.20, 0.20])
    assert not check_routing_collapse([0.89, 0.03, 0.04, 0.04])
    assert not check_routing_collapse([])


def test_check_routing_collapse_violations() -> None:
    """Extreme distributions (max > 0.90 or min < 0.02) must trigger collapse pruning."""
    assert check_routing_collapse([0.91, 0.03, 0.03, 0.03])
    assert check_routing_collapse([0.95, 0.02, 0.02, 0.01])
    assert check_routing_collapse([0.50, 0.35, 0.14, 0.01])
    assert check_routing_collapse([0.60, 0.38, 0.02, 0.00])


def test_search_space_ranges() -> None:
    """Verify all sampled hyperparameters fall strictly within configured bounds."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = sample_hyperparameters(trial)

    assert 1e-5 <= params["fine_tune_lr"] <= 1e-3
    assert 0.1 <= params["temperature"] <= 5.0
    assert 0.01 <= params["lambda_ce"] <= 0.5
    assert 0.001 <= params["lambda_balance"] <= 0.1
    assert 0.5 <= params["reconstruction_alpha"] <= 1.0

    lambdas = compute_derived_lambdas(params)
    assert lambdas[0] == params["reconstruction_alpha"]
    assert pytest.approx(lambdas[0] + lambdas[1], abs=1e-6) == 1.0
    assert lambdas[2] == params["lambda_ce"]
    assert lambdas[3] == params["lambda_balance"]


def test_export_best_params_json(tmp_path: Path) -> None:
    """Verify study results correctly serialize to config/task3_best_params.json."""
    study = optuna.create_study(study_name="task3-moe-joint", direction="maximize")
    dists = {
        "fine_tune_lr": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
        "temperature": optuna.distributions.FloatDistribution(0.1, 5.0),
        "lambda_ce": optuna.distributions.FloatDistribution(0.01, 0.5, log=True),
        "lambda_balance": optuna.distributions.FloatDistribution(0.001, 0.1, log=True),
        "reconstruction_alpha": optuna.distributions.FloatDistribution(0.5, 1.0),
    }
    t1 = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE, value=0.7100,
        params={"fine_tune_lr": 1e-4, "temperature": 1.5, "lambda_ce": 0.05,
                "lambda_balance": 0.01, "reconstruction_alpha": 0.8},
        distributions=dists,
    )
    t2 = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE, value=0.7450,
        params={"fine_tune_lr": 2e-4, "temperature": 0.8, "lambda_ce": 0.10,
                "lambda_balance": 0.02, "reconstruction_alpha": 0.7},
        distributions=dists,
    )
    study.add_trial(t1)
    study.add_trial(t2)

    json_path = tmp_path / "task3_best_params.json"
    exported = export_best_params(study, output_path=json_path)

    assert json_path.is_file()
    assert exported["best_trial_number"] == 1
    assert exported["best_val_ssim"] == 0.7450
    assert exported["best_params"]["temperature"] == 0.8
    assert exported["derived_lambdas"]["lambda_l1"] == 0.7
    assert exported["derived_lambdas"]["lambda_ssim"] == 0.3
    assert exported["total_trials"] == 2
    assert exported["completed_trials"] == 2


def test_evaluate_per_corruption_structure() -> None:
    """Verify evaluate_per_corruption returns valid per-class metrics dictionary."""
    torch.manual_seed(42)
    corr = torch.rand(8, 3, 128, 128)
    clean = torch.rand(8, 3, 128, 128)
    lbl = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    loader = DataLoader(TensorDataset(corr, clean, lbl), batch_size=4)

    model = build_soft_moe(channels=(16, 32, 64, 128), bottleneck_dim=128)
    dev = torch.device("cpu")
    metrics = evaluate_per_corruption(model, loader, dev, tau=1.0)

    assert "val_psnr" in metrics
    assert "val_ssim" in metrics
    assert "val_mae" in metrics
    assert "avg_routing_vector" in metrics
    assert "per_corruption" in metrics

    per_c = metrics["per_corruption"]
    for c_name in ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"]:
        assert c_name in per_c
        assert "val_psnr" in per_c[c_name]
        assert "val_ssim" in per_c[c_name]
        assert "val_mae" in per_c[c_name]
        assert "avg_routing_vector" in per_c[c_name]
        assert len(per_c[c_name]["avg_routing_vector"]) == 4


def test_moe_objective_execution_step() -> None:
    """Verify Optuna objective executes 1 warmup and 1 joint step on small mock data."""
    torch.manual_seed(42)
    clean = torch.rand(4, 3, 128, 128)
    corr = (clean + 0.05 * torch.randn(4, 3, 128, 128)).clamp(0.0, 1.0)
    lbl = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loader = DataLoader(TensorDataset(corr, clean, lbl), batch_size=2)

    dev = torch.device("cpu")
    objective = create_moe_objective(loader, loader, dev, warmup_epochs=1, joint_epochs=1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    t = study.trials[0]
    assert t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    if t.state == optuna.trial.TrialState.COMPLETE:
        assert isinstance(t.value, float)
        assert -1.0 <= t.value <= 1.0

