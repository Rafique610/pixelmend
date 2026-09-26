"""
tests/test_optuna_utils.py
--------------------------
Unit tests for Optuna SQLite study factory, trial pruning,
MLflow synchronization, and study export utilities.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import optuna
from src.shared.config import Settings
from src.shared.optuna_utils import (
    create_or_load_study,
    export_study_summary,
    get_pruner,
    make_tracking_callback,
)
from src.shared.tracking import ExperimentTracker


@pytest.fixture
def custom_env(tmp_path: Path):
    """Provide isolated SQLite paths for Optuna and MLflow."""
    optuna_db = tmp_path / "optuna" / "test_optuna.db"
    mlflow_db = tmp_path / "mlflow" / "test_mlflow.db"
    settings = Settings(
        optuna_db=optuna_db,
        mlflow_tracking_uri=f"sqlite:///{mlflow_db.resolve()}",
    )
    tracker = ExperimentTracker(settings=settings)
    return settings, tracker, tmp_path


def test_get_pruner():
    median_p = get_pruner("median")
    assert isinstance(median_p, optuna.pruners.MedianPruner)

    hb_p = get_pruner("hyperband")
    assert isinstance(hb_p, optuna.pruners.HyperbandPruner)

    nop_p = get_pruner("none")
    assert isinstance(nop_p, optuna.pruners.NopPruner)

    with pytest.raises(ValueError):
        get_pruner("invalid_pruner_type")


def test_study_persistence_and_reload(custom_env):
    settings, tracker, tmp_path = custom_env
    study_name = "test-persistent-study"

    # 1. Create study and run 3 trials
    study = create_or_load_study(
        study_name=study_name,
        direction="minimize",
        settings=settings,
    )

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5.0, 5.0)
        y = trial.suggest_float("y", -5.0, 5.0)
        return (x - 2.0) ** 2 + (y + 1.0) ** 2

    study.optimize(objective, n_trials=3)
    assert len(study.trials) == 3
    initial_best_value = study.best_value

    # 2. Reload study in a new instance and verify persistence
    reloaded_study = create_or_load_study(
        study_name=study_name,
        direction="minimize",
        settings=settings,
    )
    assert len(reloaded_study.trials) == 3
    assert reloaded_study.best_value == initial_best_value
    assert "x" in reloaded_study.best_params
    assert "y" in reloaded_study.best_params


def test_tracking_callback_and_export(custom_env):
    settings, tracker, tmp_path = custom_env
    study_name = "test-tracking-export-study"

    study = create_or_load_study(
        study_name=study_name,
        direction="minimize",
        settings=settings,
    )

    callback = make_tracking_callback(
        tracker=tracker,
        experiment_name="optuna-sync-test",
        metric_name="val_loss",
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        trial.report(lr * 10, step=0)
        trial.report(lr * 5, step=1)
        return lr * 2

    study.optimize(objective, n_trials=2, callbacks=[callback])
    assert len(study.trials) == 2

    # Verify MLflow recorded the 2 trials
    client = tracker.client
    exp = client.get_experiment_by_name("optuna-sync-test")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 2

    # Test export to CSV & JSON
    export_dir = tmp_path / "exports"
    summary = export_study_summary(study, output_dir=export_dir)

    assert Path(summary["csv_path"]).exists()
    assert Path(summary["json_path"]).exists()

    df = pd.read_csv(summary["csv_path"])
    assert len(df) == 2
    assert "params_lr" in df.columns

    with open(summary["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["study_name"] == study_name
    assert data["completed_trials"] == 2
    assert "best_params" in data
