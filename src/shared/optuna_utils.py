"""
src/shared/optuna_utils.py
--------------------------
Centralized Optuna study factory, persistent SQLite storage manager,
trial-to-MLflow synchronization callback, and trial export utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler

import optuna
from src.shared.config import Settings, get_settings
from src.shared.tracking import ExperimentTracker, get_tracker


def get_pruner(pruner_name: str = "median", warmup_steps: int = 5) -> BasePruner:
    """Return configured Optuna pruner instance."""
    normalized = pruner_name.lower().strip()
    if normalized == "median":
        return MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=warmup_steps,
            interval_steps=1,
        )
    if normalized == "hyperband":
        return HyperbandPruner(
            min_resource=1,
            max_resource="auto",
            reduction_factor=3,
        )
    if normalized in {"nop", "none"}:
        return NopPruner()
    raise ValueError(f"Unknown pruner '{pruner_name}'. Choose 'median', 'hyperband', or 'none'.")


def create_or_load_study(
    study_name: str,
    direction: str = "minimize",
    pruner_name: str = "median",
    warmup_steps: int = 5,
    seed: int = 42,
    settings: Optional[Settings] = None,
) -> optuna.Study:
    """Create a new study or reload an existing study from the persistent SQLite database."""
    cfg = settings or get_settings()
    cfg.optuna_db.parent.mkdir(parents=True, exist_ok=True)

    storage = optuna.storages.RDBStorage(
        url=cfg.optuna_db_url,
        engine_kwargs={
            "connect_args": {"timeout": 30},
        },
    )

    sampler = TPESampler(seed=seed)
    pruner = get_pruner(pruner_name=pruner_name, warmup_steps=warmup_steps)

    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def make_tracking_callback(
    tracker: Optional[ExperimentTracker] = None,
    experiment_name: Optional[str] = None,
    metric_name: str = "val_objective",
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Return an Optuna callback that logs each completed or pruned trial to MLflow."""
    t = tracker or get_tracker()

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        target_exp = experiment_name or f"optuna-{study.study_name}"
        run_name = f"trial-{trial.number:03d}"

        with t.run(run_name=run_name, experiment_name=target_exp, nested=True):
            # Log hyperparameters
            t.log_params(trial.params)
            t.log_param("trial_number", trial.number)
            t.log_param("trial_state", trial.state.name)

            if trial.datetime_start and trial.datetime_complete:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                t.log_metric("duration_seconds", duration)

            # Log objective value if available (completed trial)
            if trial.value is not None:
                t.log_metric(metric_name, trial.value)

            # Log intermediate step values if reported
            for step, val in trial.intermediate_values.items():
                t.log_metric(f"intermediate_{metric_name}", val, step=step)

    return callback


def export_study_summary(
    study: optuna.Study,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Export study trials to CSV and summary statistics to JSON for IEEE reports."""
    target_dir = output_dir or Path(get_settings().results_dir) / study.study_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export trials dataframe to CSV
    df: pd.DataFrame = study.trials_dataframe()
    csv_path = target_dir / "trials_history.csv"
    df.to_csv(csv_path, index=False)

    # 2. Build summary dictionary
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
    }

    if len(completed_trials) > 0:
        best = study.best_trial
        summary["best_trial_number"] = best.number
        summary["best_value"] = float(best.value)
        summary["best_params"] = best.params

    json_path = target_dir / "study_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["csv_path"] = str(csv_path)
    summary["json_path"] = str(json_path)
    return summary
