"""
tests/test_task2_optuna.py
--------------------------
Unit tests for Task 2 Corruption Classifier Optuna hyperparameter optimization.
"""

import tempfile
from pathlib import Path
import optuna
import pytest
import torch

from src.shared.config import Settings
from src.task2.optuna_classifier import (
    CHANNEL_CONFIGS,
    create_classifier_objective,
    run_classifier_optuna_study,
)


def test_classifier_objective_executes_and_reports():
    """Verify create_classifier_objective runs a single trial and returns valid Macro-F1."""
    fake_clean = [torch.rand(3, 128, 128) for _ in range(64)]
    fake_val_pairs = [(torch.rand(3, 128, 128), i % 4) for i in range(16)]

    objective = create_classifier_objective(
        clean_tensors=fake_clean,
        cached_val_pairs=fake_val_pairs,
        epochs_per_trial=1,
        device=torch.device("cpu"),
        seed=42,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.value is not None
    assert 0.0 <= trial.value <= 1.0
    assert "learning_rate" in trial.params
    assert "batch_size" in trial.params
    assert "channel_config" in trial.params
    assert "dropout" in trial.params
    assert "weight_decay" in trial.params


def test_channel_configs_mapping():
    """Verify all defined channel configurations exist and have valid tuple progressions."""
    for name, channels in CHANNEL_CONFIGS.items():
        assert len(channels) >= 3
        assert all(c > 0 for c in channels)
        assert channels[0] <= channels[1] <= channels[2]
