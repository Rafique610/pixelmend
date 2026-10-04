"""Unit tests for Task 2 Specialist Optuna hyperparameter optimization."""

from __future__ import annotations

import optuna
import pytest
import torch

from src.task2.optuna_specialists import (
    SPECIALIST_CHANNEL_CONFIGS,
    SimplePairDataset,
    create_specialist_objective,
)


def test_specialist_channel_configs():
    """Verify channel configurations are non-empty and increasing."""
    for name, channels in SPECIALIST_CHANNEL_CONFIGS.items():
        assert len(channels) in (3, 4)
        assert all(c > 0 for c in channels)
        assert channels[0] <= channels[1] <= channels[2]


def test_simple_pair_dataset():
    """Verify SimplePairDataset wraps tensor pairs correctly."""
    pairs = [(torch.rand(3, 128, 128), torch.rand(3, 128, 128)) for _ in range(4)]
    dataset = SimplePairDataset(pairs)
    assert len(dataset) == 4
    corr, clean = dataset[0]
    assert corr.shape == (3, 128, 128)
    assert clean.shape == (3, 128, 128)


def test_specialist_objective_single_trial():
    """Verify create_specialist_objective executes 1 trial and returns valid loss."""
    fake_train = [(torch.rand(3, 128, 128), torch.rand(3, 128, 128)) for _ in range(8)]
    fake_val = [(torch.rand(3, 128, 128), torch.rand(3, 128, 128)) for _ in range(4)]

    objective = create_specialist_objective(
        train_pairs=fake_train,
        val_pairs=fake_val,
        epochs_per_trial=1,
        device=torch.device("cpu"),
        seed=42,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.value is not None
    assert trial.value > 0.0
    assert "learning_rate" in trial.params
    assert "bottleneck_dim" in trial.params
    assert "channel_config" in trial.params
    assert "batch_size" in trial.params
    assert "alpha" in trial.params
    assert "use_residual" in trial.params
