"""Unit tests for Task 1 Optuna hyperparameter optimization."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
import optuna

from src.shared.config import Settings
from src.shared.optuna_utils import create_or_load_study
from src.task1.optuna_search import create_objective


class DummyDataset(Dataset):
    def __init__(self, size: int = 8):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return torch.rand(3, 128, 128), torch.rand(3, 128, 128), torch.tensor(0)


def test_optuna_objective_and_study_execution(tmp_path: Path):
    """Verify Optuna objective can sample params, run a step, and report loss."""
    db_path = tmp_path / "test_optuna.db"
    settings = Settings(optuna_db=db_path)

    study = create_or_load_study(
        study_name="test-study",
        direction="minimize",
        pruner_name="none",
        settings=settings,
    )

    dummy_val = DataLoader(DummyDataset(4), batch_size=2)

    def dummy_factory(batch_size: int):
        return DataLoader(DummyDataset(4), batch_size=batch_size)

    objective = create_objective(
        train_loader_factory=dummy_factory,
        val_loader=dummy_val,
        epochs_per_trial=1,
        device=torch.device("cpu"),
    )

    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    assert study.best_trial.value is not None
    assert study.best_trial.value > 0.0
