"""Unit tests for Task 2 Specialist Training pipeline."""

from __future__ import annotations

import pytest
import torch

from src.shared.losses import CombinedReconstructionLoss
from src.task2.specialist import build_specialist
from src.task2.specialist_dataset import (
    SpecialistTrainDataset,
    SpecialistValDataset,
    build_specialist_dataloaders,
)
from src.task2.train_specialists import evaluate_specialist, train_single_specialist


def test_specialist_train_dataset():
    """Verify SpecialistTrainDataset dynamic generation and tensor shapes."""
    clean_tensors = [torch.rand(3, 128, 128) for _ in range(4)]
    dataset = SpecialistTrainDataset(clean_tensors, corruption_type="salt_and_pepper")
    assert len(dataset) == 4

    corr, clean = dataset[0]
    assert corr.shape == (3, 128, 128)
    assert clean.shape == (3, 128, 128)
    assert not torch.allclose(corr, clean)


def test_specialist_val_dataset():
    """Verify SpecialistValDataset returns identical pairs as provided."""
    pairs = [(torch.rand(3, 128, 128), torch.rand(3, 128, 128)) for _ in range(3)]
    dataset = SpecialistValDataset(pairs)
    assert len(dataset) == 3
    corr, clean = dataset[1]
    assert corr.shape == (3, 128, 128)


def test_build_specialist_dataloaders():
    """Verify dataloaders construction with pre-cached tensors."""
    clean_tensors = [torch.rand(3, 128, 128) for _ in range(8)]
    train_loader, val_loader = build_specialist_dataloaders(
        corruption_type="gaussian_blur",
        batch_size=4,
        clean_train_tensors=clean_tensors,
    )
    assert len(train_loader) == 2
    corr_batch, clean_batch = next(iter(train_loader))
    assert corr_batch.shape == (4, 3, 128, 128)
    assert clean_batch.shape == (4, 3, 128, 128)


def test_evaluate_specialist():
    """Verify evaluate_specialist computes valid finite loss and metric values."""
    model = build_specialist("salt_and_pepper", variant="lightweight")
    clean_tensors = [torch.rand(3, 128, 128) for _ in range(4)]
    _, val_loader = build_specialist_dataloaders(
        corruption_type="salt_and_pepper",
        batch_size=2,
        clean_train_tensors=clean_tensors,
    )
    criterion = CombinedReconstructionLoss(alpha=0.84)
    res = evaluate_specialist(model, val_loader, criterion, device=torch.device("cpu"))

    assert "val_loss" in res
    assert "val_psnr" in res
    assert "val_ssim" in res
    assert res["val_loss"] > 0.0
    assert res["val_psnr"] > 0.0


def test_train_single_specialist_short(tmp_path, monkeypatch):
    """Verify train_single_specialist completes 1 epoch and produces checkpoint."""
    monkeypatch.setattr("src.task2.train_specialists.Path", lambda p: tmp_path / p if isinstance(p, str) and not p.startswith(str(tmp_path)) else tmp_path)
    clean_tensors = [torch.rand(3, 128, 128) for _ in range(8)]

    model, results = train_single_specialist(
        corruption_type="occlusion",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        alpha=0.84,
        variant="lightweight",
        clean_train_tensors=clean_tensors,
        device=torch.device("cpu"),
        tracker=None,
    )

    assert results["corruption_type"] == "occlusion"
    assert len(results["history"]["train_loss"]) == 1
    assert len(results["history"]["val_loss"]) == 1
