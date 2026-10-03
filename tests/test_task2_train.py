"""
tests/test_task2_train.py
-------------------------
Unit tests for Task 2 Corruption Classifier training pipeline and datasets.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.task2.classifier import build_classifier
from src.task2.dataset import (
    BalancedBatchSampler,
    BalancedDynamicTrainDataset,
    CachedValDataset,
)
from src.task2.train_classifier import evaluate_classifier, train_classifier


def test_balanced_dynamic_train_dataset():
    """Verify BalancedDynamicTrainDataset assigns cyclically balanced labels and produces valid tensors."""
    fake_clean = [torch.rand(3, 128, 128) for _ in range(12)]
    dataset = BalancedDynamicTrainDataset(fake_clean, image_size=128)

    assert len(dataset) == 12
    # Check cyclic label assignment
    assert dataset.labels == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

    for i in range(4):
        corr, lbl = dataset[i]
        assert corr.shape == (3, 128, 128)
        assert lbl == i
        assert (corr >= 0.0).all() and (corr <= 1.0).all()


def test_balanced_batch_sampler():
    """Verify BalancedBatchSampler strictly enforces equal class representation in every batch."""
    labels = [i % 4 for i in range(64)]  # 16 of each class
    sampler = BalancedBatchSampler(labels=labels, batch_size=16, shuffle=True, seed=42)

    assert len(sampler) == 4  # 64 / 16 = 4 batches

    for batch_indices in sampler:
        assert len(batch_indices) == 16
        batch_labels = [labels[idx] for idx in batch_indices]
        for c in range(4):
            assert batch_labels.count(c) == 4  # exactly 16 / 4 = 4 per class


def test_balanced_batch_sampler_invalid_batch_size():
    """Verify sampler raises ValueError if batch size is not divisible by 4."""
    labels = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="divisible by 4"):
        _ = BalancedBatchSampler(labels=labels, batch_size=10)


def test_evaluate_classifier_metrics():
    """Verify evaluation returns all required scalar metrics and normalized confusion matrix."""
    model = build_classifier(backbone="custom_conv")
    fake_val_pairs = [(torch.rand(3, 128, 128), i % 4) for i in range(16)]
    loader = DataLoader(CachedValDataset(fake_val_pairs), batch_size=8)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_classifier(model, loader, criterion, torch.device("cpu"))

    assert "val_loss" in metrics
    assert "val_acc" in metrics
    assert "val_macro_f1" in metrics
    assert "val_macro_precision" in metrics
    assert "val_macro_recall" in metrics
    assert "per_class" in metrics
    assert len(metrics["per_class"]) == 4

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (4, 4)
    # Check row-normalization
    row_sums = cm.sum(axis=1)
    for s in row_sums:
        if s > 0:
            assert np.isclose(s, 1.0, atol=1e-4)


def test_train_classifier_smoke_run():
    """Verify train_classifier executes stably for 1 epoch on custom run name without errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = train_classifier(
            epochs=1,
            batch_size=32,
            lr=1e-3,
            run_name="classifier-smoke-test",
            seed=42,
        )

        assert "best_val_macro_f1" in summary
        assert "history" in summary
        assert len(summary["history"]["train_loss"]) == 1
        assert Path("checkpoints/task2/classifier_latest.pt").is_file()

        # Clean up smoke test output artifacts
        for f in Path("results/task2").glob("classifier-smoke-test*"):
            f.unlink(missing_ok=True)
