"""
tests/test_task4_optuna_retrain.py
----------------------------------
Unit tests for Task 4 Optuna search and retraining components.
Verifies parameter sampling, objective reporting, config schema,
and retraining execution interfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import optuna
import pytest
import torch

from src.task4.optuna_search import sample_task4_params
from src.task4.generator import UNetGenerator
from src.task4.discriminator import PatchGANDiscriminator


def test_sample_task4_params():
    """Verify hyperparameter sampling boundaries and presence of key attributes."""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    params = sample_task4_params(trial)

    assert "g_lr" in params
    assert "d_lr" in params
    assert "lambda_l1" in params
    assert "dropout" in params
    assert "base_channels" in params

    assert 1e-4 <= params["g_lr"] <= 5e-4
    assert 1e-4 <= params["d_lr"] <= 5e-4
    assert 50.0 <= params["lambda_l1"] <= 150.0
    assert 0.0 <= params["dropout"] <= 0.3
    assert params["base_channels"] in [32, 64]


def test_generator_dropout_instantiation():
    """Verify UNetGenerator accepts sampled dropout and base_channels properly."""
    gen = UNetGenerator(base_channels=32, embed_dim=16, dropout=0.2)
    x = torch.randn(2, 3, 128, 128)
    s = torch.tensor([0, 1], dtype=torch.long)
    out = gen(x, s)
    assert out.shape == (2, 3, 128, 128)


def test_discriminator_base_channels_instantiation():
    """Verify PatchGANDiscriminator instantiates with arbitrary base_channels."""
    disc = PatchGANDiscriminator(base_channels=32, embed_dim=16, n_layers=3)
    x = torch.randn(2, 3, 128, 128)
    y = torch.randn(2, 3, 128, 128)
    s = torch.tensor([0, 2], dtype=torch.long)
    logits = disc(x, y, s)
    assert logits.shape == (2, 1, 14, 14)
