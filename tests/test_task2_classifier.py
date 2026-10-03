"""
tests/test_task2_classifier.py
------------------------------
Unit tests for Task 2 Corruption Classifier architectures.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.task2.classifier import (
    CorruptionClassifier,
    CustomConvClassifier,
    MobileNetClassifier,
    ResNet18Classifier,
    build_classifier,
)


@pytest.mark.parametrize("backbone", ["custom_conv", "mobilenet", "resnet18"])
def test_classifier_output_shape(backbone: str):
    """Verify that all classifier backbones produce (B, 4) logits."""
    model = build_classifier(backbone=backbone, in_channels=3, num_classes=4)
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)

    assert logits.shape == (2, 4)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize("backbone", ["custom_conv", "mobilenet", "resnet18"])
def test_classifier_probabilities_and_predictions(backbone: str):
    """Verify that predict_proba returns valid distributions and predict returns valid classes."""
    model = build_classifier(backbone=backbone, in_channels=3, num_classes=4)
    x = torch.randn(4, 3, 128, 128)

    probs = model.predict_proba(x)
    preds = model.predict(x)

    assert probs.shape == (4, 4)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    assert preds.shape == (4,)
    assert (preds >= 0).all() and (preds < 4).all()
    assert preds.dtype in (torch.int64, torch.long)


@pytest.mark.parametrize("backbone", ["custom_conv", "mobilenet", "resnet18"])
def test_classifier_gradient_flow(backbone: str):
    """Ensure gradients flow to all trainable parameters without NaNs."""
    model = build_classifier(backbone=backbone, in_channels=3, num_classes=4)
    x = torch.randn(2, 3, 128, 128)
    target = torch.tensor([0, 2], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


@pytest.mark.parametrize("backbone", ["custom_conv", "mobilenet", "resnet18"])
def test_classifier_extract_features(backbone: str):
    """Verify feature extraction returns 2D pooled embedding tensors."""
    model = build_classifier(backbone=backbone, in_channels=3, num_classes=4)
    x = torch.randn(3, 3, 128, 128)
    feats = model.extract_features(x)

    assert feats.dim() == 2
    assert feats.shape[0] == 3
    assert feats.shape[1] > 0


def test_custom_channels_and_dropout():
    """Verify custom channel progression and dropout parameters in CustomConvClassifier."""
    custom_c = (16, 32, 64)
    model = CustomConvClassifier(in_channels=3, channels=custom_c, num_classes=4, dropout=0.3)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert out.shape == (2, 4)
    assert model.dropout_rate == 0.3
    assert model.fc.in_features == 64


def test_classifier_state_dict_save_load():
    """Verify roundtrip state_dict serialization and exact numerical reproduction."""
    model1 = build_classifier(backbone="custom_conv")
    model2 = build_classifier(backbone="custom_conv")

    x = torch.randn(2, 3, 128, 128)
    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model1(x)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "classifier.pt"
        torch.save(model1.state_dict(), ckpt_path)
        model2.load_state_dict(torch.load(ckpt_path))

    with torch.no_grad():
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_unknown_backbone_raises():
    """Verify that an unsupported backbone name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backbone"):
        _ = build_classifier(backbone="nonexistent_backbone")


def test_parameter_counts_positive():
    """Verify that parameter count helper returns realistic parameter counts."""
    for backbone in ["custom_conv", "mobilenet", "resnet18"]:
        model = build_classifier(backbone=backbone)
        params = model.count_parameters()
        assert params > 50_000, f"Unexpectedly low parameter count {params} for {backbone}"
