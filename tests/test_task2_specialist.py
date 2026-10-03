"""Unit tests for Task 2 Specialist Autoencoders."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.shared.losses import CombinedReconstructionLoss
from src.task2.specialist import (
    SpecialistAutoencoder,
    VALID_SPECIALISTS,
    build_specialist,
)


def test_specialist_initialization():
    """Verify specialist initialization with default and custom configurations."""
    model = SpecialistAutoencoder(corruption_type="salt_and_pepper")
    assert model.corruption_type == "salt_and_pepper"
    assert model.in_channels == 3
    assert model.out_channels == 3
    assert model.channels == (32, 64, 128, 256)
    assert model.bottleneck_dim == 256
    assert model.count_parameters() > 1_000_000


def test_specialist_invalid_corruption():
    """Verify ValueError when invalid corruption type is supplied."""
    with pytest.raises(ValueError, match="Invalid corruption_type"):
        SpecialistAutoencoder(corruption_type="unknown_noise")


def test_build_specialist_variants():
    """Verify factory creation of homogeneous, lightweight, and tailored variants."""
    for corr in VALID_SPECIALISTS:
        # Homogeneous
        m_homo = build_specialist(corr, variant="homogeneous")
        assert len(m_homo.channels) == 4
        assert m_homo.bottleneck_dim == 256
        assert m_homo.use_residual is True

        # Lightweight
        m_lite = build_specialist(corr, variant="lightweight")
        assert len(m_lite.channels) == 3
        assert m_lite.bottleneck_dim == 128
        assert m_lite.use_residual is False

        # Tailored
        m_tail = build_specialist(corr, variant="tailored")
        assert m_tail.corruption_type == corr
        if corr == "salt_and_pepper":
            assert len(m_tail.channels) == 3
            assert m_tail.bottleneck_dim == 128
            assert m_tail.use_residual is True
        else:
            assert len(m_tail.channels) == 4
            assert m_tail.bottleneck_dim == 256


def test_specialist_forward_shape():
    """Verify forward restoration preserves dimensions and values in [0, 1]."""
    model = build_specialist("gaussian_blur", variant="homogeneous")
    model.eval()

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 128, 128)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_specialist_latent_encode():
    """Verify encode method returns expected spatial and channel dimensions."""
    m_4stage = build_specialist("occlusion", variant="homogeneous")
    x = torch.rand(2, 3, 128, 128)
    latent_4 = m_4stage.encode(x)
    assert latent_4.shape == (2, 256, 8, 8)

    m_3stage = build_specialist("salt_and_pepper", variant="lightweight")
    latent_3 = m_3stage.encode(x)
    assert latent_3.shape == (2, 128, 16, 16)


def test_specialist_gradient_backward():
    """Verify backpropagation through specialist with CombinedReconstructionLoss."""
    model = build_specialist("salt_and_pepper", variant="tailored")
    model.train()
    criterion = CombinedReconstructionLoss(alpha=0.84)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)

    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()

    # Check non-zero gradients in encoder and decoder
    assert model.encoder.stem[0].weight.grad is not None
    assert torch.any(model.encoder.stem[0].weight.grad != 0)
    optimizer.step()


def test_specialist_checkpoint_roundtrip(tmp_path):
    """Verify save_checkpoint and load_from_checkpoint restore identical weights and config."""
    ckpt_path = tmp_path / "test_specialist.pt"
    model = build_specialist("occlusion", variant="tailored")
    model.eval()
    x = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        orig_out = model(x)

    model.save_checkpoint(
        ckpt_path,
        epoch=3,
        val_loss=0.0821,
        metrics={"psnr": 22.45, "ssim": 0.6512},
    )

    restored_model, ckpt = SpecialistAutoencoder.load_from_checkpoint(ckpt_path)
    assert restored_model.corruption_type == "occlusion"
    assert restored_model.bottleneck_dim == 256
    assert ckpt["epoch"] == 3
    assert ckpt["val_loss"] == 0.0821
    assert ckpt["metrics"]["psnr"] == 22.45

    restored_model.eval()
    with torch.no_grad():
        restored_out = restored_model(x)

    torch.testing.assert_close(orig_out, restored_out, atol=1e-6, rtol=1e-6)
