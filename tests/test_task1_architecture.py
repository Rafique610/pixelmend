"""Unit tests for Task 1 Universal Autoencoder architecture."""

import tempfile
from pathlib import Path
import pytest
import torch

from src.task1.autoencoder import UniversalAutoencoder
from src.task1.decoder import Decoder
from src.task1.encoder import Encoder


def test_encoder_output_shape():
    """Verify encoder produces expected spatial resolution and bottleneck channels."""
    model = Encoder(in_channels=3, channels=(32, 64, 128, 256), bottleneck_dim=256)
    x = torch.randn(2, 3, 128, 128)
    latent, skips = model(x)

    assert skips is None
    assert latent.shape == (2, 256, 8, 8)


def test_encoder_with_skips():
    """Verify encoder extracts downsampled skip tensors when enabled."""
    model = Encoder(in_channels=3, channels=(32, 64, 128, 256), bottleneck_dim=256, use_skips=True)
    x = torch.randn(2, 3, 128, 128)
    latent, skips = model(x)

    assert skips is not None
    assert len(skips) == 4  # Stages 0, 1, 2, 3
    assert latent.shape == (2, 256, 8, 8)


def test_decoder_output_shape():
    """Verify decoder upsamples bottleneck features back to 128x128 RGB in [0, 1]."""
    decoder = Decoder(out_channels=3, channels=(32, 64, 128, 256), bottleneck_dim=256)
    latent = torch.randn(2, 256, 8, 8)
    out = decoder(latent)

    assert out.shape == (2, 3, 128, 128)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_universal_autoencoder_forward():
    """Verify end-to-end autoencoder forward pass."""
    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256),
        bottleneck_dim=256,
    )
    x = torch.rand(2, 3, 128, 128)
    out = model(x)

    assert out.shape == x.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_universal_autoencoder_gradient_flow():
    """Ensure gradients flow to all trainable parameters without NaNs."""
    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128),
        bottleneck_dim=128,
    )
    x = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)

    out = model(x)
    loss = torch.nn.functional.l1_loss(out, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("use_skips", [False, True])
@pytest.mark.parametrize("upsample_mode", ["transpose", "bilinear"])
def test_universal_autoencoder_variants(use_residual: bool, use_skips: bool, upsample_mode: str):
    """Test model forward pass across all architectural toggle variants."""
    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128),
        bottleneck_dim=64,
        use_residual=use_residual,
        use_skips=use_skips,
        upsample_mode=upsample_mode,
    )
    x = torch.rand(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 3, 128, 128)


def test_compression_ratio_and_parameter_count():
    """Verify parameter counting and compression ratio metrics."""
    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256),
        bottleneck_dim=256,
    )
    param_count = model.count_parameters()
    assert param_count > 1_000_000

    ratio = model.compression_ratio((3, 128, 128))
    # Input elements = 3 * 128 * 128 = 49152
    # Bottleneck elements = 256 * 8 * 8 = 16384
    # Expected ratio = 49152 / 16384 = 3.0
    assert pytest.approx(ratio, rel=1e-2) == 3.0


def test_checkpoint_save_and_load():
    """Verify model can be cleanly serialized and reloaded from checkpoint."""
    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128),
        bottleneck_dim=64,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pth"
        model.save_checkpoint(ckpt_path, epoch=5, val_loss=0.042)

        reloaded_model, ckpt = UniversalAutoencoder.load_from_checkpoint(ckpt_path)
        assert ckpt["epoch"] == 5
        assert pytest.approx(ckpt["val_loss"]) == 0.042

        x = torch.rand(1, 3, 128, 128)
        model.eval()
        reloaded_model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = reloaded_model(x)
        assert torch.allclose(out1, out2, atol=1e-5)
