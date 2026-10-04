"""
tests/test_task4_research.py
----------------------------
Unit tests verifying Task 4 Step 1 & 2 architectures and conditioning:
1. FS2K paired dataset loading, path resolution, and stratified splits.
2. Generator architectures (Vanilla U-Net, ResNet U-Net, Attention U-Net).
3. Style conditioning layers (Spatial Concat, FiLM, AdaIN, CIN).
4. Gradient flow and numerical stability.
"""

import io
import torch
import pytest

from src.task4.dataset import FS2KDataset
from src.task4.conditioning import (
    SpatialConcatConditioning,
    FiLMConditioning,
    AdaINConditioning,
    ConditionalInstanceNorm,
)
from src.task4.generator_variants import (
    VanillaUNetGenerator,
    ResNetUNetGenerator,
    AttentionUNetGenerator,
)


def test_fs2k_dataset_loading_and_stratification():
    train_ds = FS2KDataset(root_dir="data/fs2k", split="train", val_ratio=0.15, seed=42)
    val_ds = FS2KDataset(root_dir="data/fs2k", split="val", val_ratio=0.15, seed=42)
    test_ds = FS2KDataset(root_dir="data/fs2k", split="test", seed=42)

    assert len(train_ds) + len(val_ds) == 1058
    assert len(test_ds) == 1046
    assert abs(len(val_ds) / 1058 - 0.15) < 0.02

    # Check sample tensors
    sample = train_ds[0]
    assert sample["photo"].shape == (3, 128, 128)
    assert sample["sketch"].shape == (3, 128, 128)
    assert sample["style"] in (0, 1, 2)
    assert sample["photo"].min() >= -1.05 and sample["photo"].max() <= 1.05


@pytest.mark.parametrize("conditioning", ["film", "spatial", "adain"])
def test_vanilla_unet_forward_and_backward(conditioning):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaUNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        conditioning=conditioning,
        embed_dim=16,
    ).to(device)

    x = torch.randn(2, 3, 128, 128, device=device)
    s = torch.tensor([0, 1], dtype=torch.long, device=device)

    out = model(x, s)
    assert out.shape == (2, 3, 128, 128)
    assert out.min() >= -1.0 and out.max() <= 1.0

    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"


def test_resnet_unet_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        conditioning="film",
        embed_dim=16,
    ).to(device)

    x = torch.randn(2, 3, 128, 128, device=device)
    s = torch.tensor([1, 2], dtype=torch.long, device=device)

    out = model(x, s)
    assert out.shape == (2, 3, 128, 128)
    assert torch.all(torch.isfinite(out))


def test_attention_unet_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        conditioning="film",
        embed_dim=16,
    ).to(device)

    x = torch.randn(2, 3, 128, 128, device=device)
    s = torch.tensor([0, 2], dtype=torch.long, device=device)

    out = model(x, s)
    assert out.shape == (2, 3, 128, 128)
    assert torch.all(torch.isfinite(out))


def test_style_conditioning_layers():
    x = torch.randn(2, 64, 16, 16)
    emb = torch.randn(2, 16)
    style_idx = torch.tensor([0, 1], dtype=torch.long)

    # FiLM
    film = FiLMConditioning(in_channels=64, embed_dim=16)
    out_film = film(x, emb)
    assert out_film.shape == (2, 64, 16, 16)

    # AdaIN
    adain = AdaINConditioning(in_channels=64, embed_dim=16)
    out_adain = adain(x, emb)
    assert out_adain.shape == (2, 64, 16, 16)

    # Spatial Concat
    spatial = SpatialConcatConditioning(embed_dim=16)
    out_spatial = spatial(x, emb)
    assert out_spatial.shape == (2, 64 + 16, 16, 16)

    # CIN
    cin = ConditionalInstanceNorm(in_channels=64, num_styles=3)
    out_cin = cin(x, style_idx)
    assert out_cin.shape == (2, 64, 16, 16)


def test_generator_style_sensitivity():
    """Verify that different styles produce distinct synthesized outputs for identical input photo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaUNetGenerator(base_channels=32, conditioning="film").to(device)
    model.eval()

    photo = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        out_style0 = model(photo, torch.tensor([0], device=device))
        out_style1 = model(photo, torch.tensor([1], device=device))

    diff = torch.abs(out_style0 - out_style1).mean().item()
    assert diff > 0.0, "Style conditioning must produce distinct outputs across style indices."
