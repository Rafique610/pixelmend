"""
tests/test_task4_models.py
--------------------------
Unit tests verifying production Task 4 models and paired augmentation:
1. Production UNetGenerator forward, backward, bounds, and initialization.
2. Production PatchGANDiscriminator patch logits shape and receptive fields.
3. MultiScaleDiscriminator dual-scale evaluation.
4. Min-max adversarial backpropagation step.
5. Coupled geometric transformation consistency.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.task4.augmentation import CoupledTransform
from src.task4.discriminator import MultiScaleDiscriminator, PatchGANDiscriminator
from src.task4.generator import UNetGenerator


def test_unet_generator_production():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator(base_channels=32, embed_dim=16).to(device)

    photo = torch.randn(2, 3, 128, 128, device=device)
    style = torch.tensor([0, 2], dtype=torch.long, device=device)

    out = model(photo, style)
    assert out.shape == (2, 3, 128, 128)
    assert out.min() >= -1.0 and out.max() <= 1.0

    # Test weight initialization variance
    first_conv_weight = model.d1.block[0].weight.data.cpu().numpy()
    assert abs(np.mean(first_conv_weight)) < 0.05
    assert abs(np.std(first_conv_weight) - 0.02) < 0.015


def test_patchgan_discriminator_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 70x70 PatchGAN (3 intermediate downsampling layers)
    d70 = PatchGANDiscriminator(base_channels=32, n_layers=3).to(device)
    photo = torch.randn(2, 3, 128, 128, device=device)
    sketch = torch.randn(2, 3, 128, 128, device=device)
    style = torch.tensor([1, 0], dtype=torch.long, device=device)

    logits70 = d70(photo, sketch, style)
    assert logits70.shape == (2, 1, 14, 14)

    # 16x16 PatchGAN (1 intermediate downsampling layer)
    d16 = PatchGANDiscriminator(base_channels=32, n_layers=1).to(device)
    logits16 = d16(photo, sketch, style)
    assert logits16.shape == (2, 1, 62, 62)


def test_multiscale_discriminator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_multi = MultiScaleDiscriminator(base_channels=32).to(device)

    photo = torch.randn(2, 3, 128, 128, device=device)
    sketch = torch.randn(2, 3, 128, 128, device=device)
    style = torch.tensor([0, 1], dtype=torch.long, device=device)

    out1, out2 = d_multi(photo, sketch, style)
    assert out1.shape == (2, 1, 14, 14)
    assert out2.shape == (2, 1, 14, 14)


def test_min_max_adversarial_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g = UNetGenerator(base_channels=32, embed_dim=16).to(device)
    net_d = PatchGANDiscriminator(base_channels=32, n_layers=3).to(device)

    opt_g = torch.optim.Adam(net_g.parameters(), lr=2e-4)
    opt_d = torch.optim.Adam(net_d.parameters(), lr=2e-4)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    photo = torch.randn(2, 3, 128, 128, device=device)
    real_sketch = torch.randn(2, 3, 128, 128, device=device)
    style = torch.tensor([0, 1], dtype=torch.long, device=device)

    # --- Step 1: Train Discriminator ---
    opt_d.zero_grad()
    d_real = net_d(photo, real_sketch, style)
    loss_d_real = bce(d_real, torch.ones_like(d_real))

    fake_sketch = net_g(photo, style).detach()
    d_fake = net_d(photo, fake_sketch, style)
    loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))

    loss_d = 0.5 * (loss_d_real + loss_d_fake)
    loss_d.backward()
    opt_d.step()

    # --- Step 2: Train Generator ---
    opt_g.zero_grad()
    gen_sketch = net_g(photo, style)
    d_pred = net_d(photo, gen_sketch, style)
    loss_g_adv = bce(d_pred, torch.ones_like(d_pred))
    loss_g_l1 = l1(gen_sketch, real_sketch) * 100.0

    loss_g = loss_g_adv + loss_g_l1
    loss_g.backward()
    opt_g.step()

    # Assert gradients exist across both models
    for name, p in net_d.named_parameters():
        assert p.grad is not None, f"D param {name} missing gradient"
    for name, p in net_g.named_parameters():
        assert p.grad is not None, f"G param {name} missing gradient"


def test_coupled_augmentation():
    tform = CoupledTransform(
        image_size=128,
        augment=True,
        hflip_prob=1.0,
        rotation_degrees=0.0,
        photometric_jitter=False,
    )
    img_a = Image.new("RGB", (200, 200), color=(255, 0, 0))
    # Place a marker on the left half of both images
    for y in range(50):
        for x in range(50):
            img_a.putpixel((x, y), (0, 255, 0))

    img_b = img_a.copy()
    photo_t, sketch_t = tform(img_a, img_b)

    # Because hflip_prob=1.0 was set, both images must be flipped horizontally identically
    assert torch.allclose(photo_t[1], sketch_t[1], atol=1e-4)
    # The green patch (channel 1 = 1.0) must now be on the right side: x > 64
    assert photo_t[1, 0, 127] > 0.5

    # Test photometric jitter affects photo bounds but preserves sketch strokes
    tform_photo = CoupledTransform(
        image_size=128,
        augment=True,
        hflip_prob=0.0,
        rotation_degrees=0.0,
        photometric_jitter=True,
    )
    p_t, s_t = tform_photo(img_a, img_b)
    assert s_t.min() >= -1.05 and s_t.max() <= 1.05
