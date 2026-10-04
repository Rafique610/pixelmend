"""
tests/test_task4_train.py
-------------------------
Unit tests verifying GAN training utilities:
1. Visual progression grid generation and dimension integrity.
2. Generator validation evaluation on paired batches.
"""

from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.task4.train import create_visual_grid, evaluate_generator
from src.task4.generator import UNetGenerator


def test_create_visual_grid(tmp_path):
    photos = torch.randn(6, 3, 128, 128)
    reals = torch.randn(6, 3, 128, 128)
    fakes = torch.randn(6, 3, 128, 128)

    save_path = tmp_path / "test_grid.png"
    create_visual_grid(photos, reals, fakes, save_path)

    assert save_path.exists()
    img = Image.open(save_path)
    # 3 columns (Photo, Real, Fake) x 6 rows + padding
    assert img.width > 300 and img.height > 600


def test_evaluate_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g = UNetGenerator(base_channels=32, embed_dim=16).to(device)

    # Dummy dataset
    photos = torch.randn(4, 3, 128, 128)
    sketches = torch.randn(4, 3, 128, 128)
    styles = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    class DummyDS:
        def __len__(self): return 4
        def __getitem__(self, i):
            return {"photo": photos[i], "sketch": sketches[i], "style": styles[i]}

    loader = DataLoader(DummyDS(), batch_size=2)
    metrics = evaluate_generator(net_g, loader, device)

    assert "val_l1" in metrics
    assert "val_psnr" in metrics
    assert "val_ssim" in metrics
    assert metrics["val_l1"] > 0.0
