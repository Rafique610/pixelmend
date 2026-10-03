"""Unit tests for Task 1 training routine."""

from pathlib import Path
import tempfile
import torch

from src.shared.config import get_settings
from src.shared.losses import CombinedReconstructionLoss
from src.task1.autoencoder import UniversalAutoencoder
from src.task1.train import run_training, train_one_epoch, validate


def test_train_one_epoch_and_validate():
    """Verify single epoch training and validation step execution."""
    settings = get_settings()
    device = torch.device("cpu")

    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        channels=(16, 32),
        bottleneck_dim=32,
    ).to(device)

    criterion = CombinedReconstructionLoss(alpha=0.8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy dataloader
    dummy_corrupted = torch.rand(4, 3, 128, 128)
    dummy_clean = torch.rand(4, 3, 128, 128)
    dummy_labels = torch.zeros(4, dtype=torch.long)
    dataset = [(dummy_corrupted, dummy_clean, dummy_labels)]

    loss = train_one_epoch(model, dataset, optimizer, criterion, device)
    assert isinstance(loss, float)
    assert loss > 0.0

    metrics, s_corr, s_rest, s_clean = validate(model, dataset, criterion, device)
    assert "val_loss" in metrics
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "mae" in metrics
    assert s_corr.shape == (4, 3, 128, 128)
    assert s_rest.shape == (4, 3, 128, 128)
    assert s_clean.shape == (4, 3, 128, 128)


def test_run_training_quick_execution():
    """Verify run_training executes full pipeline and produces checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_quick_best.pth"
        result = run_training(
            epochs=1,
            batch_size=16,
            alpha=0.8,
            bottleneck_dim=32,
            channels=(16, 32),
            checkpoint_path=ckpt_path,
            run_name="pytest_smoke",
            quick=True,
        )

        assert ckpt_path.exists()
        assert "best_val_loss" in result
        assert result["best_val_loss"] > 0.0

    # Clean up test artifacts from results directory
    test_json = Path("results/task1/metrics/pytest_smoke_history.json")
    test_fig = Path("results/task1/visualizations/pytest_smoke_training_curves.png")
    if test_json.exists():
        test_json.unlink()
    if test_fig.exists():
        test_fig.unlink()
