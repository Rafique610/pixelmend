"""
tests/test_corruptions.py
-------------------------
Unit tests for corruption functions, deterministic manifests,
and CorruptedPetDataset wrapper.
"""

import torch

from src.shared.corruptions import (
    CorruptionType,
    apply_clean,
    apply_gaussian_blur,
    apply_occlusion,
    apply_salt_and_pepper,
    compute_occlusion_coverage,
    sample_occlusion_boxes,
    sample_random_corruption,
)
from src.shared.datasets.corrupted import CorruptedPetDataset, get_corrupted_pet_dataloader
from src.shared.manifests import (
    load_test_manifest,
    load_val_manifest,
)


def test_clean_identity():
    x = torch.rand((3, 128, 128))
    out = apply_clean(x)
    assert torch.equal(x, out)
    assert out.shape == x.shape
    # Ensure clone/isolation
    out[0, 0, 0] = 999.0
    assert x[0, 0, 0] != 999.0


def test_salt_and_pepper_statistics():
    torch.manual_seed(42)
    # Uniform gray image (0.5) to easily distinguish salt (1.0) and pepper (0.0)
    x = torch.full((3, 200, 200), 0.5)
    p = 0.10
    out = apply_salt_and_pepper(x, p=p, channel_independent=False)

    # Check pixel values
    is_pepper = (out == 0.0).all(dim=0)
    is_salt = (out == 1.0).all(dim=0)
    is_unmodified = (out == 0.5).all(dim=0)

    total_pixels = 200 * 200
    num_pepper = is_pepper.sum().item()
    num_salt = is_salt.sum().item()
    num_corrupted = num_pepper + num_salt

    assert (is_pepper | is_salt | is_unmodified).all()
    # Check that total noise proportion is close to p=0.10 (within +/- 0.02)
    observed_p = num_corrupted / total_pixels
    assert abs(observed_p - p) < 0.02
    # Check approximately equal split between salt and pepper
    assert abs(num_pepper - num_salt) < (total_pixels * 0.02)


def test_gaussian_blur_attenuation():
    # High frequency checkerboard pattern
    x = torch.zeros((3, 128, 128))
    x[:, ::2, ::2] = 1.0
    x[:, 1::2, 1::2] = 1.0

    blurred = apply_gaussian_blur(x, kernel_size=5, sigma=1.5)
    assert blurred.shape == x.shape
    assert 0.0 <= blurred.min() and blurred.max() <= 1.0

    # Gaussian blur must reduce spatial variance / attenuate high frequencies
    var_original = torch.var(x).item()
    var_blurred = torch.var(blurred).item()
    assert var_blurred < var_original

    # Mean intensity should be conserved within boundary reflection effects (<5% difference)
    mean_original = torch.mean(x).item()
    mean_blurred = torch.mean(blurred).item()
    assert abs(mean_original - mean_blurred) < 0.05


def test_occlusion_area_and_fill():
    x = torch.ones((3, 128, 128))
    target_ratio = 0.20
    boxes = sample_occlusion_boxes(height=128, width=128, num_boxes=2, target_ratio=target_ratio)

    assert len(boxes) == 2
    for y1, x1, y2, x2 in boxes:
        assert 0 <= y1 < y2 <= 128
        assert 0 <= x1 < x2 <= 128

    coverage = compute_occlusion_coverage(boxes, height=128, width=128)
    # Given aspect ratio clamping and rounding, coverage should be within [0.12, 0.28]
    assert 0.10 <= coverage <= 0.30

    occluded = apply_occlusion(x, boxes=boxes, fill_value=0.0)
    # Check that pixels in boxes are zeroed
    for y1, x1, y2, x2 in boxes:
        assert (occluded[:, y1:y2, x1:x2] == 0.0).all()


def test_sample_random_corruption():
    for _ in range(20):
        c_type, label, params = sample_random_corruption(image_size=128)
        assert c_type in list(CorruptionType)
        assert label in [0, 1, 2, 3]

        if c_type == CorruptionType.SALT_AND_PEPPER:
            assert 0.02 <= params["p"] <= 0.15
        elif c_type == CorruptionType.GAUSSIAN_BLUR:
            assert params["kernel_size"] in [3, 5, 7]
            assert 0.5 <= params["sigma"] <= 2.5
        elif c_type == CorruptionType.OCCLUSION:
            assert 1 <= len(params["boxes"]) <= 3
            assert 0.10 <= params["target_ratio"] <= 0.35


def test_manifest_validation_distribution():
    val_entries = load_val_manifest()
    assert len(val_entries) == 736

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for e in val_entries:
        counts[e["corruption_label"]] += 1

    # Exactly 25% across all 4 classes: 184 each
    assert counts[0] == 184
    assert counts[1] == 184
    assert counts[2] == 184
    assert counts[3] == 184


def test_manifest_test_benchmark_count():
    test_manifest = load_test_manifest()
    meta = test_manifest["metadata"]
    assert meta["num_test_images"] == 3669
    assert meta["variations_per_image"] == 10
    assert meta["total_evaluation_instances"] == 36690
    assert len(test_manifest["entries"]) == 36690


def test_corrupted_pet_dataset_train_and_val():
    # Test training mode (dynamic corruption)
    train_ds = CorruptedPetDataset(split="train", return_meta=True)
    assert len(train_ds) == 2944

    corr_tensor, clean_tensor, label, meta = train_ds[0]
    assert corr_tensor.shape == (3, 128, 128)
    assert clean_tensor.shape == (3, 128, 128)
    assert label in [0, 1, 2, 3]
    assert 0.0 <= corr_tensor.min() and corr_tensor.max() <= 1.0
    assert 0.0 <= clean_tensor.min() and clean_tensor.max() <= 1.0
    assert meta["split"] == "train"

    # Test validation mode (deterministic manifest)
    val_ds = CorruptedPetDataset(split="val", return_meta=True)
    assert len(val_ds) == 736

    corr_v, clean_v, label_v, meta_v = val_ds[0]
    assert corr_v.shape == (3, 128, 128)
    assert clean_v.shape == (3, 128, 128)
    assert label_v in [0, 1, 2, 3]
    assert meta_v["split"] == "val"
    assert "val_id" in meta_v


def test_corrupted_pet_dataloader_batch():
    loader = get_corrupted_pet_dataloader(
        split="val",
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    corr_batch, clean_batch, labels = next(iter(loader))

    assert corr_batch.shape == (8, 3, 128, 128)
    assert clean_batch.shape == (8, 3, 128, 128)
    assert labels.shape == (8,)
    assert labels.dtype == torch.int64
    assert 0.0 <= corr_batch.min() and corr_batch.max() <= 1.0
    assert 0.0 <= clean_batch.min() and clean_batch.max() <= 1.0
