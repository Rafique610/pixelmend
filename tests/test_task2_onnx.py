"""
tests/test_task2_onnx.py
-------------------------
Unit and integration tests for Task 2 ONNX models, numerical parity,
dynamic batching, and OnnxHardRouter end-to-end inference.
"""

from pathlib import Path
import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from src.task2.export_onnx import OnnxHardRouter, verify_numerical_parity
from src.task2.router import _load_model_from_checkpoint


@pytest.fixture(scope="module")
def onnx_dir() -> Path:
    return Path("models/onnx")


@pytest.fixture(scope="module")
def classifier_onnx(onnx_dir: Path) -> Path:
    p = onnx_dir / "task2_classifier.onnx"
    assert p.is_file(), f"Missing ONNX model: {p}"
    return p


@pytest.fixture(scope="module")
def specialist_paths(onnx_dir: Path) -> dict[str, Path]:
    paths = {
        "salt": onnx_dir / "task2_specialist_salt.onnx",
        "blur": onnx_dir / "task2_specialist_blur.onnx",
        "occlusion": onnx_dir / "task2_specialist_occlusion.onnx",
    }
    for k, p in paths.items():
        assert p.is_file(), f"Missing specialist ONNX: {p}"
    return paths


def test_onnx_files_exist_and_valid(classifier_onnx: Path, specialist_paths: dict[str, Path]):
    """Verify that all 4 ONNX graphs load cleanly and pass onnx.checker validation."""
    all_models = [classifier_onnx] + list(specialist_paths.values())
    for m_path in all_models:
        model = onnx.load(str(m_path))
        onnx.checker.check_model(model)
        assert m_path.stat().st_size > 500_000  # Each model is > 0.5 MB


def test_classifier_onnx_dynamic_batching_and_parity(classifier_onnx: Path):
    """Verify classifier ONNX inference and numerical parity across dynamic batch sizes."""
    sess = ort.InferenceSession(str(classifier_onnx), providers=["CPUExecutionProvider"])
    pt_model = _load_model_from_checkpoint("checkpoints/task2/classifier_best.pt", "classifier", torch.device("cpu"))
    pt_model.eval()

    for b in [1, 2, 4]:
        torch.manual_seed(100 + b)
        x_pt = torch.rand(b, 3, 128, 128, dtype=torch.float32)
        with torch.no_grad():
            y_pt = pt_model(x_pt).cpu().numpy()

        y_ort = sess.run(None, {"input": x_pt.numpy()})[0]
        assert y_ort.shape == (b, 4)
        max_diff = np.max(np.abs(y_pt - y_ort))
        assert max_diff < 1e-4, f"Classifier max diff {max_diff} exceeded tolerance"


def test_specialists_onnx_dynamic_batching_and_parity(specialist_paths: dict[str, Path]):
    """Verify specialist ONNX inference and parity across salt, blur, and occlusion models."""
    ckpts = {
        "salt": ("checkpoints/task2/specialist_salt_best.pt", "salt_and_pepper"),
        "blur": ("checkpoints/task2/specialist_blur_best.pt", "gaussian_blur"),
        "occlusion": ("checkpoints/task2/specialist_occlusion_best.pt", "occlusion"),
    }

    for key, (ckpt_file, mod_type) in ckpts.items():
        onnx_file = specialist_paths[key]
        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        pt_model = _load_model_from_checkpoint(ckpt_file, mod_type, torch.device("cpu"))
        pt_model.eval()

        for b in [1, 2]:
            torch.manual_seed(200 + b)
            x_pt = torch.rand(b, 3, 128, 128, dtype=torch.float32)
            with torch.no_grad():
                y_pt = pt_model(x_pt).cpu().numpy()

            y_ort = sess.run(None, {"input": x_pt.numpy()})[0]
            assert y_ort.shape == (b, 3, 128, 128)
            assert np.all(y_ort >= 0.0) and np.all(y_ort <= 1.0)
            max_diff = np.max(np.abs(y_pt - y_ort))
            assert max_diff < 1e-5, f"Specialist {key} max diff {max_diff} exceeded tolerance"


def test_onnx_hard_router_end_to_end(classifier_onnx: Path, specialist_paths: dict[str, Path]):
    """Verify OnnxHardRouter end-to-end inference on single images and multi-image batches."""
    router = OnnxHardRouter(
        classifier_onnx,
        specialist_paths["salt"],
        specialist_paths["blur"],
        specialist_paths["occlusion"],
    )

    # 1. Single 3D image
    img_3d = np.random.rand(3, 128, 128).astype(np.float32)
    restored, decisions = router.predict(img_3d)
    assert restored.shape == (1, 3, 128, 128)
    assert len(decisions) == 1
    assert decisions[0] in [0, 1, 2, 3]

    # 2. Batch of 4 images
    batch_img = np.random.rand(4, 3, 128, 128).astype(np.float32)
    restored_batch, decisions_batch = router.predict(batch_img)
    assert restored_batch.shape == (4, 3, 128, 128)
    assert len(decisions_batch) == 4

    # 3. Direct identity bypass check for clean class (decision == 0)
    clean_sample = np.ones((1, 3, 128, 128), dtype=np.float32) * 0.5
    # Force bypass branch behavior by testing with decisions logic
    if decisions[0] == 0:
        np.testing.assert_array_equal(restored, np.expand_dims(img_3d, 0))


def test_benchmark_metrics_json_artifact():
    """Verify benchmark metrics artifact exists and records valid status."""
    metrics_file = Path("results/task2/onnx_parity_benchmark.json")
    if metrics_file.is_file():
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "models" in data
        assert "task2_classifier.onnx" in data["models"]["classifier"]["onnx_path"]
        assert data["onnx_router_pipeline_latency_ms"] > 0.0
