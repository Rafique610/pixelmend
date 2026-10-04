"""Test suite for Task 2 Hard-Routed Restoration backend router."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.backend.main import app

REAL_IMAGE_PATH = Path("data/oxford-iiit-pet/images/Abyssinian_1.jpg")


def _get_clean_image_bytes() -> bytes:
    """Return bytes of a natural clean image for classifier evaluation."""
    if REAL_IMAGE_PATH.exists():
        return REAL_IMAGE_PATH.read_bytes()
    rng = np.random.default_rng(42)
    arr = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_dummy_image_bytes(
    fmt: str = "PNG", size: tuple = (128, 128), color: tuple = (110, 140, 190)
) -> bytes:
    """Helper to generate encoded image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestHardRoutedEndpoint:
    """Validate POST /api/v1/restore/hard-routed endpoint."""

    def test_restore_clean_identity_bypass(self, client):
        img_bytes = _get_clean_image_bytes()
        files = {"file": ("clean.jpg", img_bytes, "image/jpeg")}
        data = {"apply_corruption": "false"}

        response = client.post("/api/v1/restore/hard-routed", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["predicted_class"] == "clean"
        assert result["selected_expert"] == "Identity Bypass"
        assert result["inference_time"]["specialist_ms"] == 0.0
        assert result["inference_time"]["classifier_ms"] > 0.0
        assert result["inference_time"]["total_ms"] == result["inference_time"]["classifier_ms"]

        # Validate softmax probabilities sum to 1.0
        probs = result["class_probabilities"]
        assert len(probs) == 4
        assert "clean" in probs
        assert "salt_and_pepper" in probs
        assert "gaussian_blur" in probs
        assert "rectangular_occlusion" in probs
        total_p = sum(probs.values())
        assert abs(total_p - 1.0) < 1e-3

    def test_restore_gaussian_blur_routing(self, client):
        img_bytes = _make_dummy_image_bytes("JPEG")
        files = {"file": ("blur.jpg", img_bytes, "image/jpeg")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "gaussian_blur",
            "severity": "2",
        }

        response = client.post("/api/v1/restore/hard-routed", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["predicted_class"] == "gaussian_blur"
        assert result["selected_expert"] == "task2_specialist_blur.onnx"
        assert result["confidence"] > 0.5
        assert result["inference_time"]["specialist_ms"] > 0.0
        assert result["inference_time"]["total_ms"] > result["inference_time"]["specialist_ms"]
        assert result["restored_image"].startswith("data:image/png;base64,")

    def test_restore_salt_and_pepper_routing(self, client):
        img_bytes = _make_dummy_image_bytes("PNG")
        files = {"file": ("salt.png", img_bytes, "image/png")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "salt_and_pepper",
            "severity": "2",
        }

        response = client.post("/api/v1/restore/hard-routed", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["predicted_class"] == "salt_and_pepper"
        assert result["selected_expert"] == "task2_specialist_salt.onnx"

    def test_restore_occlusion_routing(self, client):
        img_bytes = _make_dummy_image_bytes("PNG")
        files = {"file": ("occl.png", img_bytes, "image/png")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "rectangular_occlusion",
            "severity": "2",
        }

        response = client.post("/api/v1/restore/hard-routed", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["predicted_class"] == "rectangular_occlusion"
        assert result["selected_expert"] == "task2_specialist_occlusion.onnx"

    def test_restore_manual_expert_override(self, client):
        img_bytes = _get_clean_image_bytes()
        files = {"file": ("clean.jpg", img_bytes, "image/jpeg")}
        # Even though image is clean, force routing to blur specialist
        data = {"apply_corruption": "false", "force_expert": "blur"}

        response = client.post("/api/v1/restore/hard-routed", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["predicted_class"] == "clean"
        assert result["selected_expert"] == "task2_specialist_blur.onnx"
        assert result["inference_time"]["specialist_ms"] > 0.0

    def test_restore_rejects_empty_file(self, client):
        files = {"file": ("empty.png", b"", "image/png")}
        response = client.post("/api/v1/restore/hard-routed", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
