"""Test suite for Task 3 Soft Mixture-of-Experts (MoE) backend router."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.backend.main import app

REAL_IMAGE_PATH = Path("data/oxford-iiit-pet/images/Abyssinian_1.jpg")


def _get_test_image_bytes() -> bytes:
    """Return bytes of a natural test image."""
    if REAL_IMAGE_PATH.exists():
        return REAL_IMAGE_PATH.read_bytes()
    rng = np.random.default_rng(42)
    arr = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestSoftMoEEndpoint:
    """Validate POST /api/v1/restore/soft-moe endpoint."""

    def test_restore_clean_image(self, client):
        img_bytes = _get_test_image_bytes()
        files = {"file": ("clean.jpg", img_bytes, "image/jpeg")}
        data = {"apply_corruption": "false"}

        response = client.post("/api/v1/restore/soft-moe", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["original_image"].startswith("data:image/png;base64,")
        assert result["corrupted_image"].startswith("data:image/png;base64,")
        assert result["restored_image"].startswith("data:image/png;base64,")
        assert result["error_map"].startswith("data:image/png;base64,")
        assert result["clean_error_map"] is None
        assert result["corruption_applied"] is None
        assert result["inference_time_ms"] > 0.0

        weights = result["routing_weights"]
        assert len(weights) == 4
        assert "identity_clean" in weights
        assert "expert_salt" in weights
        assert "expert_blur" in weights
        assert "expert_occlusion" in weights

        # Weights sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-3

        # Clean image has identity as dominant expert
        assert result["dominant_expert"] == "identity_clean"
        assert result["dominant_weight"] == max(weights.values())
        assert result["entropy"] > 0.0

    def test_restore_with_synthetic_salt_and_pepper(self, client):
        img_bytes = _get_test_image_bytes()
        files = {"file": ("salt.jpg", img_bytes, "image/jpeg")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "salt_and_pepper",
            "severity": "3",
        }

        response = client.post("/api/v1/restore/soft-moe", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["corruption_applied"] is not None
        assert result["corruption_applied"]["type"] == "salt_and_pepper"
        assert result["clean_error_map"] is not None
        assert result["clean_error_map"].startswith("data:image/png;base64,")

        weights = result["routing_weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-3

        # At severe noise, expert_salt becomes dominant
        assert result["dominant_expert"] == "expert_salt"
        assert weights["expert_salt"] > 0.50

    def test_restore_with_synthetic_gaussian_blur(self, client):
        img_bytes = _get_test_image_bytes()
        files = {"file": ("blur.jpg", img_bytes, "image/jpeg")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "gaussian_blur",
            "severity": "2",
        }

        response = client.post("/api/v1/restore/soft-moe", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["corruption_applied"]["type"] == "gaussian_blur"
        weights = result["routing_weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-3
        # Blur expert receives significant weight allocation
        assert weights["expert_blur"] > 0.20
        assert result["dominant_weight"] == max(weights.values())

    def test_restore_with_synthetic_occlusion(self, client):
        img_bytes = _get_test_image_bytes()
        files = {"file": ("occl.jpg", img_bytes, "image/jpeg")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "rectangular_occlusion",
            "severity": "2",
        }

        response = client.post("/api/v1/restore/soft-moe", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["corruption_applied"]["type"] == "rectangular_occlusion"
        weights = result["routing_weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-3
        # Occlusion expert receives significant allocation
        assert weights["expert_occlusion"] > 0.20
        assert result["dominant_weight"] == max(weights.values())

    def test_restore_rejects_empty_file(self, client):
        files = {"file": ("empty.png", b"", "image/png")}
        response = client.post("/api/v1/restore/soft-moe", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_restore_rejects_invalid_corruption_type(self, client):
        img_bytes = _get_test_image_bytes()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "invalid_distortion",
        }
        response = client.post("/api/v1/restore/soft-moe", files=files, data=data)
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()
