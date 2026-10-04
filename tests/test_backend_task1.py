"""Test suite for Task 1 Universal Restoration backend router and corruption services."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.backend.main import app
from src.app.backend.services.corruption import apply_synthetic_corruption


def _make_dummy_image_bytes(
    fmt: str = "PNG", size: tuple = (128, 128), color: tuple = (100, 150, 200)
) -> bytes:
    """Helper to generate encoded image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestSyntheticCorruptionService:
    """Validate corruption generation matching assignment benchmark parameters."""

    def test_apply_clean(self):
        arr = np.ones((1, 3, 128, 128), dtype=np.float32) * 0.5
        corrupted, meta = apply_synthetic_corruption(arr, "clean")
        assert meta["type"] == "clean"
        np.testing.assert_allclose(corrupted, arr)

    def test_apply_salt_and_pepper_severities(self):
        arr = np.ones((1, 3, 128, 128), dtype=np.float32) * 0.5
        for sev, expected_p in [(1, 0.03), (2, 0.08), (3, 0.15)]:
            corrupted, meta = apply_synthetic_corruption(
                arr, "salt_and_pepper", severity=sev, seed=42
            )
            assert meta["type"] == "salt_and_pepper"
            assert meta["p"] == expected_p
            # Values should contain 0.0 and/or 1.0 noise pixels
            assert np.any(corrupted == 0.0) or np.any(corrupted == 1.0)

    def test_apply_gaussian_blur_severities(self):
        arr = np.random.uniform(0.0, 1.0, (1, 3, 128, 128)).astype(np.float32)
        for sev, (k, sig) in [(1, (3, 0.7)), (2, (5, 1.5)), (3, (7, 2.5))]:
            corrupted, meta = apply_synthetic_corruption(arr, "gaussian_blur", severity=sev)
            assert meta["type"] == "gaussian_blur"
            assert meta["kernel_size"] == k
            assert meta["sigma"] == sig
            assert corrupted.shape == (1, 3, 128, 128)

    def test_apply_occlusion_severities(self):
        arr = np.ones((1, 3, 128, 128), dtype=np.float32)
        for sev, expected_boxes in [(1, 1), (2, 2), (3, 3)]:
            corrupted, meta = apply_synthetic_corruption(
                arr, "rectangular_occlusion", severity=sev, seed=42
            )
            assert meta["type"] == "rectangular_occlusion"
            assert meta["num_boxes"] == expected_boxes
            assert meta["actual_coverage"] > 0.0
            # Some pixels should be black (0.0)
            assert np.any(corrupted == 0.0)

    def test_unsupported_corruption_raises_value_error(self):
        arr = np.ones((1, 3, 128, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported corruption type"):
            apply_synthetic_corruption(arr, "unknown_type")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestUniversalRestoreEndpoint:
    """Validate POST /api/v1/restore/universal endpoint."""

    def test_restore_direct_upload(self, client):
        img_bytes = _make_dummy_image_bytes("PNG")
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {"apply_corruption": "false"}

        response = client.post("/api/v1/restore/universal", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["original_image"].startswith("data:image/png;base64,")
        assert result["corrupted_image"].startswith("data:image/png;base64,")
        assert result["restored_image"].startswith("data:image/png;base64,")
        assert result["error_map"].startswith("data:image/png;base64,")
        assert result["clean_error_map"] is None
        assert result["corruption_applied"] is None
        assert result["inference_time_ms"] > 0.0
        assert result["model_name"] == "task1_universal_ae.onnx"

    def test_restore_with_synthetic_gaussian_blur(self, client):
        img_bytes = _make_dummy_image_bytes("JPEG")
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "gaussian_blur",
            "severity": "2",
        }

        response = client.post("/api/v1/restore/universal", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["corruption_applied"] is not None
        assert result["corruption_applied"]["type"] == "gaussian_blur"
        assert result["corruption_applied"]["kernel_size"] == 5
        assert result["clean_error_map"] is not None
        assert result["clean_error_map"].startswith("data:image/png;base64,")

    def test_restore_with_synthetic_salt_and_pepper(self, client):
        img_bytes = _make_dummy_image_bytes("PNG")
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "salt_and_pepper",
            "severity": "1",
        }

        response = client.post("/api/v1/restore/universal", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["corruption_applied"]["type"] == "salt_and_pepper"
        assert result["corruption_applied"]["p"] == 0.03

    def test_restore_with_synthetic_occlusion(self, client):
        img_bytes = _make_dummy_image_bytes("PNG")
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "rectangular_occlusion",
            "severity": "3",
        }

        response = client.post("/api/v1/restore/universal", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["corruption_applied"]["type"] == "rectangular_occlusion"
        assert result["corruption_applied"]["num_boxes"] == 3

    def test_restore_rejects_empty_file(self, client):
        files = {"file": ("empty.png", b"", "image/png")}
        response = client.post("/api/v1/restore/universal", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_restore_rejects_invalid_corruption_type(self, client):
        img_bytes = _make_dummy_image_bytes("PNG")
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {
            "apply_corruption": "true",
            "corruption_type": "invalid_degradation",
        }

        response = client.post("/api/v1/restore/universal", files=files, data=data)
        assert response.status_code == 400
        assert "unsupported corruption type" in response.json()["detail"].lower()
