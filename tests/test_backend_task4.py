"""Test suite for Task 4 Face-to-Sketch synthesis backend router."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.backend.main import app

FS2K_SAMPLE_PATH = Path("data/fs2k/photo/photo1/image0001.jpg")
PET_SAMPLE_PATH = Path("data/oxford-iiit-pet/images/Abyssinian_1.jpg")


def _get_test_face_bytes() -> bytes:
    """Return real face photo bytes or fallback test image."""
    if FS2K_SAMPLE_PATH.exists():
        return FS2K_SAMPLE_PATH.read_bytes()
    if PET_SAMPLE_PATH.exists():
        return PET_SAMPLE_PATH.read_bytes()
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


class TestFaceToSketchEndpoint:
    """Validate POST /api/v1/sketch/generate endpoint."""

    def test_generate_sketch_style_1(self, client):
        img_bytes = _get_test_face_bytes()
        files = {"file": ("face.jpg", img_bytes, "image/jpeg")}
        data = {"style": "1"}

        response = client.post("/api/v1/sketch/generate", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["original_image"].startswith("data:image/png;base64,")
        assert result["sketch_image"].startswith("data:image/png;base64,")
        assert result["selected_style"] == 1
        assert "Style 1" in result["style_description"]
        assert result["inference_time_ms"] > 0.0

    def test_generate_sketch_style_2(self, client):
        img_bytes = _get_test_face_bytes()
        files = {"file": ("face.jpg", img_bytes, "image/jpeg")}
        data = {"style": "2"}

        response = client.post("/api/v1/sketch/generate", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["selected_style"] == 2
        assert "Style 2" in result["style_description"]
        assert result["sketch_image"].startswith("data:image/png;base64,")

    def test_generate_sketch_style_3(self, client):
        img_bytes = _get_test_face_bytes()
        files = {"file": ("face.jpg", img_bytes, "image/jpeg")}
        data = {"style": "3"}

        response = client.post("/api/v1/sketch/generate", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["selected_style"] == 3
        assert "Style 3" in result["style_description"]
        assert result["sketch_image"].startswith("data:image/png;base64,")

    def test_accepts_zero_indexed_style(self, client):
        img_bytes = _get_test_face_bytes()
        files = {"file": ("face.jpg", img_bytes, "image/jpeg")}
        data = {"style": "0"}

        response = client.post("/api/v1/sketch/generate", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["selected_style"] == 1
        assert "Style 1" in result["style_description"]

    def test_rejects_invalid_style(self, client):
        img_bytes = _get_test_face_bytes()
        files = {"file": ("face.jpg", img_bytes, "image/jpeg")}
        data = {"style": "99"}

        response = client.post("/api/v1/sketch/generate", files=files, data=data)
        assert response.status_code == 400
        assert "invalid style" in response.json()["detail"].lower()

    def test_rejects_empty_file(self, client):
        files = {"file": ("empty.png", b"", "image/png")}
        data = {"style": "1"}

        response = client.post("/api/v1/sketch/generate", files=files, data=data)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
