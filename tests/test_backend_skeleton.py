"""Test suite for FastAPI backend skeleton, configuration, preprocessing, and health check."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.backend.config import Settings
from src.app.backend.main import app
from src.app.backend.services.inference import ModelSessionManager
from src.app.backend.services.preprocessing import (
    ImageValidationError,
    base64_to_pil,
    compute_error_map_base64,
    pil_to_base64_data_url,
    postprocess_nchw_to_pil,
    preprocess_image_to_nchw,
    validate_image_bytes,
)


def _make_dummy_image_bytes(
    fmt: str = "PNG", size: tuple = (64, 64), color: tuple = (120, 150, 180)
) -> bytes:
    """Helper to generate encoded image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestBackendConfig:
    """Validate Pydantic Settings and environment prefix constraints."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.port == 8000
        assert "http://localhost:3000" in settings.cors_origins
        assert settings.models_dir == "models/onnx"
        assert settings.task1_model_name == "task1_universal_ae.onnx"
        assert settings.task3_moe_name == "task3_soft_moe.onnx"
        assert settings.task4_generator_name == "task4_generator.onnx"

    def test_env_prefix_override(self, monkeypatch):
        monkeypatch.setenv("GENAI_PORT", "9000")
        monkeypatch.setenv("GENAI_DEBUG", "true")
        custom_settings = Settings()
        assert custom_settings.port == 9000
        assert custom_settings.debug is True


class TestImagePreprocessing:
    """Validate image decoding, dimension checks, normalization, and base64 conversions."""

    def test_validate_valid_image(self):
        raw = _make_dummy_image_bytes("JPEG")
        img = validate_image_bytes(raw)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_validate_empty_bytes_raises(self):
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image_bytes(b"")

    def test_validate_corrupt_bytes_raises(self):
        with pytest.raises(ImageValidationError, match="Invalid or corrupted"):
            validate_image_bytes(b"NOT_AN_IMAGE_PAYLOAD_12345")

    def test_validate_oversized_bytes_raises(self):
        # 2 MB limit for test
        raw = _make_dummy_image_bytes("PNG")
        with pytest.raises(ImageValidationError, match="exceeds limit"):
            validate_image_bytes(raw, max_size_bytes=10)

    def test_preprocess_to_nchw_and_postprocess(self):
        img = Image.new("RGB", (200, 300), color=(100, 150, 200))
        nchw = preprocess_image_to_nchw(img, target_size=(128, 128))

        assert isinstance(nchw, np.ndarray)
        assert nchw.shape == (1, 3, 128, 128)
        assert nchw.dtype == np.float32
        assert 0.0 <= nchw.min() <= nchw.max() <= 1.0

        recovered_img = postprocess_nchw_to_pil(nchw)
        assert recovered_img.size == (128, 128)
        assert recovered_img.mode == "RGB"

    def test_base64_data_url_roundtrip(self):
        original = Image.new("RGB", (32, 32), color=(50, 100, 150))
        data_url = pil_to_base64_data_url(original, img_format="PNG")

        assert data_url.startswith("data:image/png;base64,")
        recovered = base64_to_pil(data_url)
        assert recovered.size == (32, 32)
        assert recovered.mode == "RGB"

    def test_compute_error_map_base64(self):
        orig = np.zeros((1, 3, 128, 128), dtype=np.float32)
        rest = np.ones((1, 3, 128, 128), dtype=np.float32) * 0.5
        heatmap_url = compute_error_map_base64(orig, rest, colormap_name="turbo")

        assert heatmap_url.startswith("data:image/png;base64,")
        heatmap_pil = base64_to_pil(heatmap_url)
        assert heatmap_pil.size == (128, 128)


class TestModelSessionManager:
    """Validate ONNX Runtime session lifecycle management."""

    def test_load_all_models(self):
        mgr = ModelSessionManager()
        mgr.load_all_models()

        status, provider, model_status = mgr.get_health_status()
        assert status == "ok"
        assert provider in ("CPUExecutionProvider", "CUDAExecutionProvider")

        # Verify all 7 models are loaded
        for key in mgr.EXPECTED_MODELS:
            assert model_status[key] is True
            sess = mgr.get_session(key)
            assert sess is not None

    def test_get_missing_model_raises_key_error(self):
        mgr = ModelSessionManager()
        with pytest.raises(KeyError, match="not loaded"):
            mgr.get_session("nonexistent_model_key")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestFastAPIRoutes:
    """Validate REST endpoints using TestClient with lifespan events enabled."""

    def test_read_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["docs_url"] == "/docs"
        assert data["health_url"] == "/health"

    def test_get_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "task1_universal_ae" in data["models_loaded"]
        assert "task3_soft_moe" in data["models_loaded"]
        assert "task4_generator" in data["models_loaded"]
        assert all(data["models_loaded"].values()) is True
        assert "timestamp_utc" in data

    def test_openapi_docs_accessible(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/" in schema["paths"]
