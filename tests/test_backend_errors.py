"""Cross-endpoint error-handling matrix: 400 (bad payload), 413 (oversized), 503 (model missing)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.backend.config import get_settings
from src.app.backend.main import app
from src.app.backend.services.inference import session_manager

ENDPOINTS = [
    "/api/v1/restore/universal",
    "/api/v1/restore/hard-routed",
    "/api/v1/restore/soft-moe",
    "/api/v1/sketch/generate",
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _post(client, url, name, data, ctype):
    return client.post(url, files={"file": (name, data, ctype)}, data={"style": "1"})


@pytest.mark.parametrize("url", ENDPOINTS)
class TestErrorMatrix:
    def test_non_image_returns_400(self, client, url):
        r = _post(client, url, "notes.txt", b"hello world, not an image", "text/plain")
        assert r.status_code == 400
        assert "detail" in r.json()

    def test_empty_payload_returns_400(self, client, url):
        assert _post(client, url, "empty.png", b"", "image/png").status_code == 400

    def test_unsupported_format_returns_400(self, client, url):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buf, format="BMP")
        r = _post(client, url, "x.bmp", buf.getvalue(), "image/bmp")
        assert r.status_code == 400
        assert "unsupported" in r.json()["detail"].lower()

    def test_oversized_returns_413(self, client, url, monkeypatch):
        monkeypatch.setattr(get_settings(), "max_upload_size_bytes", 100)
        r = _post(client, url, "big.png", _png(), "image/png")
        assert r.status_code == 413
        assert "exceeds" in r.json()["detail"].lower()

    def test_missing_model_returns_503(self, client, url, monkeypatch):
        monkeypatch.setattr(session_manager, "has_model", lambda key: False)
        r = _post(client, url, "ok.png", _png(), "image/png")
        assert r.status_code == 503
        assert "detail" in r.json()
