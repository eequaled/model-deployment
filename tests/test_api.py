"""API tests that need no model weights.

The engine is replaced with a stub so the HTTP contract -- status codes, JSON shape,
upload limits, and the promise that bad input never crashes the process -- can be
checked in CI or on a laptop before the ONNX graphs exist. The real graphs are
exercised by `python -m bench.accuracy` against a running server.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server import app as app_module


class StubEngine:
    """Stands in for FlorenceOCR: same call shape, no inference."""

    def __init__(self, text: str = "hello world"):
        self.text = text
        self.seen: list[object] = []

    def generate(self, image):
        self.seen.append(image.size)
        return self.text, {"encoder_ms": 1.0, "decode_ms": 2.0, "total_ms": 3.0}


def png_bytes(size=(64, 32), color=(200, 200, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def client(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr(app_module, "WORKERS", 1)
    monkeypatch.setattr(app_module, "QUEUE_LIMIT", 1)
    monkeypatch.setattr(app_module, "build_engine", lambda: stub)
    with TestClient(app_module.app) as test_client:
        yield test_client, stub


def test_health_reports_ready(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ocr_returns_text(client):
    test_client, stub = client
    response = test_client.post("/ocr", files={"image": ("a.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["text"] == "hello world"
    assert stub.seen == [(64, 32)]


def test_non_image_is_a_4xx_not_a_crash(client):
    test_client, _ = client
    response = test_client.post("/ocr", files={"image": ("a.png", b"not an image", "image/png")})
    assert 400 <= response.status_code < 500
    assert "error" in response.json()
    # the server is still healthy afterwards
    assert test_client.get("/health").status_code == 200


def test_empty_upload_is_rejected(client):
    test_client, _ = client
    response = test_client.post("/ocr", files={"image": ("a.png", b"", "image/png")})
    assert response.status_code == 400


def test_missing_field_is_a_json_400(client):
    test_client, _ = client
    response = test_client.post("/ocr")
    assert response.status_code == 400
    assert "error" in response.json()


def test_oversized_upload_is_413(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(app_module, "MAX_UPLOAD", 1024)
    response = test_client.post("/ocr", files={"image": ("big.png", png_bytes((512, 512)), "image/png")})
    assert response.status_code == 413


def test_decompression_bomb_is_a_json_413(client, monkeypatch):
    test_client, stub = client
    payload = png_bytes()

    def bomb(*args, **kwargs):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr(app_module.Image, "open", bomb)
    response = test_client.post("/ocr", files={"image": ("bomb.png", payload, "image/png")})
    assert response.status_code == 413
    body = response.json()
    assert "error" in body and "pixel" in body["error"]
    assert stub.seen == []  # rejected before inference
    # the server is still healthy afterwards
    assert test_client.get("/health").status_code == 200
