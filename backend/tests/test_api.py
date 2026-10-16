"""
Phase 7 Backend Tests
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Tests:
  - GET  /api/health   — basic and model-loaded assertions
  - POST /api/predict  — valid image, missing file, bad extension, oversized file
  - POST /api/predict?gradcam=true — Grad-CAM overlay path is returned

Run with:
    cd backend
    pytest tests/ -v
"""

import io
import json
import os
import sys
from pathlib import Path

import pytest

# ── Make sure the repo root resolves so ml.* imports work ────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Flask test client fixture ─────────────────────────────────────────────────
from app import create_app


@pytest.fixture(scope="session")
def client():
    """Session-scoped test client (model loaded once for the whole suite)."""
    app = create_app("development")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_test_image() -> bytes:
    """
    Loads a real leaf image from the PlantVillage test split.
    Falls back to generating a tiny synthetic PNG if none is available.
    """
    test_dirs = [
        _REPO_ROOT / "ml" / "data" / "split" / "test",
    ]
    for test_dir in test_dirs:
        if test_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                images = list(test_dir.rglob(ext))
                if images:
                    return images[0].read_bytes()

    # Synthetic fallback: 1×1 white PNG
    import struct, zlib
    def _png_chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png  = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    png += _png_chunk(b"IEND", b"")
    return png


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_status_code_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, client):
        data = resp = client.get("/api/health").get_json()
        assert data["status"] == "ok"

    def test_response_has_model_loaded(self, client):
        data = client.get("/api/health").get_json()
        assert "model_loaded" in data

    def test_response_has_num_classes(self, client):
        data = client.get("/api/health").get_json()
        assert "num_classes" in data
        assert isinstance(data["num_classes"], int)

    def test_model_loaded_is_true(self, client):
        """Model should be preloaded successfully during app creation."""
        data = client.get("/api/health").get_json()
        assert data["model_loaded"] is True, (
            "Model was not loaded – check model_service paths and dependencies."
        )

    def test_num_classes_equals_13(self, client):
        data = client.get("/api/health").get_json()
        assert data["num_classes"] == 13


# ── Predict endpoint ──────────────────────────────────────────────────────────

class TestPredictEndpoint:

    # ── Positive cases ────────────────────────────────────────────────────────

    def test_predict_valid_image_returns_200(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_predict_response_structure(self, client):
        img_bytes = _load_test_image()
        data  = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        resp  = client.post("/api/predict", data=data, content_type="multipart/form-data")
        body  = resp.get_json()

        required_keys = [
            "status", "predicted_class", "class_index",
            "confidence", "confidence_pct", "all_probabilities", "inference_ms",
        ]
        for key in required_keys:
            assert key in body, f"Missing key '{key}' in response"

    def test_predict_status_success(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict", data=data, content_type="multipart/form-data").get_json()
        assert body["status"] == "success"

    def test_predict_confidence_range(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict", data=data, content_type="multipart/form-data").get_json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_confidence_pct_range(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict", data=data, content_type="multipart/form-data").get_json()
        assert 0.0 <= body["confidence_pct"] <= 100.0

    def test_predict_all_probabilities_sum_to_one(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict", data=data, content_type="multipart/form-data").get_json()
        total = sum(body["all_probabilities"].values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_predict_class_index_in_range(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict", data=data, content_type="multipart/form-data").get_json()
        assert 0 <= body["class_index"] < 13

    def test_predict_predicted_class_is_known(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict", data=data, content_type="multipart/form-data").get_json()
        assert body["predicted_class"] in body["all_probabilities"]

    # ── PNG upload ────────────────────────────────────────────────────────────

    def test_predict_png_upload(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.png"))
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200

    # ── Negative cases ────────────────────────────────────────────────────────

    def test_predict_missing_file_field_returns_400(self, client):
        resp = client.post("/api/predict", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["status"] == "error"

    def test_predict_empty_filename_returns_400(self, client):
        data = dict(file=(io.BytesIO(b""), ""))
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_predict_disallowed_extension_returns_400(self, client):
        data = dict(file=(io.BytesIO(b"fake content"), "malware.exe"))
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "not supported" in body["message"].lower() or "error" in body["status"]

    def test_predict_pdf_extension_returns_400(self, client):
        data = dict(file=(io.BytesIO(b"%PDF-1.4 fake"), "document.pdf"))
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_predict_oversized_file_returns_400(self, client):
        # 11 MB of zeros — exceeds the 10 MB limit
        big_bytes = b"\x00" * (11 * 1024 * 1024)
        data = dict(file=(io.BytesIO(big_bytes), "huge.jpg"))
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "mb" in body["message"].lower()


# ── Grad-CAM endpoint ─────────────────────────────────────────────────────────

class TestGradCAMEndpoint:

    def test_gradcam_returns_200(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        resp = client.post("/api/predict?gradcam=true", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_gradcam_response_has_gradcam_block(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict?gradcam=true", data=data, content_type="multipart/form-data").get_json()
        assert body["gradcam"] is not None
        assert "image_url" in body["gradcam"]
        assert "target_layer" in body["gradcam"]

    def test_gradcam_image_url_format(self, client):
        img_bytes = _load_test_image()
        data = dict(file=(io.BytesIO(img_bytes), "leaf.jpg"))
        body = client.post("/api/predict?gradcam=true", data=data, content_type="multipart/form-data").get_json()
        assert body["gradcam"]["image_url"].startswith("/static/gradcam/")

    def test_gradcam_predicted_class_matches_plain_predict(self, client):
        """Both paths must predict the same class for the same image."""
        img_bytes = _load_test_image()
        plain = client.post(
            "/api/predict",
            data=dict(file=(io.BytesIO(img_bytes), "leaf.jpg")),
            content_type="multipart/form-data",
        ).get_json()
        gcam = client.post(
            "/api/predict?gradcam=true",
            data=dict(file=(io.BytesIO(img_bytes), "leaf.jpg")),
            content_type="multipart/form-data",
        ).get_json()
        assert plain["predicted_class"] == gcam["predicted_class"]
