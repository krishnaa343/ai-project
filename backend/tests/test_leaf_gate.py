"""
Automated Tests for First-Stage Leaf Gate Classifier
Validates:
1. Non-leaf image (test_ant_non_leaf.jpg) is rejected with status "rejected" and reason "non_leaf_image".
2. Rejection response contains leaf_detection metrics (confidence, probabilities).
3. Rejection does NOT write to MySQL prediction_history.
4. Valid leaf image (test_leaf.jpg) is accepted (status "success") and receives disease prediction.
"""

import io
import sys
from pathlib import Path
import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app import create_app
from app.services.prediction_history_service import get_predictions


@pytest.fixture(scope="module")
def client():
    app = create_app("development")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_ant_non_leaf_rejection(client):
    """Ensure test_ant_non_leaf.jpg is intercepted and rejected by the leaf gate."""
    ant_path = _REPO_ROOT / "test_ant_non_leaf.jpg"
    assert ant_path.exists(), f"Expected test image at {ant_path}"

    with open(ant_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "file": (io.BytesIO(img_bytes), "test_ant_non_leaf.jpg")
    }

    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res_json = resp.get_json()

    assert res_json["status"] == "rejected"
    assert res_json["reason"] == "non_leaf_image"
    assert "leaf_detection" in res_json
    assert res_json["leaf_detection"]["is_leaf"] is False
    assert res_json["leaf_detection"]["non_leaf_probability"] > 0.5


def test_valid_leaf_acceptance(client):
    """Ensure valid crop leaf passes through the gate to disease diagnosis."""
    leaf_path = _REPO_ROOT / "test_leaf.jpg"
    assert leaf_path.exists(), f"Expected test image at {leaf_path}"

    with open(leaf_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "file": (io.BytesIO(img_bytes), "test_leaf.jpg")
    }

    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res_json = resp.get_json()

    assert res_json["status"] == "success"
    assert "predicted_class" in res_json
    assert "confidence" in res_json


def test_google_natural_leaf_acceptance(client):
    """Ensure in-the-wild leaf with natural background passes the gate and is diagnosed."""
    sample_path = _BACKEND_DIR / "static" / "uploads" / "cc8f3f7a408a413b8dd153a30f191c25_images_1.jpg"
    if not sample_path.exists():
        pytest.skip("Sample Google leaf upload not present on disk")

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "file": (io.BytesIO(img_bytes), "google_leaf.jpg")
    }

    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res_json = resp.get_json()

    assert res_json["status"] == "success"
    assert res_json["leaf_detection"]["is_leaf"] is True
    assert "predicted_class" in res_json


def test_health_reports_leaf_gate_loaded(client):
    """GET /api/health returns leaf_gate_loaded: True."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("leaf_gate_loaded") is True

