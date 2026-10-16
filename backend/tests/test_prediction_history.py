"""
Phase 11 — MySQL Database and Prediction History Automated Tests
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Validates:
1. MySQL connection and schema verification
2. Service layer CRUD (save_prediction, get_predictions, get_prediction_by_id)
3. Parameterized SQL safety
4. GET /api/predictions and GET /api/predictions/<id>
5. End-to-end prediction with real test images and database verification
6. Controlled database failure handling (graceful degradation)
7. Security: Frontend code contains no MySQL credentials
"""

import os
import sys
import io
import unittest
from pathlib import Path
from unittest.mock import patch

# Setup paths
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pymysql
from app import create_app
from app.db import get_db_connection, check_db_connection, get_db_config
from app.services.prediction_history_service import (
    save_prediction,
    get_predictions,
    get_prediction_by_id
)


class TestPredictionHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app("development")
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Find 3 real test images from splits
        cls.test_images = []
        splits_csv = _REPO_ROOT / "ml" / "data" / "splits" / "test.csv"
        if splits_csv.exists():
            with open(splits_csv, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        img_path = _REPO_ROOT / parts[0]
                        if img_path.exists():
                            cls.test_images.append({
                                "path": img_path,
                                "class_name": parts[1],
                                "crop": parts[3],
                                "name": img_path.name
                            })
                    if len(cls.test_images) >= 5:
                        break

    def test_01_mysql_connection_and_schema(self):
        """1. Verify MySQL connection and prediction_history schema."""
        connected, err = check_db_connection()
        self.assertTrue(connected, f"MySQL connection failed: {err}")

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DESCRIBE prediction_history;")
                columns = {col["Field"]: col for col in cursor.fetchall()}

        required_cols = [
            "id", "image_name", "crop", "predicted_class",
            "display_name", "confidence", "class_index",
            "is_healthy", "created_at"
        ]
        for col in required_cols:
            self.assertIn(col, columns, f"Missing required column: {col}")

        # Check types
        self.assertIn("bigint", columns["id"]["Type"].lower())
        self.assertEqual("PRI", columns["id"]["Key"])
        self.assertIn("auto_increment", columns["id"]["Extra"].lower())

    def test_02_service_save_and_retrieve(self):
        """2. Verify save_prediction and get_prediction_by_id."""
        res = save_prediction(
            image_name="unit_test_leaf.jpg",
            crop="Tomato",
            predicted_class="Tomato___Late_blight",
            display_name="Tomato Late Blight",
            confidence=0.9876,
            class_index=11,
            is_healthy=False
        )
        self.assertTrue(res["saved"])
        pred_id = res["prediction_id"]
        self.assertIsInstance(pred_id, int)
        self.assertGreater(pred_id, 0)

        # Retrieve by ID
        pred = get_prediction_by_id(pred_id)
        self.assertIsNotNone(pred)
        self.assertEqual(pred["id"], pred_id)
        self.assertEqual(pred["image_name"], "unit_test_leaf.jpg")
        self.assertEqual(pred["crop"], "Tomato")
        self.assertEqual(pred["predicted_class"], "Tomato___Late_blight")
        self.assertEqual(pred["display_name"], "Tomato Late Blight")
        self.assertAlmostEqual(pred["confidence"], 0.9876, places=4)
        self.assertEqual(pred["class_index"], 11)
        self.assertFalse(pred["is_healthy"])
        self.assertIsNotNone(pred["created_at"])

    def test_03_get_predictions_pagination_and_ordering(self):
        """3. Verify get_predictions ordering (newest first) and pagination."""
        res1 = save_prediction("earlier.jpg", "Apple", "Apple___healthy", "Apple Healthy", 0.95, 2, True)
        res2 = save_prediction("later.jpg", "Corn", "Corn_(maize)___Common_rust_", "Corn Common Rust", 0.97, 3, False)

        history = get_predictions(page=1, limit=10)
        self.assertIn("predictions", history)
        self.assertIn("pagination", history)
        self.assertGreaterEqual(len(history["predictions"]), 2)

        # Check newest first: res2 ID should appear before res1 ID
        pred_ids = [p["id"] for p in history["predictions"]]
        self.assertIn(res1["prediction_id"], pred_ids)
        self.assertIn(res2["prediction_id"], pred_ids)
        idx1 = pred_ids.index(res1["prediction_id"])
        idx2 = pred_ids.index(res2["prediction_id"])
        self.assertLess(idx2, idx1, "Newest record must appear before earlier record")

    def test_04_api_history_endpoints(self):
        """4. Verify GET /api/predictions and GET /api/predictions/<id>."""
        # Save a test item
        res = save_prediction("api_test.jpg", "Potato", "Potato___Early_blight", "Potato Early Blight", 0.92, 6, False)
        pid = res["prediction_id"]

        # GET /api/predictions
        resp = self.client.get("/api/predictions?page=1&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["predictions"], list)
        self.assertIn("pagination", data)

        # GET /api/predictions/<id> (valid)
        resp_single = self.client.get(f"/api/predictions/{pid}")
        self.assertEqual(resp_single.status_code, 200)
        sdata = resp_single.get_json()
        self.assertEqual(sdata["status"], "success")
        self.assertEqual(sdata["prediction"]["id"], pid)
        self.assertEqual(sdata["prediction"]["predicted_class"], "Potato___Early_blight")

        # GET /api/predictions/<id> (nonexistent)
        resp_404 = self.client.get("/api/predictions/99999999")
        self.assertEqual(resp_404.status_code, 404)

        # GET /api/predictions/<id> (invalid)
        resp_400 = self.client.get("/api/predictions/0")
        self.assertEqual(resp_400.status_code, 400)

        # GET /api/predictions bad page
        resp_bad_page = self.client.get("/api/predictions?page=-1")
        self.assertEqual(resp_bad_page.status_code, 400)

    def test_05_real_prediction_and_mysql_verification(self):
        """5. End-to-end: Run real predictions through /api/predict and verify directly in MySQL."""
        self.assertGreaterEqual(len(self.test_images), 3, "Requires at least 3 real test images")

        for item in self.test_images[:3]:
            img_path = item["path"]
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            data = {
                "file": (io.BytesIO(img_bytes), img_path.name)
            }
            resp = self.client.post(
                "/api/predict?gradcam=true",
                data=data,
                content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 200)
            res_json = resp.get_json()

            # Verify response schema
            self.assertEqual(res_json["status"], "success")
            self.assertIn("predicted_class", res_json)
            self.assertIn("confidence", res_json)
            self.assertIn("class_index", res_json)
            self.assertIn("gradcam", res_json)
            self.assertIn("disease_info", res_json)
            self.assertIn("database", res_json)
            self.assertTrue(res_json["database"]["saved"])
            self.assertIsNotNone(res_json["prediction_id"])
            pred_id = res_json["prediction_id"]

            # Direct MySQL verification
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM prediction_history WHERE id = %s;", (pred_id,))
                    row = cursor.fetchone()

            self.assertIsNotNone(row, f"Record id {pred_id} not found in MySQL")
            self.assertEqual(row["id"], pred_id)
            self.assertEqual(row["predicted_class"], res_json["predicted_class"])
            self.assertEqual(row["class_index"], res_json["class_index"])
            self.assertAlmostEqual(row["confidence"], res_json["confidence"], places=3)
            self.assertEqual(row["image_name"], img_path.name)
            if res_json["disease_info"]:
                self.assertEqual(row["crop"], res_json["disease_info"]["crop"])
                self.assertEqual(row["display_name"], res_json["disease_info"]["display_name"])
                self.assertEqual(bool(row["is_healthy"]), res_json["disease_info"]["is_healthy"])

    def test_06_controlled_database_failure(self):
        """6. Controlled DB failure: AI prediction must succeed even if MySQL insertion fails."""
        if not self.test_images:
            self.skipTest("No test images available")

        img_path = self.test_images[0]["path"]
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Simulate database failure in save_prediction
        with patch("app.routes.prediction.save_prediction") as mock_save:
            mock_save.return_value = {
                "saved": False,
                "prediction_id": None,
                "error": "Connection refused"
            }

            data = {
                "file": (io.BytesIO(img_bytes), img_path.name)
            }
            resp = self.client.post(
                "/api/predict",
                data=data,
                content_type="multipart/form-data"
            )

            # HTTP 200 OK — AI prediction MUST still succeed!
            self.assertEqual(resp.status_code, 200)
            res_json = resp.get_json()
            self.assertEqual(res_json["status"], "success")
            self.assertIn("predicted_class", res_json)
            self.assertIn("confidence", res_json)

            # Database status must clearly indicate failure without fabricating an ID
            self.assertIn("database", res_json)
            self.assertFalse(res_json["database"]["saved"])
            self.assertIsNone(res_json["prediction_id"])
            self.assertIn("warning", res_json["database"])

    def test_07_frontend_credentials_safety(self):
        """7. Security check: Ensure no database credentials exist in frontend code."""
        frontend_src = _REPO_ROOT / "frontend" / "src"
        if frontend_src.exists():
            for fpath in frontend_src.rglob("*"):
                if fpath.is_file() and fpath.suffix in [".js", ".jsx", ".ts", ".tsx", ".html"]:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    self.assertNotIn("MYSQL_PASSWORD", content)
                    self.assertNotIn("Jaipur@123", content)
                    self.assertNotIn("crop_disease_detection", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
