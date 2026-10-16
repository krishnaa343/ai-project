"""
Prediction Route  –  POST /api/predict
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Accepts a multipart/form-data image upload, validates it, runs the real
MobileNetV2 inference (+ optional Grad-CAM), and returns a structured
JSON response.

Query parameters:
    gradcam=true   →  include Grad-CAM heatmap overlay in the response.
"""

import os
import uuid
import time
from pathlib import Path

from flask import Blueprint, request, jsonify, current_app, url_for

from ..utils.file_validation import validate_image_upload, FileValidationError
from ..services.prediction_service import predict_from_path
from ..services.gradcam_service import generate_gradcam
from ..services.disease_info_service import get_disease_info
from ..services.prediction_history_service import save_prediction
from ..services.leaf_detector_service import classify_leaf_gate

prediction_bp = Blueprint("prediction", __name__)

# Temporary upload folder (backend/static/uploads/)
_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "static" / "uploads"


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /api/predict

    Body (multipart/form-data):
        file   – image file (jpg / jpeg / png / webp, max 10 MB)

    Query params:
        gradcam=true  – also generate and return Grad-CAM overlay

    Response 200:
    {
        "status":           "success",
        "predicted_class":  "Tomato___Early_blight",
        "class_index":      10,
        "confidence":       0.9421,
        "confidence_pct":   94.21,
        "all_probabilities": { ... },
        "inference_ms":     105.3,
        "gradcam": {                          // present only when ?gradcam=true
            "image_url":    "/static/gradcam/gradcam_<uuid>.png",
            "target_layer": "out_relu"
        },
        "disease_info": {                     // null if no entry found
            "display_name":  "...",
            "crop":          "...",
            "type":          "disease" | "healthy",
            "is_healthy":    false,
            "description":   "...",
            "symptoms":      [...],
            "management":    [...],
            "prevention":    [...],
            "severity_note": "...",
            "disclaimer":    "..."
        }
    }

    Response 4xx / 5xx:
    {
        "status":  "error",
        "message": "..."
    }
    """
    # ── 1. Presence check ─────────────────────────────────────────────────────
    if "file" not in request.files:
        return _error("No 'file' field found in the request.", 400)

    file_storage = request.files["file"]

    # ── 2. Validate upload ────────────────────────────────────────────────────
    try:
        safe_name = validate_image_upload(file_storage)
    except FileValidationError as exc:
        return _error(str(exc), 400)

    # ── 3. Save to disk with a unique prefix ──────────────────────────────────
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    unique_name  = f"{uuid.uuid4().hex}_{safe_name}"
    upload_path  = _UPLOAD_DIR / unique_name

    try:
        file_storage.seek(0)
        file_storage.save(str(upload_path))
    except Exception as exc:
        current_app.logger.exception("Failed to save uploaded file: %s", exc)
        msg = f"Failed to save uploaded file: {exc}" if current_app.config.get("DEBUG") else "Failed to process uploaded image."
        return _error(msg, 500)

    # ── 4. First-Stage Gate: Leaf / Non-Leaf Detection ────────────────────────
    try:
        leaf_gate = classify_leaf_gate(upload_path)
    except Exception as gate_exc:
        current_app.logger.warning("Leaf gate classification failed: %s; allowing through to disease model", gate_exc)
        leaf_gate = {"is_leaf": True, "leaf_probability": 1.0, "non_leaf_probability": 0.0, "confidence": 1.0}

    if not leaf_gate["is_leaf"]:
        current_app.logger.info(
            "Image rejected by Leaf Gate: P(NON_LEAF)=%.4f, P(LEAF)=%.4f",
            leaf_gate["non_leaf_probability"],
            leaf_gate["leaf_probability"]
        )
        return jsonify({
            "status": "rejected",
            "reason": "non_leaf_image",
            "message": "The uploaded image does not appear to contain a crop leaf. Please upload a clear photo of a crop leaf.",
            "leaf_detection": {
                "is_leaf": False,
                "confidence": leaf_gate["confidence"],
                "leaf_probability": leaf_gate["leaf_probability"],
                "non_leaf_probability": leaf_gate["non_leaf_probability"]
            }
        }), 200

    # ── 5. Run 13-Class AI Disease Prediction (Only for valid leaves) ──────────
    try:
        want_gradcam = request.args.get("gradcam", "false").lower() == "true"

        if want_gradcam:
            t_start = time.perf_counter()
            result = generate_gradcam(upload_path)
            inference_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
            predicted_class = result["predicted_class"]
            response_payload = {
                "status":            "success",
                "predicted_class":   predicted_class,
                "class_index":       result["class_index"],
                "confidence":        result["confidence"],
                "confidence_pct":    result["confidence_pct"],
                "all_probabilities": result["all_probabilities"],
                "inference_ms":      inference_ms,
                "leaf_detection": {
                    "is_leaf": True,
                    "confidence": leaf_gate["confidence"],
                    "leaf_probability": leaf_gate["leaf_probability"],
                    "non_leaf_probability": leaf_gate["non_leaf_probability"]
                },
                "gradcam": {
                    "image_url":    f"/static/gradcam/{result['gradcam_filename']}",
                    "target_layer": result["target_layer"],
                },
            }
        else:
            pred = predict_from_path(upload_path)
            predicted_class = pred["predicted_class"]
            response_payload = {
                "status":            "success",
                "predicted_class":   predicted_class,
                "class_index":       pred["class_index"],
                "confidence":        pred["confidence"],
                "confidence_pct":    pred["confidence_pct"],
                "all_probabilities": pred["all_probabilities"],
                "inference_ms":      pred["inference_ms"],
                "leaf_detection": {
                    "is_leaf": True,
                    "confidence": leaf_gate["confidence"],
                    "leaf_probability": leaf_gate["leaf_probability"],
                    "non_leaf_probability": leaf_gate["non_leaf_probability"]
                },
                "gradcam":           None,
            }

        # ── 5. Attach disease information (isolated — never breaks prediction) ──
        try:
            response_payload["disease_info"] = get_disease_info(predicted_class)
        except Exception as info_exc:
            current_app.logger.warning(
                "Disease info lookup failed for '%s': %s", predicted_class, info_exc
            )
            response_payload["disease_info"] = None

        # ── 6. Persist prediction to MySQL (isolated — never breaks prediction) ──
        try:
            disease_info = response_payload.get("disease_info") or {}
            crop_name = disease_info.get("crop")
            if not crop_name:
                crop_name = predicted_class.split("___")[0].replace("_", " ")

            display_name = disease_info.get("display_name")
            if not display_name:
                display_name = predicted_class.replace("___", " - ").replace("_", " ")

            is_healthy = bool(disease_info.get("is_healthy", False))
            if not disease_info and "healthy" in predicted_class.lower():
                is_healthy = True

            original_filename = getattr(file_storage, "filename", None) or safe_name

            db_res = save_prediction(
                image_name=original_filename,
                crop=crop_name,
                predicted_class=predicted_class,
                display_name=display_name,
                confidence=float(response_payload["confidence"]),
                class_index=int(response_payload["class_index"]),
                is_healthy=is_healthy
            )

            if db_res.get("saved"):
                response_payload["database"] = {
                    "saved": True,
                    "prediction_id": db_res["prediction_id"]
                }
                response_payload["prediction_id"] = db_res["prediction_id"]
            else:
                current_app.logger.warning("Database persistence failed: %s", db_res.get("error"))
                response_payload["database"] = {
                    "saved": False,
                    "warning": "Prediction succeeded, but database persistence failed.",
                    "prediction_id": None
                }
                response_payload["prediction_id"] = None
        except Exception as db_exc:
            current_app.logger.exception("Unexpected error during database persistence: %s", db_exc)
            response_payload["database"] = {
                "saved": False,
                "warning": "Prediction succeeded, but database persistence failed.",
                "prediction_id": None
            }
            response_payload["prediction_id"] = None

    except Exception as exc:
        current_app.logger.exception("Prediction failed")
        msg = f"Prediction error: {exc}" if current_app.config.get("DEBUG") else "An error occurred while processing the prediction. Please try again."
        return _error(msg, 500)

    finally:
        # Clean up the temporary upload (best-effort)
        try:
            if upload_path.exists():
                os.remove(upload_path)
        except OSError:
            pass

    return jsonify(response_payload), 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error(message: str, status_code: int):
    return jsonify({"status": "error", "message": message}), status_code
