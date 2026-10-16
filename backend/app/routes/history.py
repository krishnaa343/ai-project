"""
Prediction History Route
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Endpoints:
    GET /api/predictions         - Retrieves paginated prediction history (newest first)
    GET /api/predictions/<id>    - Retrieves single prediction record by ID
"""

from flask import Blueprint, request, jsonify, current_app

from ..services.prediction_history_service import (
    get_predictions,
    get_prediction_by_id
)

history_bp = Blueprint("history", __name__)


@history_bp.route("/predictions", methods=["GET"])
def list_predictions():
    """
    GET /api/predictions?page=1&limit=20

    Query Parameters:
        page   - Page number (integer >= 1, default 1)
        limit  - Items per page (integer 1-100, default 20)

    Response 200:
    {
        "status": "success",
        "predictions": [
            {
                "id": 1,
                "image_name": "leaf.jpg",
                "crop": "Tomato",
                "predicted_class": "Tomato___Early_blight",
                "display_name": "Tomato Early Blight",
                "confidence": 0.9421,
                "class_index": 10,
                "is_healthy": false,
                "created_at": "2026-10-05T14:30:00"
            }
        ],
        "pagination": {
            "page": 1,
            "limit": 20,
            "total": 1,
            "total_pages": 1
        }
    }
    """
    page_param = request.args.get("page", 1)
    limit_param = request.args.get("limit", 20)

    try:
        page = int(page_param)
        if page < 1:
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({
            "status": "error",
            "message": "Query parameter 'page' must be a positive integer."
        }), 400

    try:
        limit = int(limit_param)
        if limit < 1 or limit > 500:
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({
            "status": "error",
            "message": "Query parameter 'limit' must be an integer between 1 and 500."
        }), 400

    try:
        result = get_predictions(page=page, limit=limit)
        return jsonify({
            "status": "success",
            "predictions": result["predictions"],
            "pagination": result["pagination"]
        }), 200
    except Exception as exc:
        current_app.logger.exception("Failed to retrieve prediction history: %s", exc)
        return jsonify({
            "status": "error",
            "message": "Could not retrieve prediction history from the database."
        }), 500


@history_bp.route("/predictions/<int:pred_id>", methods=["GET"])
def get_prediction(pred_id: int):
    """
    GET /api/predictions/<id>

    Returns:
        200: Prediction object if found
        400: Invalid ID
        404: Prediction not found
        500: Database error
    """
    if pred_id <= 0:
        return jsonify({
            "status": "error",
            "message": "Prediction ID must be a positive integer."
        }), 400

    try:
        prediction = get_prediction_by_id(pred_id)
        if prediction is None:
            return jsonify({
                "status": "error",
                "message": f"Prediction with ID {pred_id} was not found."
            }), 404

        return jsonify({
            "status": "success",
            "prediction": prediction
        }), 200
    except Exception as exc:
        current_app.logger.exception("Failed to retrieve prediction id %s: %s", pred_id, exc)
        return jsonify({
            "status": "error",
            "message": "Could not retrieve the prediction from the database."
        }), 500
