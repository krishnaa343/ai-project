"""
Prediction History Service Layer
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Handles persistence and retrieval of crop disease predictions to/from MySQL.
All SQL queries are strictly parameterized to prevent SQL injection.
Exceptions during database operations are caught and handled gracefully so that
database failures never crash the AI prediction pipeline.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..db import get_db_connection

logger = logging.getLogger(__name__)


def save_prediction(
    image_name: Optional[str],
    crop: str,
    predicted_class: str,
    display_name: str,
    confidence: float,
    class_index: int,
    is_healthy: bool
) -> Dict[str, Any]:
    """
    Persists a prediction record into the prediction_history table.

    Returns:
        dict: {"saved": True, "prediction_id": <int>} on success, or
              {"saved": False, "prediction_id": None, "error": <str>} on failure.
    """
    sql = """
        INSERT INTO prediction_history
            (image_name, crop, predicted_class, display_name, confidence, class_index, is_healthy)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s);
    """
    params = (
        image_name,
        crop,
        predicted_class,
        display_name,
        float(confidence),
        int(class_index),
        bool(is_healthy)
    )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                prediction_id = cursor.lastrowid
                logger.info("Saved prediction id %s for class '%s'", prediction_id, predicted_class)
                return {
                    "saved": True,
                    "prediction_id": prediction_id,
                    "error": None
                }
    except Exception as exc:
        logger.error("Failed to persist prediction into MySQL: %s", exc)
        return {
            "saved": False,
            "prediction_id": None,
            "error": "Database persistence failed"
        }


def get_predictions(page: int = 1, limit: int = 20) -> Dict[str, Any]:
    """
    Retrieves paginated prediction records ordered by newest first (created_at DESC).

    Args:
        page: 1-indexed page number (min: 1)
        limit: number of items per page (min: 1, max: 100)

    Returns:
        dict containing 'predictions' list and 'pagination' metadata.
    """
    page = max(1, int(page))
    limit = max(1, min(500, int(limit)))
    offset = (page - 1) * limit

    count_sql = "SELECT COUNT(*) AS total FROM prediction_history;"
    fetch_sql = """
        SELECT
            id,
            image_name,
            crop,
            predicted_class,
            display_name,
            confidence,
            class_index,
            is_healthy,
            created_at
        FROM prediction_history
        ORDER BY created_at DESC, id DESC
        LIMIT %s OFFSET %s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(count_sql)
            count_row = cursor.fetchone()
            total = count_row["total"] if count_row else 0

            cursor.execute(fetch_sql, (limit, offset))
            rows = cursor.fetchall()

    predictions: List[Dict[str, Any]] = []
    for r in rows:
        created_val = r["created_at"]
        if isinstance(created_val, datetime):
            created_str = created_val.isoformat()
        else:
            created_str = str(created_val) if created_val else None

        predictions.append({
            "id": r["id"],
            "image_name": r["image_name"],
            "crop": r["crop"],
            "predicted_class": r["predicted_class"],
            "display_name": r["display_name"],
            "confidence": round(float(r["confidence"]), 4),
            "class_index": int(r["class_index"]),
            "is_healthy": bool(r["is_healthy"]),
            "created_at": created_str
        })

    total_pages = (total + limit - 1) // limit if total > 0 else 1

    return {
        "predictions": predictions,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages
        }
    }


def get_prediction_by_id(prediction_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single prediction by its primary key ID.

    Returns:
        dict if found, None if record does not exist.
    """
    sql = """
        SELECT
            id,
            image_name,
            crop,
            predicted_class,
            display_name,
            confidence,
            class_index,
            is_healthy,
            created_at
        FROM prediction_history
        WHERE id = %s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (int(prediction_id),))
            r = cursor.fetchone()
            if not r:
                return None

            created_val = r["created_at"]
            if isinstance(created_val, datetime):
                created_str = created_val.isoformat()
            else:
                created_str = str(created_val) if created_val else None

            return {
                "id": r["id"],
                "image_name": r["image_name"],
                "crop": r["crop"],
                "predicted_class": r["predicted_class"],
                "display_name": r["display_name"],
                "confidence": round(float(r["confidence"]), 4),
                "class_index": int(r["class_index"]),
                "is_healthy": bool(r["is_healthy"]),
                "created_at": created_str
            }
