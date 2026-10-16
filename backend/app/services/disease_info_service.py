"""
Disease Information Service
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Loads the local disease_information.json dataset once at module import and
provides O(1) class-name lookups.  Never raises for missing classes — returns
a controlled sentinel value so that prediction failures are never caused by
a missing information entry.

Data file: backend/data/disease_information.json
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────────
# This file lives at: <repo>/backend/app/services/disease_info_service.py
# parents[0]=services  parents[1]=app  parents[2]=backend  parents[3]=repo_root
_BACKEND_DIR  = Path(__file__).resolve().parents[2]
_DATA_FILE    = _BACKEND_DIR / "data" / "disease_information.json"

# ── Module-level cache (load once, reuse forever) ─────────────────────────────
_disease_db: Optional[dict] = None


def _load_db() -> dict:
    """Loads the disease information JSON file into the module-level cache."""
    global _disease_db
    if _disease_db is not None:
        return _disease_db

    if not _DATA_FILE.exists():
        logger.error("Disease information file not found: %s", _DATA_FILE)
        _disease_db = {}
        return _disease_db

    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _disease_db = raw.get("entries", {})
        logger.info(
            "Disease information loaded: %d entries from %s",
            len(_disease_db),
            _DATA_FILE,
        )
    except Exception as exc:
        logger.error("Failed to load disease information: %s", exc)
        _disease_db = {}

    return _disease_db


# ── Public API ────────────────────────────────────────────────────────────────

def get_disease_info(class_name: str) -> Optional[dict]:
    """
    Returns the disease information dict for a given model class name.

    Args:
        class_name: The exact class name as produced by the model,
                    e.g. "Tomato___Early_blight".

    Returns:
        dict with keys: class_name, crop, display_name, type, is_healthy,
                        description, symptoms, management, prevention,
                        severity_note, disclaimer
        OR None if no entry exists for this class.
    """
    db = _load_db()
    entry = db.get(class_name)
    if entry is None:
        logger.warning("No disease information found for class: %s", class_name)
    return entry


def list_supported_classes() -> list[str]:
    """Returns a sorted list of all class names in the information database."""
    return sorted(_load_db().keys())


def is_healthy_class(class_name: str) -> bool:
    """
    Returns True if the given class is a healthy-leaf class according to
    the information database.  Falls back to string heuristic if no entry.
    """
    db = _load_db()
    entry = db.get(class_name)
    if entry is not None:
        return bool(entry.get("is_healthy", False))
    # Fallback heuristic
    return "healthy" in class_name.lower()
