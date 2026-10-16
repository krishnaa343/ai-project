"""
Model Service
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Singleton loader for the trained MobileNetV2 Keras model and class-index mapping.
Thread-safe initialisation; model is loaded once at startup and reused across
all subsequent requests to avoid the overhead of repeated disk I/O.
"""

import json
import threading
from pathlib import Path

import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
# model_service.py lives at: <repo>/backend/app/services/model_service.py
# parents: [0]=services, [1]=app, [2]=backend, [3]=repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_DIR    = _REPO_ROOT / "ml"

DEFAULT_MODEL_PATH         = _ML_DIR / "models" / "best_model.keras"
DEFAULT_CLASS_INDICES_PATH = _ML_DIR / "data"   / "class_indices.json"

# ── Module-level singleton state ──────────────────────────────────────────────
_model        = None
_class_to_idx = None
_idx_to_class = None
_lock         = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(model_path=None, class_indices_path=None):
    """
    Loads (or returns the already-loaded) Keras model and class mappings.

    Args:
        model_path:         Optional path override for the .keras model file.
        class_indices_path: Optional path override for class_indices.json.

    Returns:
        model (tf.keras.Model): the loaded model.

    Raises:
        FileNotFoundError: if either required file is missing.
        RuntimeError:      if model loading fails for any reason.
    """
    global _model, _class_to_idx, _idx_to_class

    # Fast path – already initialised
    if _model is not None:
        return _model

    with _lock:
        # Double-checked locking
        if _model is not None:
            return _model

        m_path  = Path(model_path  or DEFAULT_MODEL_PATH)
        ci_path = Path(class_indices_path or DEFAULT_CLASS_INDICES_PATH)

        if not m_path.exists():
            raise FileNotFoundError(f"Model file not found: {m_path}")
        if not ci_path.exists():
            raise FileNotFoundError(f"Class indices file not found: {ci_path}")

        # Load class mappings first (fast, low-risk)
        with open(ci_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        _class_to_idx = mapping_data["class_to_idx"]
        _idx_to_class = {v: k for k, v in _class_to_idx.items()}

        # Load Keras model
        try:
            _model = tf.keras.models.load_model(str(m_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {m_path}: {exc}") from exc

        return _model


def get_model():
    """Returns the singleton model instance (must have been loaded first)."""
    if _model is None:
        return load_model()
    return _model


def get_class_to_idx():
    """Returns {class_name: index} dict (loads model/classes if needed)."""
    if _class_to_idx is None:
        load_model()
    return _class_to_idx


def get_idx_to_class():
    """Returns {index: class_name} dict (loads model/classes if needed)."""
    if _idx_to_class is None:
        load_model()
    return _idx_to_class


def get_num_classes():
    """Returns the total number of disease classes."""
    return len(get_class_to_idx())
