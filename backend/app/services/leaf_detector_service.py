"""
Leaf Detector Service
AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Singleton loader and inference pipeline for the Leaf / Non-Leaf Gate Classifier.
Acts as a first-stage input validation gate before the 13-class disease model.
"""

import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATE_MODEL_PATH = _REPO_ROOT / "ml" / "models" / "leaf_detector_model.keras"

_gate_model = None
_gate_lock = threading.Lock()
IMAGE_SIZE = (224, 224)

# Configurable rejection policy
# If non_leaf_probability >= NON_LEAF_THRESHOLD -> image is rejected
# Threshold 0.80 provides high recall against non-leaves (ant = 0.985) while avoiding
# false rejections on real in-the-wild leaves with natural garden/soil backgrounds.
DEFAULT_NON_LEAF_THRESHOLD = 0.80


def load_leaf_detector(model_path: Optional[Path] = None):
    """
    Loads (or returns the already-loaded) binary Leaf/Non-Leaf Gate Keras model.
    Thread-safe singleton with double-checked locking.
    """
    global _gate_model

    if _gate_model is not None:
        return _gate_model

    with _gate_lock:
        if _gate_model is not None:
            return _gate_model

        target_path = Path(model_path or DEFAULT_GATE_MODEL_PATH)
        if not target_path.exists():
            raise FileNotFoundError(f"Leaf gate detector model not found at: {target_path}")

        try:
            _gate_model = tf.keras.models.load_model(str(target_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load leaf detector model: {exc}") from exc

    return _gate_model


def is_leaf_detector_loaded() -> bool:
    """Returns True if the leaf detector model is currently loaded in memory."""
    return _gate_model is not None


def classify_leaf_gate(
    image_path: Path,
    threshold: float = DEFAULT_NON_LEAF_THRESHOLD
) -> Dict[str, Any]:
    """
    Evaluates whether an image contains a crop leaf before running the disease classifier.

    Args:
        image_path: Path to the image file to classify.
        threshold: Decision threshold for non_leaf_probability (default: 0.50).

    Returns:
        Dict with keys:
            - is_leaf: bool (True if image passes gate and is a leaf)
            - confidence: float (model confidence for the winning class)
            - leaf_probability: float [0, 1]
            - non_leaf_probability: float [0, 1]
            - latency_ms: float (inference latency in milliseconds)
    """
    model = load_leaf_detector()

    t_start = time.perf_counter()

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        # Dual-aspect evaluation: evaluate both full squashed view and aspect-ratio
        # preserving center-fit view. This ensures vertical phone photos (e.g. Corn leaves)
        # do not have their cellular venation distorted by horizontal compression.
        t_direct = np.array(im.resize(IMAGE_SIZE), dtype=np.float32) / 255.0
        t_fit = np.array(ImageOps.fit(im, IMAGE_SIZE, method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    batch = np.stack([t_direct, t_fit], axis=0)
    preds = model.predict(batch, verbose=0)
    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

    # Class 0: NON_LEAF, Class 1: LEAF
    # Take the minimum non-leaf probability across the canonical and aspect-fit views
    p_non_leaf = min(float(preds[0][0]), float(preds[1][0]))
    p_leaf = round(1.0 - p_non_leaf, 4)
    p_non_leaf = round(p_non_leaf, 4)

    is_non_leaf = p_non_leaf >= threshold
    is_leaf = not is_non_leaf
    winning_conf = p_leaf if is_leaf else p_non_leaf

    return {
        "is_leaf": bool(is_leaf),
        "confidence": winning_conf,
        "leaf_probability": p_leaf,
        "non_leaf_probability": p_non_leaf,
        "threshold": threshold,
        "latency_ms": latency_ms
    }
