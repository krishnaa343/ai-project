"""
Prediction Service
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Runs deterministic preprocessing on an uploaded image and returns the
disease prediction + confidence using the loaded MobileNetV2 model.

Preprocessing is IDENTICAL to Phase 3 / 4 / 5 / 6:
  - RGB conversion
  - tf.image.resize to 224 x 224
  - Pixel normalisation to [0.0, 1.0]
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .model_service import get_model, get_idx_to_class


# ── Constants (must match training) ──────────────────────────────────────────
TARGET_HEIGHT = 224
TARGET_WIDTH  = 224


# ── Public API ────────────────────────────────────────────────────────────────

def predict_from_path(image_path: str | Path) -> dict:
    """
    Runs inference on the image at *image_path*.

    Args:
        image_path: absolute path to the saved upload on disk.

    Returns:
        dict with keys:
            predicted_class  (str)
            class_index      (int)
            confidence       (float)  e.g. 0.9421
            confidence_pct   (float)  e.g. 94.21
            all_probabilities (dict)  {class_name: float}
            inference_ms     (float)  wall-clock inference time in ms
    """
    tensor = _preprocess_image(Path(image_path))
    return _run_inference(tensor)


def predict_from_bytes(raw_bytes: bytes) -> dict:
    """
    Runs inference on a raw byte buffer (in-memory uploaded image).

    Args:
        raw_bytes: image file bytes.

    Returns:
        Same dict as predict_from_path.
    """
    tensor = _preprocess_bytes(raw_bytes)
    return _run_inference(tensor)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _preprocess_image(path: Path) -> np.ndarray:
    """Loads an image from disk and returns a model-ready (1, 224, 224, 3) array."""
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image at: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _apply_preprocessing(rgb)


def _preprocess_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes and returns a model-ready (1, 224, 224, 3) array."""
    import cv2

    nparr = np.frombuffer(raw_bytes, np.uint8)
    bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image from provided bytes.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _apply_preprocessing(rgb)


def _apply_preprocessing(rgb: np.ndarray) -> np.ndarray:
    """
    Applies the deterministic preprocessing pipeline used in all training phases.

    Steps:
      1. tf.image.resize → (224, 224)
      2. Pixel / 255.0 normalisation
      3. Batch dimension expansion → (1, 224, 224, 3)
    """
    img_tensor  = tf.convert_to_tensor(rgb, dtype=tf.float32)
    resized     = tf.image.resize(img_tensor, [TARGET_HEIGHT, TARGET_WIDTH])
    normalised  = resized / 255.0
    batched     = tf.expand_dims(normalised, axis=0)
    return batched.numpy()


def _run_inference(tensor: np.ndarray) -> dict:
    """Runs the model forward pass and structures the response dict."""
    model        = get_model()
    idx_to_class = get_idx_to_class()

    t0     = time.perf_counter()
    preds  = model.predict(tensor, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs       = preds[0]                        # shape: (num_classes,)
    class_idx   = int(np.argmax(probs))
    confidence  = float(probs[class_idx])
    class_name  = idx_to_class[class_idx]

    all_probs = {
        idx_to_class[i]: round(float(probs[i]), 6)
        for i in range(len(probs))
    }

    return {
        "predicted_class":   class_name,
        "class_index":       class_idx,
        "confidence":        round(confidence, 6),
        "confidence_pct":    round(confidence * 100, 2),
        "all_probabilities": all_probs,
        "inference_ms":      round(elapsed_ms, 1),
    }
