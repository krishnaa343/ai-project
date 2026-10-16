"""
Grad-CAM Service
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Bridges Flask requests to the Phase 6 GradCAMExplainer engine.
Saves the heatmap overlay to a temporary file and returns its path
alongside the structured prediction payload.

All heavy computation is delegated to ml/explainability/gradcam.py.
"""

import sys
import uuid
import threading
from pathlib import Path

import numpy as np
import cv2

# ── Ensure the repo root is on sys.path so ml.* imports resolve ──────────────
# gradcam_service.py lives at: <repo>/backend/app/services/gradcam_service.py
# parents: [0]=services, [1]=app, [2]=backend, [3]=repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.explainability.gradcam import GradCAMExplainer

# ── Singleton explainer ───────────────────────────────────────────────────────
_explainer = None
_lock      = threading.Lock()

# Generated images are stored here (backend/static/gradcam/)
_GRADCAM_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "static" / "gradcam"


def _get_explainer() -> GradCAMExplainer:
    """Returns the singleton GradCAMExplainer, creating it on first call."""
    global _explainer
    if _explainer is not None:
        return _explainer
    with _lock:
        if _explainer is None:
            _explainer = GradCAMExplainer()
    return _explainer


# ── Public API ────────────────────────────────────────────────────────────────

def generate_gradcam(image_path: str | Path, class_index: int | None = None) -> dict:
    """
    Runs full prediction + Grad-CAM for the image at *image_path*.

    Args:
        image_path:  Absolute path to the uploaded image on disk.
        class_index: Optional target class index (None → use predicted class).

    Returns:
        dict with keys:
            predicted_class      (str)
            class_index          (int)
            confidence           (float)
            confidence_pct       (float)
            target_layer         (str)
            all_probabilities    (dict)
            gradcam_image_path   (str)  absolute path to saved overlay PNG
            gradcam_filename     (str)  filename only, for URL construction
    """
    explainer = _get_explainer()
    result    = explainer.explain(image_path, class_index=class_index)

    # Persist the overlay image to disk
    _GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename    = f"gradcam_{uuid.uuid4().hex}.png"
    output_path = _GRADCAM_OUTPUT_DIR / filename

    overlay_rgb = result["overlay"]                           # np.ndarray uint8 RGB
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay_bgr)

    return {
        "predicted_class":    result["predicted_class"],
        "class_index":        result["predicted_class_index"],
        "confidence":         result["confidence"],
        "confidence_pct":     result["confidence_pct"],
        "target_layer":       result["target_layer_name"],
        "all_probabilities":  result["all_probabilities"],
        "gradcam_image_path": str(output_path),
        "gradcam_filename":   filename,
    }
