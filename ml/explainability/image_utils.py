"""
Image Utilities for Grad-CAM Explainability Pipeline
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Provides image loading, deterministic preprocessing (matching Phase 3/4/5),
heatmap colormap rendering, and transparency overlay blending.
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
INPUT_SHAPE  = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)


def load_and_preprocess_image(image_input, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    """
    Loads an image from file path, PIL Image, or numpy array, and prepares both:
    1. The preprocessed model-ready float tensor (1, H, W, 3) in [0.0, 1.0].
    2. The original RGB uint8 numpy image for display and overlay blending.

    Args:
        image_input: str, Path, PIL.Image, or np.ndarray (RGB or BGR).
        target_size: tuple (height, width), default (224, 224).

    Returns:
        tensor: np.ndarray of shape (1, H, W, 3), float32 scaled to [0, 1].
        original_rgb: np.ndarray of shape (orig_H, orig_W, 3), uint8 RGB.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise ValueError(f"Failed to decode image at path: {p}")
        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        original_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim != 3 or image_input.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel image array, got shape {image_input.shape}")
        if image_input.dtype != np.uint8:
            original_rgb = np.clip(image_input * 255 if image_input.max() <= 1.0 else image_input, 0, 255).astype(np.uint8)
        else:
            original_rgb = image_input.copy()
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # Exact same deterministic preprocessing as Phase 3, 4, 5
    img_tensor = tf.convert_to_tensor(original_rgb, dtype=tf.float32)
    resized_tensor = tf.image.resize(img_tensor, [target_size[0], target_size[1]])
    normalized_tensor = resized_tensor / 255.0
    tensor = tf.expand_dims(normalized_tensor, axis=0).numpy()

    return tensor, original_rgb


def render_heatmap_colormap(heatmap_2d, target_shape):
    """
    Resizes a 2D normalized float heatmap [0.0, 1.0] to target (H, W)
    and applies a JET colormap (RGB format).

    Args:
        heatmap_2d: 2D numpy array [0.0, 1.0] of activation intensities.
        target_shape: tuple (target_height, target_width).

    Returns:
        heatmap_rgb: np.ndarray uint8 of shape (target_height, target_width, 3).
    """
    h_target, w_target = target_shape[:2]
    # Resize 2D map to target resolution
    resized_map = cv2.resize(heatmap_2d, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
    resized_map = np.clip(resized_map, 0.0, 1.0)

    # Convert to 8-bit [0, 255] and apply JET colormap
    heatmap_uint8 = np.uint8(255 * resized_map)
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def create_gradcam_overlay(original_rgb, heatmap_rgb, alpha=0.55):
    """
    Blends the original RGB image with the colored Grad-CAM heatmap.

    Formula: overlay = (1 - alpha) * original + alpha * heatmap

    Args:
        original_rgb: np.ndarray uint8 (H, W, 3).
        heatmap_rgb: np.ndarray uint8 (H, W, 3).
        alpha: float blending weight for heatmap (0.0 to 1.0).

    Returns:
        overlay: np.ndarray uint8 (H, W, 3) blended RGB image.
    """
    if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def save_image_rgb(image_rgb, file_path):
    """Saves an RGB image array to disk in standard format."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)
    return p
