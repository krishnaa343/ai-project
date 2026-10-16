"""
Grad-CAM Explainable AI Engine
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) (Selvaraju et al., 2017)
for the trained MobileNetV2 crop disease classifier.
Provides:
  - Automated target convolutional layer discovery
  - Gradient computation via tf.GradientTape
  - Normalized 2D heatmap generation
  - Reusable prediction + explanation pipeline
  - Full modularity for backend integration
"""

import sys, json
from pathlib import Path
import numpy as np
import tensorflow as tf

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from image_utils import load_and_preprocess_image, render_heatmap_colormap, create_gradcam_overlay
except ImportError:
    from ml.explainability.image_utils import load_and_preprocess_image, render_heatmap_colormap, create_gradcam_overlay


# ── Default Paths ─────────────────────────────────────────────────────────────
REPO_ROOT          = Path(__file__).resolve().parent.parent.parent
ML_DIR             = REPO_ROOT / "ml"
DEFAULT_MODEL_PATH = ML_DIR / "models" / "best_model.keras"
CLASS_INDICES_PATH = ML_DIR / "data" / "class_indices.json"


def find_last_conv_layer(model, preferred_names=("out_relu", "Conv_1")):
    """
    Programmatically discovers the last suitable convolutional layer in the model.
    Handles both direct models and nested transfer-learning functional backbones.

    Args:
        model: tf.keras.Model
        preferred_names: tuple of preferred layer names in order of priority.

    Returns:
        (base_layer_or_model, target_layer)
    """
    # Look for nested backbone (e.g., mobilenetv2_1.00_224)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
            backbone = layer
            break

    search_container = backbone if backbone is not None else model

    # 1. Check preferred names first
    for name in preferred_names:
        try:
            target_l = search_container.get_layer(name)
            return search_container, target_l
        except (ValueError, KeyError):
            pass

    # 2. Otherwise search backwards for any Conv2D / DepthwiseConv2D layer
    for layer in reversed(search_container.layers):
        if "conv" in layer.__class__.__name__.lower() or "relu" in layer.name.lower():
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return search_container, layer

    raise ValueError(f"Could not locate a suitable 4D convolutional layer in {model.name}")


class GradCAMExplainer:
    """
    Reusable Grad-CAM explainability engine for crop disease diagnosis.
    Encapsulates the trained model and builds optimized sub-models for feature
    extraction and gradient propagation.
    """

    def __init__(self, model_path=None, class_indices_path=None, target_layer_name=None):
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        class_indices_path = Path(class_indices_path or CLASS_INDICES_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not class_indices_path.exists():
            raise FileNotFoundError(f"Class indices file not found at: {class_indices_path}")

        # Load class mappings
        with open(class_indices_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        self.class_to_idx = mapping_data["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes  = len(self.class_to_idx)

        # Load trained model
        self.model = tf.keras.models.load_model(model_path)

        # Identify convolutional target layer
        if target_layer_name:
            self.backbone, self.target_layer = self._get_specific_layer(target_layer_name)
        else:
            self.backbone, self.target_layer = find_last_conv_layer(self.model)

        self.target_layer_name = self.target_layer.name

        # Build sub-models for gradient tape
        self._build_submodels()

    def _get_specific_layer(self, layer_name):
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                try:
                    return layer, layer.get_layer(layer_name)
                except ValueError:
                    pass
        return self.model, self.model.get_layer(layer_name)

    def _build_submodels(self):
        """Constructs conv_model and head_model for efficient gradient tracking."""
        base_inputs = self.backbone.inputs
        target_output = self.target_layer.output

        # 1. Convolutional feature extractor
        self.conv_model = tf.keras.Model(inputs=base_inputs, outputs=target_output, name="gradcam_conv_extractor")

        # 2. Classifier head model
        head_input = tf.keras.Input(shape=target_output.shape[1:], name="gradcam_head_input")
        x = self.model.get_layer("global_avg_pool")(head_input)
        x = self.model.get_layer("dropout_head")(x, training=False)
        head_output = self.model.get_layer("predictions")(x)
        self.head_model = tf.keras.Model(inputs=head_input, outputs=head_output, name="gradcam_head_predictor")

    def generate_heatmap(self, preprocessed_tensor, target_class_index=None):
        """
        Computes the 2D Grad-CAM heatmap for a preprocessed image tensor.

        Args:
            preprocessed_tensor: np.ndarray or tf.Tensor of shape (1, 224, 224, 3) in [0, 1].
            target_class_index: int (optional). If None, uses model's predicted class.

        Returns:
            heatmap: 2D np.ndarray of shape (H_feat, W_feat), float in [0.0, 1.0].
            predicted_idx: int.
            confidence: float.
            all_probs: np.ndarray (1, num_classes).
        """
        tensor = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # 1. Forward pass through convolutional feature extractor
            conv_outputs = self.conv_model(tensor, training=False)
            tape.watch(conv_outputs)

            # 2. Forward pass through classifier head
            preds = self.head_model(conv_outputs, training=False)
            all_probs = preds.numpy()[0]

            predicted_idx = int(np.argmax(all_probs))
            chosen_class_idx = target_class_index if target_class_index is not None else predicted_idx
            loss = preds[:, chosen_class_idx]

        # 3. Calculate gradients of target class output w.r.t. conv feature maps
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradient computation returned None. Check if the target layer is connected to the loss.")

        # 4. Global average pooling of gradients (importance weights alpha_k)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 5. Weighted combination of feature maps
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 6. Apply ReLU (retain positive influence only)
        heatmap = tf.maximum(heatmap, 0.0)

        # 7. Normalize between 0 and 1
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val
        else:
            heatmap = tf.zeros_like(heatmap)

        heatmap_np = heatmap.numpy()
        confidence = float(all_probs[predicted_idx])
        return heatmap_np, predicted_idx, confidence, all_probs

    def explain(self, image_input, class_index=None, alpha=0.55):
        """
        End-to-end explanation pipeline for an input image.

        Args:
            image_input: file path, PIL Image, or numpy array.
            class_index: int (optional target class index).
            alpha: float transparency blending factor (default 0.55).

        Returns:
            dict containing prediction details, raw heatmap, colored heatmap, and blended overlay.
        """
        tensor, original_rgb = load_and_preprocess_image(image_input)

        heatmap_raw, pred_idx, confidence, all_probs = self.generate_heatmap(
            tensor, target_class_index=class_index
        )

        orig_shape = original_rgb.shape[:2]
        heatmap_colored = render_heatmap_colormap(heatmap_raw, orig_shape)
        overlay_rgb     = create_gradcam_overlay(original_rgb, heatmap_colored, alpha=alpha)

        pred_class_name = self.idx_to_class[pred_idx]

        probs_dict = {
            self.idx_to_class[i]: round(float(all_probs[i]), 6)
            for i in range(self.num_classes)
        }

        return {
            "predicted_class":      pred_class_name,
            "predicted_class_index":pred_idx,
            "confidence":           round(confidence, 6),
            "confidence_pct":       round(confidence * 100, 2),
            "target_layer_name":    self.target_layer_name,
            "heatmap_raw":          heatmap_raw,
            "heatmap_colored":      heatmap_colored,
            "overlay":              overlay_rgb,
            "original_rgb":         original_rgb,
            "all_probabilities":    probs_dict,
        }


# ── Global Reusable Helper ───────────────────────────────────────────────────
_global_explainer = None

def get_explainer():
    global _global_explainer
    if _global_explainer is None:
        _global_explainer = GradCAMExplainer()
    return _global_explainer


def explain_prediction(image_input, class_index=None, alpha=0.55):
    """
    Convenience function for prediction + Grad-CAM explanation.
    Direct drop-in for future Flask API calls.
    """
    explainer = get_explainer()
    return explainer.explain(image_input, class_index=class_index, alpha=alpha)
