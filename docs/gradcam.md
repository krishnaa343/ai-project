# Phase 6 -- Grad-CAM Explainable AI Specification

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Document:** Explainable AI (XAI) System Architecture & Grad-CAM Specification  
**Phase:** 6 (Explainable AI via Grad-CAM)  
**Status:** Completed  

---

## 1. What Grad-CAM Is

**Grad-CAM** (Gradient-weighted Class Activation Mapping; Selvaraju et al., 2017) is an Explainable Artificial Intelligence (XAI) visual localization technique designed for convolutional neural networks. 

Unlike black-box models that merely emit a class label and numerical confidence score, Grad-CAM utilizes the gradient of the predicted class score with respect to the final convolutional feature maps to produce a coarse 2D localization map highlighting the discriminative spatial regions in the image that most heavily influenced the network's prediction.

---

## 2. Why Grad-CAM Is Used in This Project

In agricultural disease diagnostics:
1. **Farmer & Agronomist Trust:** Farmers will not adopt AI guidance if recommendations appear opaque. Visual heatmaps confirm that the model detects actual foliar lesions (spots, blights, rust pustules) rather than spurious background artifacts (soil, sunlight glare, camera borders).
2. **Model Debugging & Error Analysis:** As identified in Phase 5, certain diseases (such as *Tomato Late Blight* vs *Tomato Early Blight*) share visual similarities. Grad-CAM visualizes whether the model focuses on the true concentric necrosis or peripheral healthy leaf tissue during ambiguous classifications.
3. **Regulatory & Extension Compliance:** College and institutional agricultural frameworks increasingly mandate explainability to ensure responsible AI practices.

---

## 3. Model Being Explained

- **Base Architecture:** `MobileNetV2` (Sandler et al., 2018)
- **Weights:** Trained two-stage transfer learning weights loaded from `ml/models/best_model.keras`
- **Trained Classes:** 13 crop disease and healthy states across Apple, Corn, Potato, and Tomato
- **Input Resolution:** `224 × 224 × 3` RGB format

---

## 4. Target Layer Selected & Selection Rationale

- **Selected Layer:** `out_relu` (nested within the `mobilenetv2_1.00_224` backbone)
- **Layer Output Shape:** `(None, 7, 7, 1280)`

### Why `out_relu` Was Chosen:
1. **Highest-Level Semantic Features:** In CNNs, early layers detect generic edges and color gradients, while deep convolutional layers capture complex domain-specific patterns (pustule textures, necrotic rings, chlorotic halos).
2. **Spatial Retention:** `out_relu` is the final layer in MobileNetV2 that preserves 2D spatial dimensions (`7 × 7`) before the `GlobalAveragePooling2D` layer compresses spatial coordinates into a flat 1D vector (`1280`).
3. **Post-Activation Stability:** As the rectified linear output of the final 1x1 point-wise expansion (`Conv_1` + `Conv_1_bn`), `out_relu` contains strictly non-negative activated activations, preventing gradient vanishing across dead paths.

---

## 5. Mathematical Grad-CAM Procedure

Given an input image $I$, the prediction score for class $c$ before softmax is $y^c$:

1. **Forward Propagation:** Pass $I$ through the convolutional backbone to extract feature activations $A^k \in \mathbb{R}^{u \times v}$ for each channel $k \in \{1, \dots, 1280\}$ at spatial resolution $7 \times 7$.
2. **Backpropagation of Target Score:** Compute the gradient of $y^c$ with respect to each activation map $A^k$:
   $$\frac{\partial y^c}{\partial A^k}$$
3. **Global Average Pooling of Gradients:** Calculate neuron importance weights $\alpha_k^c$ via global average pooling over height $u$ and width $v$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i, j}^k}$$
4. **Linear Combination:** Compute the weighted sum of forward feature maps:
   $$\sum_k \alpha_k^c A^k$$
5. **Rectified Linear Activation:** Apply ReLU to isolate features that have a *positive* influence on class $c$ (suppressing pixels that evidence competing classes):
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
6. **Normalization & Upsampling:** Normalize $L_{\text{Grad-CAM}}^c$ into range $[0.0, 1.0]$:
   $$H = \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c) + \epsilon}$$
   Bilinearly interpolate $H$ from $7 \times 7$ to the original input image resolution ($256 \times 256$ or native dimensions).
7. **Colormap & Alpha Blending:** Map normalized float intensities through a `JET` colormap (blue = low influence, green/yellow = moderate, red = maximum influence) and blend with original RGB image:
   $$\text{Overlay} = (1 - \alpha) \cdot I_{\text{orig}} + \alpha \cdot H_{\text{colormap}}, \quad \alpha = 0.55$$

---

## 6. Software Architecture & Module Interface

The system is implemented as a standalone, framework-agnostic Python package under `ml/explainability/` with zero UI/HTTP dependencies:

```
ml/explainability/
├── __init__.py           # Clean package exports
├── gradcam.py            # GradCAMExplainer class & gradient tape pipeline
├── image_utils.py        # Loading, normalization, colormap, overlay blending
└── test_gradcam.py       # 12 automated unit & integration tests
```

### Programmatic Python Interface
```python
from ml.explainability.gradcam import explain_prediction

# Drop-in one-line function for future Flask API
result = explain_prediction("path/to/leaf_image.jpg")

print(result["predicted_class"])      # e.g., 'Tomato___Early_blight'
print(result["confidence_pct"])       # e.g., 98.76
print(result["target_layer_name"])    # 'out_relu'
# result['heatmap_raw']               # (7, 7) float32 [0.0, 1.0]
# result['overlay']                   # (H, W, 3) uint8 RGB blended array
```

---

## 7. Preprocessing Consistency

To prevent subtle distribution shifts between inference and attribution:
- Raw images are loaded and decoded as 3-channel RGB.
- Bilinearly resized to $224 \times 224$ using `tf.image.resize` (exact match to Phase 3/4/5 pipelines).
- Divided by `255.0` to cast to `float32` in $[0.0, 1.0]$.
- Original image resolution is preserved separately so that the final overlay displays in full native resolution.

---

## 8. Empirical Demonstration Outputs

Eight representative test images spanning all 4 crops and multiple disease/healthy states were processed and saved to `ml/evaluation/gradcam/`:

| Sample ID | Crop | Actual Ground Truth | Predicted Class | Confidence | Prediction Status | Target Layer |
|:---:|:---|:---|:---|:---:|:---:|:---:|
| 1 | Apple | `Apple___Apple_scab` | `Apple___Apple_scab` | 98.76% | **CORRECT** | `out_relu` |
| 2 | Apple | `Apple___healthy` | `Apple___healthy` | 98.99% | **CORRECT** | `out_relu` |
| 3 | Corn | `Corn_(maize)___Common_rust_` | `Corn_(maize)___Common_rust_` | 100.0% | **CORRECT** | `out_relu` |
| 4 | Potato | `Potato___Early_blight` | `Potato___Early_blight` | 100.0% | **CORRECT** | `out_relu` |
| 5 | Tomato | `Tomato___healthy` | `Tomato___healthy` | 99.99% | **CORRECT** | `out_relu` |
| 6 | Tomato | `Tomato___Late_blight` | `Tomato___Early_blight` | 74.89% | **MISCLASSIFIED** | `out_relu` |
| 7 | Tomato | `Tomato___Early_blight` | `Tomato___Early_blight` | 53.02% | **CORRECT** | `out_relu` |
| 8 | Potato | `Potato___Late_blight` | `Tomato___Late_blight` | 98.43% | **MISCLASSIFIED** | `out_relu` |

### Key Visual Observations
- **Lesion Localization (Correct Diseases):** In `Apple___Apple_scab`, `Corn___Common_rust_`, and `Potato___Early_blight`, peak heatmap intensity (red/yellow) centers squarely on the necrotic lesions, pustules, and fungal rings.
- **Uniform Distribution (Healthy Leaves):** In `Apple___healthy` and `Tomato___healthy`, attention is diffusely spread across broad leaf lamina surfaces rather than focused on sharp focal spots.
- **Misclassified Analysis:** In Sample 6 (*Tomato Late Blight* predicted as *Early Blight*), the heatmap demonstrates that the model focused heavily on the dry concentric leaf margin rather than the diffuse water-soaked center, explaining why it associated the visual feature with Early Blight.

---

## 9. Latency Benchmark

Measured across 30 repeated runs on host CPU (Intel x86_64, AVX2 enabled):
- **Model Forward Pass Latency:** ~104 ms
- **Gradient Backpropagation & Heatmap Synthesis:** ~111 ms
- **Total End-to-End Latency (`explain_prediction`):** **215.04 ms per image**
- **Throughput:** ~4.65 full explanations/second on a single CPU core

---

## 10. Automated Test Results

Automated test suite (`ml/explainability/test_gradcam.py`) executed 12 rigorous checks:
- [x] **TEST 1:** Valid image produces a prediction
- [x] **TEST 2:** Prediction class exists in canonical class mapping
- [x] **TEST 3:** Confidence is bounded in $[0.0, 1.0]$
- [x] **TEST 4:** Grad-CAM heatmap is generated
- [x] **TEST 5:** Heatmap contains strictly finite values (0 NaNs, 0 Infs)
- [x] **TEST 6:** Heatmap dimensions match final conv tensor ($7 \times 7$)
- [x] **TEST 7:** Overlay successfully generated with matching native RGB dimensions
- [x] **TEST 8:** Selected convolutional layer verified (`out_relu`)
- [x] **TEST 9:** Gradients are non-zero and actively computed
- [x] **TEST 10:** Original image array preserved immutably (no in-place mutation)
- [x] **TEST 11:** Deterministic reproducibility verified (max numerical difference: $0.00 \times 10^0$)
- [x] **TEST 12:** Multi-class generalization verified across diverse crops
- [x] **EDGE CASES:** `FileNotFoundError` and `ValueError` raised cleanly on invalid input paths and malformed dimensions

---

## 11. Crucial Limitations: Explainability vs. Biological Diagnosis

> [!WARNING]
> **Scientific Integrity Notice:**  
> Grad-CAM visualizes **model attention/influence**, NOT biological disease causation.
> 1. A red region on a Grad-CAM overlay indicates only that the convolutional filters at that spatial location activated the network's final class logit.
> 2. It does not prove the presence of bacterial or fungal pathogens under a microscope.
> 3. Grad-CAM resolution is inherently bounded by the $7 \times 7$ grid of the final convolutional layer; fine micro-lesions smaller than ~32 pixels are spatially pooled.
