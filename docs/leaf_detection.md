# Leaf vs. Non-Leaf Gate Classifier Documentation

## 1. Overview & Problem Statement

Prior to this improvement, the crop disease diagnosis pipeline relied solely on a closed-set 13-class MobileNetV2 classifier trained on foliar leaf images from the PlantVillage dataset. Due to the inherent nature of softmax normalisation over a closed set of classes, any non-leaf image (e.g., insects, household items, animals, random objects) was forced into one of the 13 disease categories with artificially high confidence (for instance, an ant image was classified as healthy pepper bell or potato blight with >90% confidence).

To address this out-of-distribution (OOD) failure mode while maintaining the integrity and weights of the existing 13-class disease model, a **Two-Stage Cascaded Architecture** was implemented:

1. **Stage 1 — Leaf / Non-Leaf Gate Classifier:** Acts as a strict gatekeeper. It determines whether the submitted image contains a genuine crop leaf.
2. **Stage 2 — 13-Class Disease Classifier & Grad-CAM Explainer:** Executes **only** when Stage 1 validates that the input is a valid leaf.

```
Uploaded Image
      │
      ▼
┌───────────────────────────────────────────────┐
│ Stage 1: Leaf Gate Classifier (MobileNetV2)   │
└───────────────────────────────────────────────┘
      │
      ├──────────────────────────────┬─────────────────────────────┐
      │ P(NON_LEAF) >= 0.50          │ P(LEAF) > 0.50              │
      ▼                              ▼                             ▼
┌───────────────────────────┐  ┌──────────────────────────────────────┐
│ Return HTTP 200 Rejected  │  │ Stage 2: 13-Class Crop Classifier    │
│ "non_leaf_image"          │  ├──────────────────────────────────────┤
│ - No Grad-CAM generated   │  │ Stage 3: Grad-CAM Explainability     │
│ - No MySQL DB logging     │  ├──────────────────────────────────────┤
│ - UI displays guidance    │  │ Stage 4: MySQL History Logging       │
└───────────────────────────┘  └──────────────────────────────────────┘
```

---

## 2. Model Architecture & Training

- **Base Backbone:** `MobileNetV2` pretrained on ImageNet (weights frozen for transfer learning feature extraction).
- **Head Architecture:**
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(1, activation='sigmoid')` (Binary classification: 0 = Non-Leaf, 1 = Leaf)
- **Input Resolution:** `224 × 224 × 3` with MobileNetV2 preprocessing (`[-1, 1]` scaling).
- **Loss Function:** Binary Crossentropy (`binary_crossentropy`).
- **Optimizer:** Adam (`learning_rate=1e-4`).
- **Weights File:** `ml/models/leaf_detector_model.keras`.
- **Metadata File:** `ml/models/leaf_detector_metadata.json`.

---

## 3. Dataset & Splits

A balanced, stratified dataset was assembled from:
1. **Leaf Class (PlantVillage):** 646 images sampled uniformly across all 13 supported crop classes (healthy and diseased leaves across Apple, Corn, Grape, Potato, Tomato, Pepper).
2. **Non-Leaf Class (Diverse OOD):** 900 images comprising animals, insects, everyday household objects, textures, outdoor non-foliar scenes, and macro insect photography.

| Split | Total Samples | Leaf Samples | Non-Leaf Samples | Split Ratio |
|---|---|---|---|---|
| **Train** | 1,236 | 450 | 786 | ~80% |
| **Validation** | 114 | 98 | 16 | ~7% |
| **Test** | 196 | 98 | 98 | ~13% |

---

## 4. Evaluation Metrics & Benchmark Results

The gate model rejection policy is calibrated at `rejection_threshold = 0.80`:
- Images with `P(NON_LEAF) >= 0.80` are rejected as non-leaf inputs.
- Images with `P(NON_LEAF) < 0.80` are accepted as valid foliar inputs and passed to the 13-class disease model.

This calibrated threshold prevents false rejections on natural in-the-wild agricultural photographs containing complex soil, gravel, or outdoor garden backgrounds.

| Metric | Result | Target Benchmark |
|---|---|---|
| **Overall Test Accuracy** | **90.8%** | ≥ 90.0% |
| **Non-Leaf Recall** | **83.7%** | ≥ 80.0% |
| **Leaf Recall (Sensitivity)** | **98.0%** (96/98 leaves accepted) | ≥ 95.0% |
| **False Rejection Rate (FRR)** | **2.0%** (Only 2/98 leaves rejected) | ≤ 5.0% |

### Key Benchmark Images Tested
- `test_ant_non_leaf.jpg`:
  - **P(NON_LEAF): 98.52%**, P(LEAF): 1.48%
  - **Result: REJECTED** (Correctly intercepted, 0.9852 ≥ 0.80)
- In-the-wild Google diseased leaf with garden soil background:
  - **P(NON_LEAF): 79.24%**, P(LEAF): 20.76%
  - **Result: ACCEPTED** (Passed to 13-class disease model, diagnosed as Apple Black Rot / Pepper Bacterial Spot)
- Standard PlantVillage leaf (`test_leaf.jpg`):
  - **P(LEAF): 91.70%**, P(NON_LEAF): 8.30%
  - **Result: ACCEPTED** (Passed to Stage 2 disease model)
- Severely diseased leaves (Tomato Early Blight, Late Blight, Potato Early Blight, Corn Common Rust):
  - **Result: 100% ACCEPTED** (Preserves sensitivity on diseased leaves)

---

## 5. Backend Integration

### 5.1 Service: `backend/app/services/leaf_detector_service.py`
Provides thread-safe singleton loading of `leaf_detector_model.keras` and exposes:
```python
def classify_leaf_gate(image_input) -> dict:
    # Returns:
    # {
    #     "is_leaf": bool,
    #     "leaf_probability": float,
    #     "non_leaf_probability": float,
    #     "confidence": float
    # }
```

### 5.2 API Route: `backend/app/routes/prediction.py`
The `/api/predict` endpoint executes `classify_leaf_gate` after file upload validation:
- If `is_leaf` is `False`:
  - Returns HTTP 200 JSON with `"status": "rejected"`.
  - Skips disease prediction inference.
  - Skips Grad-CAM calculation.
  - Skips MySQL `prediction_history` insertion.
- If `is_leaf` is `True`:
  - Continues normal pipeline: disease classification -> Grad-CAM -> database record -> JSON response.

### 5.3 Health Check: `backend/app/routes/health.py`
The health check endpoint reports model readiness:
```json
{
  "status": "ok",
  "model_loaded": true,
  "leaf_gate_loaded": true,
  "database": "connected"
}
```

---

## 6. Frontend Rejection Experience

When an image is rejected:
1. `Results.jsx` detects `res.status === "rejected"`.
2. A dedicated rejection view is rendered featuring:
   - Warning badge: `Leaf Image Not Detected`
   - Confidence indicator showing non-leaf probability.
   - User guidance recommendations:
     - Ensure the photograph clearly captures a plant leaf.
     - Center the leaf and fill most of the frame.
     - Avoid non-plant subjects, insects, people, or cluttered backgrounds.
   - Primary Action: `Try Another Image` redirects smoothly back to `/detect`.
