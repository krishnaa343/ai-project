# Final Machine Learning Model Specification
## AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Document Version:** 1.0.0  
**Phase:** 16 — Final Documentation  
**Status:** Validated & Frozen  

---

## 1. Model Overview

The core diagnostic engine of CropScan AI utilizes a fine-tuned **MobileNetV2** deep convolutional neural network trained on a standardized 13-class subset of the PlantVillage agricultural benchmark dataset.

- **Base Backbone:** MobileNetV2 (pre-trained on ImageNet-1k)
- **Model Type:** Deep Convolutional Neural Network (CNN) with Inverted Residual Blocks and Linear Bottlenecks
- **Input Dimensions:** `(224, 224, 3)` (Height, Width, Channels — RGB)
- **Output Dimension:** `13` classes (Softmax probability distribution)
- **File Format:** Native Keras v3 format (`.keras`)
- **File Path:** `ml/models/best_model.keras`
- **File Size:** `24,750,602 bytes` (~24.75 MB)
- **Training Timestamp:** 2026-09-28T23:58:09.480881
- **Evaluation Accuracy:** **96.40%** (1,445 / 1,499 correct predictions on the held-out test split)

---

## 2. Neural Network Architecture Specification

MobileNetV2 was selected due to its exceptional parameter efficiency, low floating-point operations (FLOPs), and rapid inference execution on standard commodity hardware (CPU and mobile edge devices) without sacrificing discriminative feature extraction capacity.

```
Input Image (224 x 224 x 3 RGB, normalized [0, 1])
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Feature Extractor: MobileNetV2 Backbone                    │
│  - Pre-trained on ImageNet (1.4M images, 1,000 classes)     │
│  - 53 convolutional layers organized into 17 inverted       │
│    residual blocks with expansion factor = 6                │
│  - Depthwise separable convolutions (3x3 depthwise + 1x1    │
│    pointwise projections)                                   │
│  - Bottleneck residual connections                          │
│  - Terminal feature map layer: 'out_relu' (7 x 7 x 1280)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Global Average Pooling 2D                                  │
│  - Spatial reduction: (7 x 7 x 1280) -> (1280,) vector      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Dropout Regularization (rate = 0.3)                        │
│  - Prevents co-adaptation of features on high-weight nodes   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Classification Head (Dense Layer)                          │
│  - Units: 13                                                │
│  - Activation: Softmax                                      │
│  - Output: P(class_i | x) for i in {0, ..., 12}             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Two-Stage Transfer Learning Strategy

Training was conducted in two distinct sequential phases to prevent catastrophic forgetting of pre-trained low-level visual features (edges, textures) while specializing high-level representations for leaf pathology:

### Stage 1: Feature Extraction (Frozen Backbone)
- **Backbone Status:** Completely frozen (`base_model.trainable = False`). Only the classification head and pooling weights were updated.
- **Optimizer:** Adam
- **Learning Rate:** $\eta = 0.001$ ($10^{-3}$)
- **Epochs Run:** 5 epochs
- **Batch Size:** 32
- **Objective:** Warm up the new 13-class Dense classification layer without destroying pre-trained ImageNet representations.
- **Stage 1 Best Validation Accuracy:** 96.40% (Validation Loss: 0.1098)

### Stage 2: Fine-Tuning (Partial Unfreezing)
- **Backbone Status:** Partially unfrozen from layer index 100 through layer 154 (`base_model.layers[100:]`). The initial 99 layers remained permanently frozen to preserve basic edge and color filter primitives.
- **Optimizer:** Adam
- **Learning Rate:** $\eta = 10^{-5}$ (reduced by two orders of magnitude to avoid destabilizing pre-trained weights)
- **Epochs Run:** 4 epochs (stopped via EarlyStopping callback monitoring `val_loss` with patience = 3)
- **Stage 2 Best Validation Loss:** 0.100062 (Validation Accuracy: 96.33%)

---

## 4. Preprocessing & Data Augmentation Pipeline

### Preprocessing Pipeline (Identical for Training, Validation, and Inference):
1. **Decode:** Image file decoded to uint8 RGB array (discarding alpha channel if present).
2. **Resize:** Target spatial resolution resized to `224 x 224` pixels using bilinear interpolation.
3. **Normalization:** Pixel intensity values scaled linearly from $[0, 255]$ to $[0.0, 1.0]$ via $x_{norm} = \frac{x}{255.0}$.

### Data Augmentation (Applied Strictly to Training Set Only):
To maximize generalization across variable agricultural photography conditions without introducing spatial distortions:
- **Random Horizontal Flip:** `p = 0.5` (simulates alternate leaf orientations)
- **Random Brightness Adjustment:** $\pm 10\%$ factor (simulates differing daylight conditions)
- **Random Contrast Adjustment:** $\pm 10\%$ factor (simulates camera exposure variations)
- *Note:* No rotation or shearing was applied to avoid distorting delicate lesion edge geometries.

---

## 5. Loss Function, Optimizer & Class Balancing

- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$)
- **Class Balancing:** Class weights were dynamically computed inversely proportional to class frequencies to counteract class imbalance (e.g., Potato healthy had significantly fewer samples than Tomato Bacterial Spot):
  $$w_j = \frac{N}{C \times n_j}$$
  where $N$ is total training samples, $C=13$ classes, and $n_j$ is sample count for class $j$.

---

## 6. Supported Class Taxonomy (13 Classes)

| Class Index | Formal Trained Identifier | Crop | Disease Status | Common Display Name |
|---|---|---|---|---|
| `0` | `Apple___Apple_scab` | Apple | Diseased | Apple Scab (*Venturia inaequalis*) |
| `1` | `Apple___Black_rot` | Apple | Diseased | Apple Black Rot (*Botryosphaeria obtusa*) |
| `2` | `Apple___healthy` | Apple | Healthy | Apple Healthy |
| `3` | `Corn_(maize)___Common_rust_` | Corn (Maize) | Diseased | Common Rust (*Puccinia sorghi*) |
| `4` | `Corn_(maize)___Northern_Leaf_Blight` | Corn (Maize) | Diseased | Northern Leaf Blight (*Exserohilum turcicum*) |
| `5` | `Corn_(maize)___healthy` | Corn (Maize) | Healthy | Corn (Maize) Healthy |
| `6` | `Potato___Early_blight` | Potato | Diseased | Potato Early Blight (*Alternaria solani*) |
| `7` | `Potato___Late_blight` | Potato | Diseased | Potato Late Blight (*Phytophthora infestans*) |
| `8` | `Potato___healthy` | Potato | Healthy | Potato Healthy |
| `9` | `Tomato___Bacterial_spot` | Tomato | Diseased | Tomato Bacterial Spot (*Xanthomonas campestris*) |
| `10` | `Tomato___Early_blight` | Tomato | Diseased | Tomato Early Blight (*Alternaria solani*) |
| `11` | `Tomato___Late_blight` | Tomato | Diseased | Tomato Late Blight (*Phytophthora infestans*) |
| `12` | `Tomato___healthy` | Tomato | Healthy | Tomato Healthy |

---

## 7. Inference Pipeline & Runtime Performance

1. **Singleton In-Memory Cache:** Model is preloaded during application startup into memory (`model_service.py`), eliminating 1.5–2.5 second cold load times per request.
2. **CPU Execution:** Inference executes on standard CPU architecture without requiring specialized GPU infrastructure.
3. **Benchmark Latency:**
   - Mean Inference Latency: **108.56 ms**
   - Median Inference Latency: **104.12 ms**
   - Standard Deviation: 13.51 ms
   - Throughput: ~9.2 frames per second (FPS) on standard host CPU.
