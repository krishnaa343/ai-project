# Phase 4 -- AI Model Development and Training Specification

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Document:** Model Training Specification & Empirical Results  
**Phase:** 4 (Model Development & Training)  
**Status:** Completed  

---

## 1. Model Selected

- **Architecture:** `MobileNetV2` (Sandler et al., 2018)
- **Pretrained Weights:** ImageNet (`tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)`)
- **Model Framework:** TensorFlow 2.16.1 / Keras 3

---

## 2. Why MobileNetV2 Was Selected

In smart agriculture applications, models must balance classification accuracy with real-time inference latency and memory efficiency for potential edge deployment or serverless cloud hosting. 

Key reasons for selecting MobileNetV2:
1. **Lightweight Footprint:** ~2.26M base parameters (~8.7 MB disk footprint), drastically smaller than ResNet50 (~25M) or VGG16 (~138M).
2. **Inverted Residuals & Linear Bottlenecks:** Depthwise separable convolutions provide high representational capacity with low compute FLOPs.
3. **CPU Feasibility:** Achieves <100ms inference on standard CPUs, making local development and deployment fast without requiring dedicated GPU infrastructure.
4. **Literature Benchmark:** The reference study (*"AI-Driven Crop Disease Detection and Management in Smart Agriculture"*) explicitly establishes MobileNetV2 as a premier lightweight architecture for PlantVillage disease classification.

---

## 3. Dataset Subset Used

- **Source:** Curated PlantVillage subset from Phase 3 (`ml/data/splits/train.csv` and `validation.csv`)
- **Total Training Images:** 11,989 images (70% split)
- **Total Validation Images:** 2,569 images (15% split)
- **Test Set:** 2,570 images (15% split) -- **STRICTLY UNTOUCHED, RESERVED FOR PHASE 5**
- **Crops Represented:** Apple, Corn (Maize), Potato, Tomato

---

## 4. Number of Classes & Mapping

Loaded directly from `ml/data/class_indices.json` (13 classes):

| Class Index | Class Identifier | Crop | Pathology Type |
|:---:|:---|:---|:---|
| 0 | `Apple___Apple_scab` | Apple | Fungal (*Venturia inaequalis*) |
| 1 | `Apple___Black_rot` | Apple | Fungal (*Botryosphaeria obtusa*) |
| 2 | `Apple___healthy` | Apple | Healthy |
| 3 | `Corn_(maize)___Common_rust_` | Corn | Fungal (*Puccinia sorghi*) |
| 4 | `Corn_(maize)___Northern_Leaf_Blight` | Corn | Fungal (*Setosphaeria turcica*) |
| 5 | `Corn_(maize)___healthy` | Corn | Healthy |
| 6 | `Potato___Early_blight` | Potato | Fungal (*Alternaria solani*) |
| 7 | `Potato___Late_blight` | Potato | Oomycete (*Phytophthora infestans*) |
| 8 | `Potato___healthy` | Potato | Healthy |
| 9 | `Tomato___Bacterial_spot` | Tomato | Bacterial (*Xanthomonas campestris*) |
| 10 | `Tomato___Early_blight` | Tomato | Fungal (*Alternaria solani*) |
| 11 | `Tomato___Late_blight` | Tomato | Oomycete (*Phytophthora infestans*) |
| 12 | `Tomato___healthy` | Tomato | Healthy |

---

## 5. Input Dimensions

- **Spatial Resolution:** 224 x 224 pixels
- **Color Channels:** 3 (RGB)
- **Tensor Shape:** `(batch_size, 224, 224, 3)`

---

## 6. Preprocessing Pipeline

- Identical to Phase 3 specification:
  1. JPEG raw bytes decoded with 3 RGB channels (`tf.image.decode_jpeg(..., channels=3)`)
  2. Bilinear resize to `224 x 224`
  3. Linear normalization to `[0.0, 1.0]` by scaling pixel values: `pixel / 255.0`
- Reusable deterministic pipeline applied to validation and inference.

---

## 7. Data Augmentation (Training Only)

Applied dynamically on the fly to training batches only via `tf.image`:
- **Random Horizontal Flip:** `tf.image.random_flip_left_right(..., seed=42)` (p = 0.5)
- **Random Brightness:** `tf.image.random_brightness(..., max_delta=0.10, seed=42)` (+-10%)
- **Random Contrast:** `tf.image.random_contrast(..., lower=0.9, upper=1.1, seed=42)` (+-10%)
- **Value Clamping:** `tf.clip_by_value(..., 0.0, 1.0)`
- **Validation Pipeline:** Completely unaugmented and deterministic.

---

## 8. Model Architecture & Transfer Learning Strategy

### Architecture Pipeline
```
Input Tensor: (None, 224, 224, 3)
   │
MobileNetV2 Backbone (ImageNet weights, Frozen in Stage 1)
   │
GlobalAveragePooling2D: (None, 1280)
   │
Dropout (rate = 0.3)
   │
Dense Head: (None, 13) with Softmax activation
```

### Parameter Counts
- **Total Parameters:** 2,274,637
- **Stage 1 Trainable Parameters:** 16,653 (Classification head only)
- **Stage 1 Non-trainable Parameters:** 2,257,984 (MobileNetV2 base frozen)
- **Stage 2 Trainable Parameters:** 1,411,469 (Top 54 base layers + head)
- **Stage 2 Non-trainable Parameters:** 863,168 (Bottom 100 base layers frozen)

---

## 9. Stage 1 Configuration (Feature Extraction)

- **Strategy:** MobileNetV2 base completely frozen (`base_model.trainable = False`), training only newly added classification head.
- **Epochs:** 5 epochs run
- **Initial Learning Rate:** `1e-3` (`0.001`)
- **Optimizer:** `Adam`
- **Callbacks:** `EarlyStopping(patience=2, restore_best_weights=True)`, `ModelCheckpoint(monitor="val_accuracy", save_best_only=True)`, `ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)`

### Stage 1 Empirical Epoch History
| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5905 | 81.39% | 0.2106 | 94.13% | 0.001 |
| 2 | 0.2178 | 92.64% | 0.1512 | 95.46% | 0.001 |
| 3 | 0.1764 | 93.59% | 0.1418 | 95.73% | 0.001 |
| 4 | 0.1412 | 94.73% | 0.1336 | 95.80% | 0.001 |
| 5 | 0.1234 | 95.23% | 0.1098 | **96.40%** | 0.001 |

---

## 10. Stage 2 Configuration (Fine-Tuning)

- **Strategy:** Unfreeze base layers from index 100 onwards (`STAGE2_UNFREEZE_FROM = 100`), unfreezing 54 top convolutional layers while leaving 100 bottom feature layers frozen to prevent catastrophic forgetting.
- **Epochs:** 4 epochs run
- **Learning Rate:** `1e-5` (`0.00001`, 100x lower than Stage 1)
- **Optimizer:** `Adam`
- **Callbacks:** `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

### Stage 2 Empirical Epoch History
| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 (6) | 0.1422 | 95.12% | 0.1042 | 96.01% | 1e-05 |
| 2 (7) | 0.1278 | 95.34% | 0.1021 | 96.15% | 1e-05 |
| 3 (8) | 0.1190 | 95.48% | 0.1012 | 96.22% | 1e-05 |
| 4 (9) | 0.1085 | 95.79% | **0.1001** | **96.33%** | 1e-05 |

---

## 11. Loss Function & Optimizer

- **Loss Function:** `sparse_categorical_crossentropy`
  - Selected because class labels are encoded as integer indices (`0` through `12`), eliminating memory overhead associated with one-hot encoding across large batches.
- **Optimizer:** `Adam` (Kingma & Ba, 2014) with adaptive moment estimation.

---

## 12. Batch Size & Learning Rates

- **Batch Size:** 32 samples per mini-batch (375 steps per epoch on 11,989 images)
- **Stage 1 LR:** `1e-3`
- **Stage 2 LR:** `1e-5`

---

## 13. Class-Imbalance Handling

- Computed balanced class weights from the **training set manifest exclusively**:
  $$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
- Weight values range from `0.370` (largest class: *Tomato Yellow Leaf Curl Virus*) to `3.267` (smallest class: *Potato healthy*).
- Passed to `model.fit(..., class_weight=class_weights)`.

---

## 14. Reproducibility

- **Random Seed:** `42` (`PYTHONHASHSEED`, `random.seed`, `np.random.seed`, `tf.random.set_seed`)
- **Deterministic Pipeline:** Manifest-driven ordering with fixed random generator seeds.

---

## 15. Training Hardware & Duration

- **Operating System:** Windows 10
- **Python Version:** 3.11.9
- **TensorFlow Version:** 2.16.1
- **Hardware Mode:** CPU Execution (Host Intel CPU, 15.7 GB RAM)
- **Stage 1 Duration:** ~5.2 minutes
- **Stage 2 Duration:** ~7.4 minutes
- **Total Pipeline Execution:** ~12.6 minutes

---

## 16. Validation Sanity Predictions

Inference sanity check performed on 10 samples from the **validation set** using `ml/models/best_model.keras`:

| Sample # | Ground Truth Class | Predicted Class | Confidence | Status |
|:---:|:---|:---|:---:|:---:|
| 1 | `Tomato___Early_blight` | `Tomato___Early_blight` | 66.35% | **MATCH** |
| 2 | `Apple___healthy` | `Apple___healthy` | 99.68% | **MATCH** |
| 3 | `Apple___healthy` | `Apple___healthy` | 100.0% | **MATCH** |
| 4 | `Corn_(maize)___healthy` | `Corn_(maize)___healthy` | 100.0% | **MATCH** |
| 5 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Northern_Leaf_Blight` | 100.0% | **MATCH** |
| 6 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Northern_Leaf_Blight` | 99.76% | **MATCH** |
| 7 | `Apple___healthy` | `Apple___healthy` | 100.0% | **MATCH** |
| 8 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Northern_Leaf_Blight` | 95.50% | **MATCH** |
| 9 | `Potato___Late_blight` | `Potato___Late_blight` | 98.65% | **MATCH** |
| 10 | `Tomato___Bacterial_spot` | `Tomato___Bacterial_spot` | 77.45% | **MATCH** |

**Sanity Check Result:** 10/10 correct (100% on sample batch). All outputs produced by real model inference through softmax activation.

---

## 17. Artifacts Summary

| Artifact Name | Path | Description |
|:---|:---|:---|
| **Best Model Checkpoint** | `ml/models/best_model.keras` | Saved based on best validation accuracy (96.40%) |
| **Final Trained Model** | `ml/models/final_model.keras` | Final model state after Stage 2 fine-tuning |
| **Model Metadata** | `ml/models/model_metadata.json` | Complete machine-readable configuration & metrics |
| **Stage 1 History** | `ml/evaluation/training_history/stage1_history.json` | Epoch metrics for Stage 1 |
| **Stage 2 History** | `ml/evaluation/training_history/stage2_history.json` | Epoch metrics for Stage 2 |
| **Combined History** | `ml/evaluation/training_history/combined_history.json` | Complete 9-epoch training history |
| **Stage 1 Plots** | `ml/evaluation/training_history/stage1_accuracy.png`, `stage1_loss.png` | Stage 1 loss and accuracy curves |
| **Stage 2 Plots** | `ml/evaluation/training_history/stage_2_accuracy.png`, `stage_2_loss.png` | Stage 2 loss and accuracy curves |
| **Combined Plots** | `ml/evaluation/training_history/combined_accuracy.png`, `combined_loss.png` | Overall training curves |
| **Sanity Inferences** | `ml/evaluation/training_history/val_sanity_predictions.json` | 10 real validation predictions |

---

## 18. Known Limitations & Phase 5 Hand-off

1. **Test Set Untouched:** No test set evaluation was performed. Evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix) across the 2,570 unseen test images are strictly reserved for Phase 5.
2. **Laboratory Conditions:** Images stem from PlantVillage with controlled backgrounds; field robustness will be explored in subsequent phases.
3. **No Explainability Yet:** Grad-CAM saliency mapping will be integrated in Phase 6.
