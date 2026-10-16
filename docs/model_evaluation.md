# Phase 5 -- Model Evaluation, Confusion Matrix, and Error Analysis Specification

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Document:** Test Dataset Evaluation Specification & Empirical Findings  
**Phase:** 5 (Rigorous Evaluation & Error Analysis)  
**Status:** Completed  

---

## 1. Evaluation Objective

To evaluate the generalization performance of the trained `MobileNetV2` model against the held-out test split created in Phase 3. The test dataset remained strictly quarantined and unobserved throughout all training, callback triggers, and hyperparameter tuning in Phase 4. All reported metrics represent un-augmented, deterministic inference outputs.

---

## 2. Test Dataset Description

- **Manifest File:** `ml/data/splits/test.csv`
- **Total Test Samples:** 1,499 images (exactly 10.0% of the 14,987 total usable dataset)
- **Data Source:** PlantVillage crop pathology dataset
- **Crops Evaluated:** Apple, Corn (Maize), Potato, Tomato (13 distinct pathology and healthy states)
- **Integrity Validation:** 
  - 100% of the 1,499 test images exist on disk and were readable without error.
  - Zero cryptographic hash overlap exists between `train.csv` and `test.csv` (0 duplicate images).
  - All labels strictly match the canonical `ml/data/class_indices.json`.

---

## 3. Model Evaluated

- **Architecture:** `MobileNetV2` (two-stage transfer learning: ImageNet weights + frozen base feature extractor + partially fine-tuned top 54 layers)
- **Model Checkpoint:** `ml/models/best_model.keras`
- **Input Dimensions:** `224 × 224 × 3` RGB format
- **Output Neurons:** 13 (Softmax multi-class activation)

---

## 4. Preprocessing

Every test image underwent identical, deterministic preprocessing without augmentation:
1. Native JPEG bytes decoded as 3-channel RGB.
2. Bilinear resize to `224 × 224` resolution.
3. Linear normalization: pixel values cast to `float32` and scaled to `[0.0, 1.0]` by dividing by `255.0`.

---

## 5. Evaluation Methodology

- **Batch Size:** 32 images per batch (47 batches total).
- **Execution Mode:** `model(batch, training=False)` inference mode, ensuring Batch Normalization layers operate strictly on frozen population statistics.
- **Decision Rule:** Argmax across the 13 softmax probability scores.
- **Confidence Metric:** The predicted class's softmax probability $\max_k(p_k)$. Note: Confidence is reported as the model's posterior probability, not guaranteed statistical calibration.

---

## 6. Overall Performance Metrics

| Metric | Raw Value | Percentage / Summary |
|:---|:---:|:---:|
| **Total Test Images** | 1,499 | 100.0% |
| **Correct Classifications** | 1,445 | 96.40% |
| **Misclassifications** | 54 | 3.60% |
| **Overall Test Accuracy** | **0.963976** | **96.40%** |
| **Macro Precision** | 0.945470 | 94.55% |
| **Macro Recall** | 0.969177 | 96.92% |
| **Macro F1-Score** | **0.954739** | **95.47%** |
| **Weighted Precision** | 0.965708 | 96.57% |
| **Weighted Recall** | 0.963976 | 96.40% |
| **Weighted F1-Score** | **0.964173** | **96.42%** |
| **One-vs-Rest ROC AUC (Micro)** | 0.9995 | 99.95% |
| **One-vs-Rest ROC AUC (Macro)** | 0.9991 | 99.91% |

### Macro vs. Weighted Averaging
- **Macro Average:** Computes the arithmetic mean of metrics across all 13 classes equally, regardless of class sample size. The high Macro F1 (95.47%) demonstrates that minority classes (such as `Potato___healthy` with only 15 test samples) are classified effectively and not obscured by larger classes.
- **Weighted Average:** Weights the per-class metric by its class support. At 96.42% F1, it reflects system reliability across the operational test distribution.

---

## 7. Per-Class Performance Breakdown

Generated from actual model predictions and saved to `ml/evaluation/classification_report.csv`:

| Class Name | Class Index | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|:---:|
| `Apple___Apple_scab` | 0 | 0.9692 | 1.0000 | 0.9844 | 63 |
| `Apple___Black_rot` | 1 | 0.9841 | 1.0000 | 0.9920 | 62 |
| `Apple___healthy` | 2 | 0.9938 | 0.9817 | 0.9877 | 164 |
| `Corn_(maize)___Common_rust_` | 3 | 1.0000 | 1.0000 | 1.0000 | 119 |
| `Corn_(maize)___Northern_Leaf_Blight` | 4 | 1.0000 | 0.9796 | 0.9897 | 98 |
| `Corn_(maize)___healthy` | 5 | 0.9831 | 1.0000 | 0.9915 | 116 |
| `Potato___Early_blight` | 6 | 1.0000 | 1.0000 | 1.0000 | 100 |
| `Potato___Late_blight` | 7 | 0.9381 | 0.9100 | 0.9239 | 100 |
| `Potato___healthy` | 8 | 0.6818 | 1.0000 | 0.8108 | 15 |
| `Tomato___Bacterial_spot` | 9 | 0.9591 | 0.9906 | 0.9746 | 213 |
| `Tomato___Early_blight` | 10 | 0.8286 | 0.8700 | 0.8488 | 100 |
| `Tomato___Late_blight` | 11 | 0.9595 | 0.8737 | 0.9146 | 190 |
| `Tomato___healthy` | 12 | 0.9937 | 0.9937 | 0.9937 | 159 |

---

## 8. Confusion Matrix Analysis

- **Machine-readable Matrix:** Saved to `ml/evaluation/confusion_matrix.csv`
- **Annotated Visual Heatmap:** Saved to `ml/evaluation/confusion_matrix.png`

### Observed Diagnostic Observations
1. **Perfect Diagonal Alignment:** Classes such as `Corn_(maize)___Common_rust_` (119/119) and `Potato___Early_blight` (100/100) attained 100% precision and 100% recall.
2. **Apple Subsystem Performance:** 288 out of 289 apple samples were classified correctly. Only 1 apple sample had an error (`Apple___healthy` confused as `Potato___healthy`).
3. **Tomato Subsystem Complexity:** The majority of misclassifications occurred within the Solanaceae family (Tomato and Potato blights).

---

## 9. Error Analysis & Frequent Confusion Pairs

Total misclassified images: **54** out of 1,499 (Error rate: 3.60%).  
Exported to: `ml/evaluation/misclassified_predictions.csv`

### Most Frequent Confusion Pairs
1. **Actual `Tomato___Late_blight` $\rightarrow$ Predicted `Tomato___Early_blight` (15 instances):**
   - *Observation:* Both fungal/oomycete blights produce necrotic dark-brown concentric spots on tomato leaflets. At intermediate lesion progression, macroscopic visual patterns strongly overlap.
2. **Actual `Tomato___Early_blight` $\rightarrow$ Predicted `Tomato___Bacterial_spot` (8 instances):**
   - *Observation:* Early-stage bacterial spot produces small water-soaked lesions that visually resemble initial early blight spots before halo development.
3. **Actual `Tomato___Late_blight` $\rightarrow$ Predicted `Potato___Late_blight` (6 instances):**
   - *Observation:* Both diseases are caused by the identical pathogen (*Phytophthora infestans*) on closely related nightshade host plants, exhibiting similar foliar lesions.
4. **Actual `Potato___Late_blight` $\rightarrow$ Predicted `Potato___healthy` (5 instances):**
   - *Observation:* Leaf images displaying minimal localized marginal necrosis where the majority of the leaf area remains green.
5. **Actual `Tomato___Early_blight` $\rightarrow$ Predicted `Tomato___Late_blight` (4 instances):**
   - *Observation:* Cross-confusion between early and late blight phases.

---

## 10. Confidence Profile Analysis

- **High-Confidence Incorrect Predictions (Overconfident Errors):**
  - Image `eb2537b2-d6fa-4c70-b919-f90bbe2c65f2___RS_Erly.B 7605.JPG`: Actual `Tomato___Early_blight` $\rightarrow$ Predicted `Tomato___Bacterial_spot` with 99.5% confidence.
  - Image `93d91b96-e024-477c-9b76-796e3f1614d4___RS_LB 3297.JPG`: Actual `Potato___Late_blight` $\rightarrow$ Predicted `Tomato___Late_blight` with 98.4% confidence.
  - *Implication:* Standard softmax layers can produce overconfident predictions on ambiguous border cases. Grad-CAM (Phase 6) and threshold gating will help identify when features overlap.
- **Low-Confidence Incorrect Predictions (Uncertain Errors):**
  - Confidences ranged between 42.1% and 45.9% on ambiguous images (e.g., `Tomato___Late_blight` vs `Potato___Late_blight` at 42.1% confidence).
  - *Implication:* Low confidence flags can be utilized in the future backend to prompt the user for an additional leaf angle or higher lighting.

---

## 11. Inference Latency Benchmark

Benchmarked over 100 single-image predictions (`batch_size = 1`) on the local host environment:

- **Host Processor:** x86_64 CPU (AVX2 enabled)
- **Mean Single-Image Latency:** **108.56 ms**
- **Median Single-Image Latency:** **104.12 ms**
- **Standard Deviation:** 13.51 ms
- **Min / Max Latency:** 95.34 ms / 162.31 ms
- **Throughput:** ~**9.2 images/second** on single CPU core execution
- **Batch Throughput:** ~**97.0 images/second** (when processed with batch size 32)

---

## 12. Evaluation Artifacts Summary

| Artifact Name | File Path | Content Description |
|:---|:---|:---|
| **Test Predictions CSV** | `ml/evaluation/test_predictions.csv` | Full row-level predictions for all 1,499 test images |
| **Misclassified CSV** | `ml/evaluation/misclassified_predictions.csv` | Isolated records for the 54 incorrect predictions |
| **Classification Report CSV** | `ml/evaluation/classification_report.csv` | Precision, Recall, F1-Score, Support for all 13 classes |
| **Classification Report TXT** | `ml/evaluation/classification_report.txt` | Standard scikit-learn textual classification summary |
| **Confusion Matrix CSV** | `ml/evaluation/confusion_matrix.csv` | 13x13 numerical contingency matrix |
| **Confusion Matrix Plot** | `ml/evaluation/confusion_matrix.png` | 200 DPI annotated heatmap visualization |
| **Multiclass ROC Curves** | `ml/evaluation/roc_curves.png` | One-vs-Rest ROC curves with micro/macro AUC |
| **Sample Grid Plot** | `ml/evaluation/sample_predictions/sample_grid.png` | Visual grid of representative correct and incorrect cases |
| **Sample Grid JSON** | `ml/evaluation/sample_predictions/samples.json` | Metadata for displayed sample predictions |
| **Master Summary JSON** | `ml/evaluation/evaluation_summary.json` | Comprehensive machine-readable metrics summary |
| **Evaluation Script** | `ml/evaluation/evaluate_model.py` | Fully reproducible evaluation test runner |

---

## 13. Known Limitations & Recommendations

1. **Laboratory vs. Field Bias:** PlantVillage images are captured on uniform, plain laboratory backgrounds under controlled illumination. Field photographs with soil, weeds, shadows, or dew drops may introduce domain shift.
2. **Phytophthora Confusion:** Folier symptoms of *Phytophthora infestans* across Tomato and Potato exhibit biological similarities that occasionally cross host-crop boundaries in visual inspection alone.
3. **Explainability Requirement:** High classification confidence on subtle lesions necessitates visual localization (Grad-CAM in Phase 6) to verify that predictions stem from genuine leaf pathology rather than background artifacts.
