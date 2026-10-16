# PlantVillage Dataset Inspection & Characterization Report

## 1. Dataset Name
**PlantVillage Dataset (Raw Color Imagery Subset)**

## 2. Source
- **Official GitHub Repository:** [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Directory Acquired:** `raw/color/`
- **Seminal Reference Publication:**  
  Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science, 7, 1419. [DOI: 10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)

## 3. License Information
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- **Permissions:** Unrestricted academic and non-commercial research, sharing, and adaptation with appropriate attribution.

## 4. Dataset Purpose
To benchmark computer vision and deep learning models on plant pathology identification by categorizing crop leaf imagery into distinct taxonomic classes, distinguishing healthy leaves from specific foliar diseases.

---

## 5. Actual Dataset Statistics (Programmatically Verified)

- **Total Classes:** 38
- **Total Files Scanned:** 54,294
- **Total Valid Images:** 54,294
- **Corrupted / Unreadable Images:** 0 (100% passed Pillow container header verification)
- **Unsupported Files:** 0
- **Exact Cryptographic Duplicates (MD5):** 21 files detected
- **Image Dimensions:** $256 \times 256$ pixels (54,294 images, 100%)
- **Image Formats:**
  - `.jpg`: 54,292 images
  - `.png`: 1 image
  - `.jpeg`: 1 image
- **Color Channel Modes:**
  - `RGB` (3 Channels): 54,293 images
  - `RGBA` (4 Channels): 1 image

---

## 6. Complete Class Distribution (All 38 Classes)

| # | Exact Class Name | Valid Images | Crop Category | Health State |
| :-: | :--- | :-: | :--- | :--- |
| 1 | `Apple___Apple_scab` | 630 | Apple | Diseased |
| 2 | `Apple___Black_rot` | 621 | Apple | Diseased |
| 3 | `Apple___Cedar_apple_rust` | 275 | Apple | Diseased |
| 4 | `Apple___healthy` | 1,645 | Apple | Healthy |
| 5 | `Blueberry___healthy` | 1,502 | Blueberry | Healthy |
| 6 | `Cherry_(including_sour)___Powdery_mildew` | 1,052 | Cherry | Diseased |
| 7 | `Cherry_(including_sour)___healthy` | 854 | Cherry | Healthy |
| 8 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 513 | Corn (Maize) | Diseased |
| 9 | `Corn_(maize)___Common_rust_` | 1,192 | Corn (Maize) | Diseased |
| 10 | `Corn_(maize)___Northern_Leaf_Blight` | 985 | Corn (Maize) | Diseased |
| 11 | `Corn_(maize)___healthy` | 1,162 | Corn (Maize) | Healthy |
| 12 | `Grape___Black_rot` | 1,180 | Grape | Diseased |
| 13 | `Grape___Esca_(Black_Measles)` | 1,383 | Grape | Diseased |
| 14 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 1,076 | Grape | Diseased |
| 15 | `Grape___healthy` | 423 | Grape | Healthy |
| 16 | `Orange___Haunglongbing_(Citrus_greening)` | 5,507 | Orange | Diseased |
| 17 | `Peach___Bacterial_spot` | 2,295 | Peach | Diseased |
| 18 | `Peach___healthy` | 360 | Peach | Healthy |
| 19 | `Pepper,_bell___Bacterial_spot` | 996 | Bell Pepper | Diseased |
| 20 | `Pepper,_bell___healthy` | 1,478 | Bell Pepper | Healthy |
| 21 | `Potato___Early_blight` | 998 | Potato | Diseased |
| 22 | `Potato___Late_blight` | 999 | Potato | Diseased |
| 23 | `Potato___healthy` | 152 | Potato | Healthy |
| 24 | `Raspberry___healthy` | 371 | Raspberry | Healthy |
| 25 | `Soybean___healthy` | 5,089 | Soybean | Healthy |
| 26 | `Squash___Powdery_mildew` | 1,834 | Squash | Diseased |
| 27 | `Strawberry___Leaf_scorch` | 1,109 | Strawberry | Diseased |
| 28 | `Strawberry___healthy` | 456 | Strawberry | Healthy |
| 29 | `Tomato___Bacterial_spot` | 2,126 | Tomato | Diseased |
| 30 | `Tomato___Early_blight` | 1,000 | Tomato | Diseased |
| 31 | `Tomato___Late_blight` | 1,908 | Tomato | Diseased |
| 32 | `Tomato___Leaf_Mold` | 952 | Tomato | Diseased |
| 33 | `Tomato___Septoria_leaf_spot` | 1,771 | Tomato | Diseased |
| 34 | `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | Tomato | Diseased |
| 35 | `Tomato___Target_Spot` | 1,404 | Tomato | Diseased |
| 36 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | Tomato | Diseased |
| 37 | `Tomato___Tomato_mosaic_virus` | 373 | Tomato | Diseased |
| 38 | `Tomato___healthy` | 1,590 | Tomato | Healthy |

---

## 7. Distribution & Imbalance Observations
- **Smallest Class:** `Potato___healthy` (152 images).
- **Largest Class:** `Orange___Haunglongbing_(Citrus_greening)` (5,507 images).
- **Overall Imbalance Ratio:** $36.23 : 1$.
- **Mean Images per Class:** $1,428.79$.
- **Imbalance Impact:** Training all 38 classes without mitigation could lead to minority class starvation. Stratification or weighted loss is required if all classes are trained.

---

## 8. Duplication & Data Leakage Assessment
- **Cryptographic Hash Detection:** 21 exact byte-for-byte binary duplicates detected by MD5 hashing across the entire collection.
- **Visual Burst Redundancy:** Certain leaves were captured under uniform studio conditions with minimal angular displacement (burst capture). 
- **Leakage Prevention Requirement:** In Phase 3, deduplication and strict grouped/stratified splitting must prevent identical samples from spanning both the training and validation/test splits.

---

## 9. Proposed Project Scope Subset (Recommendation for College Project)

To balance academic rigor, training convergence, and local compute resources, we propose an initial focused subset of **4 staple crops** spanning **13 classes**:

| Selected Class | Crop | Disease Name | Image Count | Reason for Inclusion |
| :--- | :--- | :--- | :---: | :--- |
| `Potato___Early_blight` | Potato | Early Blight (*A. solani*) | 998 | Major global solanaceous crop disease |
| `Potato___Late_blight` | Potato | Late Blight (*P. infestans*) | 999 | Historical, severe agrarian pathogen |
| `Potato___healthy` | Potato | Healthy Foliage Baseline | 152 | Essential control state for Potato |
| `Tomato___Early_blight` | Tomato | Early Blight (*A. solani*) | 1,000 | Clinical counterpart to potato early blight |
| `Tomato___Late_blight` | Tomato | Late Blight (*P. infestans*) | 1,908 | Highly prevalent fungal foliar disease |
| `Tomato___Bacterial_spot` | Tomato | Bacterial Spot (*X. campestris*) | 2,126 | Representative bacterial leaf infection |
| `Tomato___healthy` | Tomato | Healthy Foliage Baseline | 1,590 | Essential control state for Tomato |
| `Corn_(maize)___Common_rust_` | Corn | Common Rust (*P. sorghi*) | 1,192 | Distinct pustule morphology on monocot leaf |
| `Corn_(maize)___Northern_Leaf_Blight` | Corn | Northern Leaf Blight (*E. turcicum*) | 985 | Distinct elliptical lesion morphology |
| `Corn_(maize)___healthy` | Corn | Healthy Foliage Baseline | 1,162 | Essential control state for Corn cereal |
| `Apple___Apple_scab` | Apple | Apple Scab (*V. inaequalis*) | 630 | Major commercial orchard pathogen |
| `Apple___Black_rot` | Apple | Black Rot (*B. obtusa*) | 621 | Frogeye foliar symptom pattern |
| `Apple___healthy` | Apple | Healthy Foliage Baseline | 1,645 | Essential control state for fruit trees |
| **Total Proposed Subset** | **4 Crops** | **9 Diseases + 4 Healthy** | **15,008** | **Balanced across 4 diverse agricultural sectors** |

### Classes Not Selected for Initial Scope:
The remaining 25 classes (including Grape, Peach, Bell Pepper, Citrus, Soybean, Squash, Strawberry, Raspberry, Blueberry, and additional Tomato variants) remain preserved in `ml/data/raw/plantvillage/` and are not deleted.

---

## 10. Known Dataset Limitations
1. **Studio Background Uniformity:** Most leaves were detached and photographed against plain or gray backgrounds, which differs from field conditions (dirt, complex weeds, direct sunlight).
2. **Crop Imbalance:** Healthy classes for certain crops (e.g., Potato with 152 images) are significantly smaller than their disease classes (~1,000 images).
3. **Single Leaf Focus:** Images contain isolated single leaves rather than full-canopy views.
