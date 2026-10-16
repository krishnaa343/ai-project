# Data Preprocessing, Stratified Splitting, and Augmentation Specification

## 1. Dataset Source
- **Dataset:** PlantVillage Dataset (Raw RGB Color Imagery).
- **Origin:** Official author repository (`spMohanty/PlantVillage-Dataset`), directory `raw/color/`.
- **Location on Disk:** `ml/data/raw/plantvillage/` (Preserved 100% unaltered).

---

## 2. Selected Classes (13 Classes Across 4 Crops)

The initial scope covers 4 major staple agricultural crops:

1. `Apple___Apple_scab`
2. `Apple___Black_rot`
3. `Apple___healthy`
4. `Corn_(maize)___Common_rust_`
5. `Corn_(maize)___Northern_Leaf_Blight`
6. `Corn_(maize)___healthy`
7. `Potato___Early_blight`
8. `Potato___Late_blight`
9. `Potato___healthy`
10. `Tomato___Bacterial_spot`
11. `Tomato___Early_blight`
12. `Tomato___Late_blight`
13. `Tomato___healthy`

Configured in `ml/data/selected_classes.json`.

---

## 3. Selection Rationale
- **Agronomic Diversity:** Balances root vegetables (Potato), nightshades/fruiting crops (Tomato), monocot cereals (Corn), and orchard trees (Apple).
- **Paired Health Controls:** Every single crop includes its corresponding `healthy` baseline to prevent model false positives on non-diseased leaves.
- **Compute Feasibility:** 14,987 images offer ample statistical depth for deep transfer learning (MobileNetV2 / EfficientNet) while maintaining rapid training epoch turnaround on standard hardware.

---

## 4. Data Cleaning Procedure
- **Scanning:** Every file in each selected class folder was inspected for format, decode validity, and pixel dimension compatibility.
- **Unsupported Files:** 0 non-image files encountered.
- **Raw Data Preservation:** The raw files are never deleted or modified; unselected or duplicate images are omitted strictly at the manifest level.

---

## 5. Corruption Handling
- **Verification Method:** Two-tier PIL decoding test (`img.verify()` for container headers followed by `img.convert("RGB")` for pixel matrix integrity).
- **Corrupted Images Found:** 0 corrupted images across the selected classes.

---

## 6. Duplicate Handling
- **Detection Algorithm:** Cryptographic MD5 hash calculation on full file byte content.
- **Findings:** Exactly 21 exact binary duplicates detected within the 15,008 raw images of the selected classes.
- **Deduplication Strategy:** The first occurrence of each unique MD5 hash was retained in the manifest; subsequent 21 duplicates were excluded from manifests and logged in `ml/data/splits/excluded_files.json`.
- **Important Distinction:** MD5 detects exact bit-level binary clones. It does not eliminate near-duplicate images captured in burst-mode photography under similar studio angles.

---

## 7. Train / Validation / Test Ratios & Partitioning
- **Target Ratios:** 80% Train, 10% Validation, 10% Test.
- **Strategy:** Two-stage stratified splitting using scikit-learn (`stratify=class_index`).
  - Stage 1: Split full dataset (14,987 unique images) into 80% Train (`11,989` images) and 20% Temporary (`2,998` images).
  - Stage 2: Split Temporary 50/50 into Validation (`1,499` images, 10.0%) and Test (`1,499` images, 10.0%).

---

## 8. Random Seed
- **Fixed Seed Value:** `42`
- Guarantees exact, deterministic reproduction of all splits across runs and environments.

---

## 9. Image Resolution & Dimension Standardization
- **Raw Input Resolution:** $256 \times 256 \times 3$ RGB.
- **Standardized Pipeline Output:** $224 \times 224 \times 3$ RGB.
- **Resampling Method:** High-fidelity Bilinear interpolation (`PIL.Image.Resampling.BILINEAR`).

---

## 10. RGB Conversion Protocol
All images are explicitly converted via `.convert("RGB")` before tensor conversion. This converts any 4-channel RGBA images to 3 channels and standardizes single-channel grayscale inputs, preventing runtime channel mismatches.

---

## 11. Normalization & Preprocessing Approach
- **Array Conversion:** Converted to NumPy `float32` arrays.
- **Pixel Value Scaling:** Scaled to $[0.0, 1.0]$ by dividing raw integer values $[0, 255]$ by $255.0$.
- **On-the-Fly Processing:** Normalized arrays are created on-the-fly during data loading and training rather than pre-saved to disk, conserving disk space.

---

## 12. Training Augmentation (TRAINING DATA ONLY)
To improve model generalization without distorting disease lesion characteristics, bounded transformations are applied on-the-fly:

| Augmentation Operator | Parameter Range | Biological Rationale |
| :--- | :--- | :--- |
| **Rotation** | $\pm 15^\circ$ | Leaves may be held at slight angles; avoids upside-down disorientation |
| **Horizontal Flip** | $p = 0.5$ (50%) | Foliar spots and pustules are bilaterally invariant |
| **Vertical Flip** | False (Disabled) | Preserves natural stem-to-tip gravity orientation |
| **Zoom** | $[0.9, 1.1]$ ($\pm 10\%$) | Simulates varying camera focal distances |
| **Brightness Adjustment** | $\pm 10\%$ | Simulates ambient lighting shifts |
| **Contrast Adjustment** | $[0.9, 1.1]$ | Simulates varying cloud and camera sensor exposures |

---

## 13. Validation & Test Preprocessing
- **Zero Augmentation:** Validation and test pipelines execute strictly deterministic preprocessing: Resize ($224 \times 224$), RGB conversion, and $[0, 1]$ float32 normalization.
- **Bitwise Determinism:** Subsequent calls on identical images yield $0.0$ maximum absolute pixel difference.

---

## 14. Class-Index Mapping
Configured in `ml/data/class_indices.json`:

```json
{
  "class_to_idx": {
    "Apple___Apple_scab": 0,
    "Apple___Black_rot": 1,
    "Apple___healthy": 2,
    "Corn_(maize)___Common_rust_": 3,
    "Corn_(maize)___Northern_Leaf_Blight": 4,
    "Corn_(maize)___healthy": 5,
    "Potato___Early_blight": 6,
    "Potato___Late_blight": 7,
    "Potato___healthy": 8,
    "Tomato___Bacterial_spot": 9,
    "Tomato___Early_blight": 10,
    "Tomato___Late_blight": 11,
    "Tomato___healthy": 12
  }
}
```

---

## 15. Data Leakage Prevention Guarantees
1. **Hash Isolation:** $0$ shared MD5 hashes between Train, Validation, and Test sets.
2. **Augmentation Isolation:** Augmentations are executed strictly inside the training data loader; validation and test splits receive zero stochastic transformations.
3. **Evaluation Holdout:** Test split (`test.csv`) is preserved solely for final Phase 5 benchmarking.

---

## 16. Final Dataset Statistics After Splitting

| Class Name | Train (80%) | Val (10%) | Test (10%) | Total Usable | Excluded Duplicates |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Apple___Apple_scab` | 504 | 63 | 63 | 630 | 0 |
| `Apple___Black_rot` | 497 | 62 | 62 | 621 | 0 |
| `Apple___healthy` | 1,310 | 164 | 164 | 1,638 | 7 |
| `Corn_(maize)___Common_rust_` | 953 | 120 | 119 | 1,192 | 0 |
| `Corn_(maize)___Northern_Leaf_Blight` | 788 | 99 | 98 | 985 | 0 |
| `Corn_(maize)___healthy` | 930 | 116 | 116 | 1,162 | 0 |
| `Potato___Early_blight` | 798 | 100 | 100 | 998 | 0 |
| `Potato___Late_blight` | 799 | 100 | 100 | 999 | 0 |
| `Potato___healthy` | 122 | 15 | 15 | 152 | 0 |
| `Tomato___Bacterial_spot` | 1,701 | 212 | 213 | 2,126 | 0 |
| `Tomato___Early_blight` | 800 | 100 | 100 | 1,000 | 0 |
| `Tomato___Late_blight` | 1,520 | 190 | 190 | 1,900 | 8 |
| `Tomato___healthy` | 1,267 | 158 | 159 | 1,584 | 6 |
| **TOTALS** | **11,989** | **1,499** | **1,499** | **14,987** | **21** |

---

## 17. Known Limitations
1. **Studio Background Artifacts:** Laboratory background uniformity does not capture outdoor field clutter (soil, pebbles, weed interference).
2. **Minority Class Sparsity:** `Potato___healthy` has only 152 total images (122 in train), creating a class imbalance relative to `Tomato___Bacterial_spot` (1,701 in train).
3. **Burst Redundancy:** Despite 100% cryptographic duplicate removal, near-identical visual bursts from original laboratory photography may exist within class folders.
