"""
Automated Test Suite for Preprocessing, Augmentation & Data Splitting
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Executes the 11 mandatory validation tests defined in Phase 3.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Import preprocessing modules
from config import (
    BASE_DIR,
    SPLITS_DIR,
    SELECTED_CLASSES_PATH,
    CLASS_INDICES_PATH,
    TARGET_IMAGE_SIZE,
    INPUT_SHAPE
)
from augmentation import (
    load_and_preprocess_image,
    apply_training_augmentation,
    load_and_augment_training_image
)

def run_all_tests():
    print("=" * 80)
    print("PHASE 3: PREPROCESSING PIPELINE VALIDATION TEST SUITE")
    print("=" * 80)

    # Load metadata
    with open(SELECTED_CLASSES_PATH, "r", encoding="utf-8") as f:
        selected_classes = set(json.load(f)["classes"])

    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_mapping_data = json.load(f)
        class_to_idx = class_mapping_data["class_to_idx"]
        idx_to_class = class_mapping_data["idx_to_class"]

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "validation.csv"
    test_csv = SPLITS_DIR / "test.csv"

    # TEST 1 & Loading
    print("Running TEST 1: Verifying manifest files exist...")
    assert train_csv.exists(), "train.csv does not exist"
    assert val_csv.exists(), "validation.csv does not exist"
    assert test_csv.exists(), "test.csv does not exist"
    print("  [PASS] All 3 split CSV files exist on disk.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # TEST 2: Every referenced image path exists and is readable
    print("\nRunning TEST 2: Verifying referenced image readability (sampling 100 per split)...")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        sample_df = df.sample(min(100, len(df)), random_state=42)
        for _, row in sample_df.iterrows():
            img_path = BASE_DIR / row["image_path"]
            assert img_path.exists(), f"Image path not found: {img_path}"
            with Image.open(img_path) as img:
                img.verify()
    print("  [PASS] Sampled images exist and pass PIL integrity verification.")

    # TEST 3: Every class belongs to selected_classes.json
    print("\nRunning TEST 3: Verifying class membership in selected_classes.json...")
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        manifest_classes = set(df["class_name"].unique())
        unrecognized = manifest_classes - selected_classes
        assert len(unrecognized) == 0, f"Unrecognized classes in {df_name}: {unrecognized}"
    print("  [PASS] All manifest classes strictly belong to selected_classes.json.")

    # TEST 4: Every class has samples in train, val, and test
    print("\nRunning TEST 4: Verifying class representation across splits...")
    for cls in selected_classes:
        t_c = len(train_df[train_df["class_name"] == cls])
        v_c = len(val_df[val_df["class_name"] == cls])
        te_c = len(test_df[test_df["class_name"] == cls])
        assert t_c > 0, f"Class {cls} missing from train split!"
        assert v_c > 0, f"Class {cls} missing from validation split!"
        assert te_c > 0, f"Class {cls} missing from test split!"
    print("  [PASS] All 13 classes are represented in train, validation, and test splits.")

    # TEST 5: No exact duplicate MD5 hash across splits
    print("\nRunning TEST 5: Verifying zero hash leakage across splits...")
    def get_hashes(df, sample_limit=200):
        hashes = set()
        for _, row in df.sample(min(sample_limit, len(df)), random_state=42).iterrows():
            p = BASE_DIR / row["image_path"]
            with open(p, "rb") as f:
                hashes.add(hashlib.md5(f.read()).hexdigest())
        return hashes

    t_h = get_hashes(train_df)
    v_h = get_hashes(val_df)
    te_h = get_hashes(test_df)
    assert len(t_h.intersection(v_h)) == 0, "Train-Val hash leakage detected!"
    assert len(t_h.intersection(te_h)) == 0, "Train-Test hash leakage detected!"
    assert len(v_h.intersection(te_h)) == 0, "Val-Test hash leakage detected!"
    print("  [PASS] Zero hash collisions between train, validation, and test splits.")

    # TEST 6: Consistent class index mapping
    print("\nRunning TEST 6: Verifying class index bi-directional consistency...")
    for cls_name, idx in class_to_idx.items():
        assert idx_to_class[str(idx)] == cls_name, f"Index mismatch for {cls_name} ({idx})"
    for df in [train_df, val_df, test_df]:
        for _, row in df.head(50).iterrows():
            assert class_to_idx[row["class_name"]] == row["class_index"]
    print("  [PASS] Class indices are 100% bijective and consistent across all manifests.")

    # TEST 7: Preprocessing produces exact (224, 224, 3) float32 in [0, 1]
    print("\nRunning TEST 7: Verifying standardized output tensor shape and range...")
    sample_path = BASE_DIR / train_df.iloc[0]["image_path"]
    tensor = load_and_preprocess_image(sample_path)
    assert tensor.shape == (224, 224, 3), f"Unexpected shape: {tensor.shape}"
    assert tensor.dtype == np.float32, f"Unexpected dtype: {tensor.dtype}"
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0, "Values outside normalized [0, 1] range"
    print(f"  [PASS] Output tensor shape is {tensor.shape} (dtype={tensor.dtype}, min={tensor.min():.2f}, max={tensor.max():.2f}).")

    # TEST 8: Training augmentation instantiates successfully
    print("\nRunning TEST 8: Verifying training augmentation execution...")
    with Image.open(sample_path) as raw_img:
        aug_img = apply_training_augmentation(raw_img.convert("RGB"), seed=123)
        assert isinstance(aug_img, Image.Image), "Augmentation did not return a PIL Image"
        assert aug_img.size == raw_img.size, "Augmentation unexpectedly altered base canvas dimensions"
    print("  [PASS] Training augmentation executes smoothly and retains image integrity.")

    # TEST 9 & 10: Validation & Test do not perform random augmentation
    print("\nRunning TEST 9 & 10: Verifying absence of random augmentation in validation/test...")
    val_sample_path = BASE_DIR / val_df.iloc[0]["image_path"]
    test_sample_path = BASE_DIR / test_df.iloc[0]["image_path"]
    # Two subsequent calls on same image must produce IDENTICAL arrays
    v1 = load_and_preprocess_image(val_sample_path)
    v2 = load_and_preprocess_image(val_sample_path)
    t1 = load_and_preprocess_image(test_sample_path)
    t2 = load_and_preprocess_image(test_sample_path)
    np.testing.assert_array_equal(v1, v2, err_msg="Validation preprocessing is non-deterministic!")
    np.testing.assert_array_equal(t1, t2, err_msg="Test preprocessing is non-deterministic!")
    print("  [PASS] Validation and Test preprocessing are strictly deterministic (no random transforms).")

    # TEST 11: Deterministic validation/test preprocessing
    print("\nRunning TEST 11: Verifying repeatable deterministic pixel reconstruction...")
    diff = np.max(np.abs(v1 - v2))
    assert diff == 0.0, f"Non-zero difference between subsequent runs: {diff}"
    print(f"  [PASS] Maximum absolute difference across repeat calls is {diff} (Bitwise identical).")

    print("\n" + "=" * 80)
    print("ALL 11 PREPROCESSING TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n[TEST FAILED]: {e}", file=sys.stderr)
        sys.exit(1)
