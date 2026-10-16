"""
Dataset Quality Filtering, Deduplication & Stratified Splitting Script
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Reads raw selected classes, verifies image integrity and RGB readability,
removes exact MD5 duplicates from manifests (preserving raw files intact),
generates deterministic 80/10/10 Train/Val/Test splits with fixed random seed 42,
and writes portable CSV manifests to ml/data/splits/.
"""

import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Import shared configuration
from config import (
    RAW_DATA_DIR,
    SPLITS_DIR,
    SELECTED_CLASSES_PATH,
    CLASS_INDICES_PATH,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)

def compute_md5(file_path, chunk_size=65536):
    """Calculates cryptographic MD5 hash of an image file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def parse_crop_and_disease(class_name):
    """Parses folder name into crop and disease components."""
    parts = class_name.split("___")
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else "Unknown"
    return crop, disease

def prepare_splits():
    print("=" * 80)
    print("DATASET PREPARATION, FILTERING & STRATIFIED SPLITTING")
    print("=" * 80)

    # 1. Load Configurations
    if not SELECTED_CLASSES_PATH.exists():
        raise FileNotFoundError(f"Missing configuration: {SELECTED_CLASSES_PATH}")
    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Missing configuration: {CLASS_INDICES_PATH}")

    with open(SELECTED_CLASSES_PATH, "r", encoding="utf-8") as f:
        selected_classes = json.load(f)["classes"]

    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)["class_to_idx"]

    print(f"Target Classes: {len(selected_classes)}")
    print(f"Split Ratios:   Train={TRAIN_RATIO*100:.0f}% | Val={VAL_RATIO*100:.0f}% | Test={TEST_RATIO*100:.0f}% (Seed={RANDOM_SEED})")
    print(f"Raw Directory:  {RAW_DATA_DIR}\n")

    # 2. Scan & Validate Images
    records = []
    excluded_records = []
    seen_hashes = {}
    duplicate_count = 0

    for class_name in selected_classes:
        class_folder = RAW_DATA_DIR / class_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Selected class folder not found: {class_folder}")

        files = [f for f in class_folder.iterdir() if f.is_file()]
        class_idx = class_mapping[class_name]
        crop, disease = parse_crop_and_disease(class_name)

        for file_path in files:
            ext = file_path.suffix.lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                excluded_records.append({
                    "file_path": str(file_path),
                    "class_name": class_name,
                    "reason": f"Unsupported extension: {ext}"
                })
                continue

            # Verify readability and RGB convertibility
            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    rgb_test = img.convert("RGB")
                    w, h = rgb_test.size
            except Exception as e:
                excluded_records.append({
                    "file_path": str(file_path),
                    "class_name": class_name,
                    "reason": f"Corrupted/unreadable image: {str(e)}"
                })
                continue

            # Check MD5 hash
            file_hash = compute_md5(file_path)
            if file_hash in seen_hashes:
                duplicate_count += 1
                excluded_records.append({
                    "file_path": str(file_path),
                    "class_name": class_name,
                    "reason": f"Exact binary duplicate of {seen_hashes[file_hash]}"
                })
                continue

            seen_hashes[file_hash] = str(file_path)

            # Store portable relative path (relative to repo root)
            rel_path = file_path.as_posix()
            if "ai project/" in rel_path:
                rel_path = rel_path.split("ai project/")[1]

            records.append({
                "image_path": rel_path,
                "class_name": class_name,
                "class_index": class_idx,
                "crop": crop,
                "disease": disease,
                "md5_hash": file_hash
            })

    df = pd.DataFrame(records)
    print(f"Total Files Scanned for Selected Classes: {len(records) + len(excluded_records)}")
    print(f"Total Valid Usable Unique Images:         {len(df)}")
    print(f"Excluded Duplicate Files:                 {duplicate_count}")
    print(f"Excluded Corrupted/Invalid Files:         {len(excluded_records) - duplicate_count}")

    # 3. Stratified Split (80% Train, 10% Val, 10% Test)
    # First split: 80% train, 20% temp (stratified)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["class_index"],
        random_state=RANDOM_SEED
    )

    # Second split: 50% of temp -> Val (10% total), 50% of temp -> Test (10% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class_index"],
        random_state=RANDOM_SEED
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    # 4. Strict Leakage Verification
    train_hashes = set(train_df["md5_hash"])
    val_hashes = set(val_df["md5_hash"])
    test_hashes = set(test_df["md5_hash"])

    overlap_train_val = train_hashes.intersection(val_hashes)
    overlap_train_test = train_hashes.intersection(test_hashes)
    overlap_val_test = val_hashes.intersection(test_hashes)

    assert len(overlap_train_val) == 0, f"LEAKAGE DETECTED: {len(overlap_train_val)} common hashes between Train and Val!"
    assert len(overlap_train_test) == 0, f"LEAKAGE DETECTED: {len(overlap_train_test)} common hashes between Train and Test!"
    assert len(overlap_val_test) == 0, f"LEAKAGE DETECTED: {len(overlap_val_test)} common hashes between Val and Test!"
    print("\n[OK] ZERO HASH LEAKAGE: All splits are cryptographically isolated.")

    # 5. Save Manifests
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["image_path", "class_name", "class_index", "crop", "disease", "split"]

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "validation.csv"
    test_csv = SPLITS_DIR / "test.csv"

    train_df[cols].to_csv(train_csv, index=False)
    val_df[cols].to_csv(val_csv, index=False)
    test_df[cols].to_csv(test_csv, index=False)

    print(f"\nManifests Created:")
    print(f"  Train:      {train_csv} ({len(train_df)} images)")
    print(f"  Validation: {val_csv} ({len(val_df)} images)")
    print(f"  Test:       {test_csv} ({len(test_df)} images)")
    print(f"  Total:      {len(train_df) + len(val_df) + len(test_df)} images")

    # 6. Save Split Summary Report JSON
    summary = {
        "random_seed": RANDOM_SEED,
        "target_ratios": {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "total_usable_images": len(df),
        "total_train_images": len(train_df),
        "total_validation_images": len(val_df),
        "total_test_images": len(test_df),
        "total_excluded_duplicates": duplicate_count,
        "total_excluded_corruptions": len(excluded_records) - duplicate_count,
        "class_breakdown": {}
    }

    print("\n" + "-" * 80)
    print(f"{'Class Name':<42} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-" * 80)

    for class_name in selected_classes:
        t_cnt = len(train_df[train_df["class_name"] == class_name])
        v_cnt = len(val_df[val_df["class_name"] == class_name])
        te_cnt = len(test_df[test_df["class_name"] == class_name])
        tot = t_cnt + v_cnt + te_cnt
        summary["class_breakdown"][class_name] = {
            "train": t_cnt,
            "validation": v_cnt,
            "test": te_cnt,
            "total": tot
        }
        print(f"{class_name:<42} | {t_cnt:<7} | {v_cnt:<7} | {te_cnt:<7} | {tot:<7}")
    print("=" * 80)

    summary_file = SPLITS_DIR / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    excluded_file = SPLITS_DIR / "excluded_files.json"
    with open(excluded_file, "w", encoding="utf-8") as f:
        json.dump(excluded_records, f, indent=2)

    print(f"Summary JSON:   {summary_file}")
    print(f"Exclusions Log: {excluded_file}\n")
    return summary

if __name__ == "__main__":
    prepare_splits()
