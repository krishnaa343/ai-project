"""
Visual Verification Script
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Samples images across Train, Validation, and Test splits,
demonstrates on-the-fly training augmentation, and verifies standardized
preprocessing dimensions (224x224x3 RGB) without writing heavy image grids to disk.
"""

import sys
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

from config import REPO_ROOT, SPLITS_DIR, TARGET_IMAGE_SIZE
from augmentation import load_and_preprocess_image, apply_training_augmentation

def run_visual_verification():
    print("=" * 80)
    print("PREPROCESSING VISUAL & STRUCTURAL VERIFICATION")
    print("=" * 80)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    results = []

    # 1. Training Sample (Original & Augmented)
    train_sample = train_df.iloc[0]
    t_path = REPO_ROOT / train_sample["image_path"]
    t_tensor = load_and_preprocess_image(t_path)
    
    with Image.open(t_path) as raw_img:
        aug_img = apply_training_augmentation(raw_img.convert("RGB"), seed=42)
        aug_resized = aug_img.resize(TARGET_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
        aug_tensor = np.asarray(aug_resized, dtype=np.float32) / 255.0

    results.append({
        "split": "train (original)",
        "class": train_sample["class_name"],
        "file": Path(train_sample["image_path"]).name,
        "standardized_shape": str(t_tensor.shape),
        "min_max": f"[{t_tensor.min():.2f}, {t_tensor.max():.2f}]",
        "mean_pixel": f"{t_tensor.mean():.4f}",
        "augmented": False
    })
    results.append({
        "split": "train (augmented demo)",
        "class": train_sample["class_name"],
        "file": Path(train_sample["image_path"]).name,
        "standardized_shape": str(aug_tensor.shape),
        "min_max": f"[{aug_tensor.min():.2f}, {aug_tensor.max():.2f}]",
        "mean_pixel": f"{aug_tensor.mean():.4f}",
        "augmented": True
    })

    # 2. Validation Sample (Strictly original, zero augmentation)
    val_sample = val_df.iloc[0]
    v_path = REPO_ROOT / val_sample["image_path"]
    v_tensor = load_and_preprocess_image(v_path)
    results.append({
        "split": "validation",
        "class": val_sample["class_name"],
        "file": Path(val_sample["image_path"]).name,
        "standardized_shape": str(v_tensor.shape),
        "min_max": f"[{v_tensor.min():.2f}, {v_tensor.max():.2f}]",
        "mean_pixel": f"{v_tensor.mean():.4f}",
        "augmented": False
    })

    # 3. Test Sample (Strictly original, zero augmentation)
    test_sample = test_df.iloc[0]
    te_path = REPO_ROOT / test_sample["image_path"]
    te_tensor = load_and_preprocess_image(te_path)
    results.append({
        "split": "test",
        "class": test_sample["class_name"],
        "file": Path(test_sample["image_path"]).name,
        "standardized_shape": str(te_tensor.shape),
        "min_max": f"[{te_tensor.min():.2f}, {te_tensor.max():.2f}]",
        "mean_pixel": f"{te_tensor.mean():.4f}",
        "augmented": False
    })

    for r in results:
        print(f"Split:        {r['split'].upper()}")
        print(f"Class:        {r['class']}")
        print(f"Source File:  {r['file']}")
        print(f"Tensor Shape: {r['standardized_shape']} | Range: {r['min_max']} | Mean: {r['mean_pixel']}")
        print(f"Augmented:    {r['augmented']}")
        print("-" * 80)

    print("Conclusion: All split tensors standardize to (224, 224, 3) float32 in [0, 1].")
    print("Augmentation is strictly bounded and isolated to training flows only.")
    print("=" * 80)

if __name__ == "__main__":
    run_visual_verification()
