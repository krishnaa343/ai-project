"""
TensorFlow Data Pipeline – CSV-Manifest-Based Image Loader
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Reads from Phase 3 CSV manifests and builds efficient tf.data pipelines with:
  - deterministic validation / test preprocessing (no augmentation)
  - on-the-fly training augmentation (via tf.image ops)
  - prefetching and parallel map for throughput
"""

import json
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

try:
    from config import (
        REPO_ROOT, CLASS_INDICES_PATH, SPLITS_DIR,
        INPUT_SHAPE, BATCH_SIZE, PREFETCH, NUM_WORKERS,
        IMAGE_HEIGHT, IMAGE_WIDTH,
        RANDOM_SEED
    )
except ImportError:
    from ml.training.config import (
        REPO_ROOT, CLASS_INDICES_PATH, SPLITS_DIR,
        INPUT_SHAPE, BATCH_SIZE, PREFETCH, NUM_WORKERS,
        IMAGE_HEIGHT, IMAGE_WIDTH,
        RANDOM_SEED
    )

# ── Class Mapping ─────────────────────────────────────────────────────────────
def load_class_mapping():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["class_to_idx"]   # {"Apple___Apple_scab": 0, ...}


# ── Preprocessing Functions ───────────────────────────────────────────────────

def _decode_and_resize(image_path, label):
    """Deterministic: read → decode → resize → normalize. No randomness."""
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(raw, channels=3)          # always 3-ch RGB
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0               # scale to [0, 1]
    return img, label


def _augment(image, label):
    """
    Bounded training-only augmentation using tf.image ops (no extra libraries).
    Applied ONLY inside the training pipeline map call.
    """
    # Random horizontal flip (50%)
    image = tf.image.random_flip_left_right(image, seed=RANDOM_SEED)
    # Random brightness ± 10%
    image = tf.image.random_brightness(image, max_delta=0.10, seed=RANDOM_SEED)
    # Random contrast ± 10%
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=RANDOM_SEED)
    # Clip to valid range after adjustments
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ── Dataset Builders ──────────────────────────────────────────────────────────

def _build_dataset(csv_path, class_to_idx, augment=False, shuffle=False):
    """
    Generic dataset builder from a Phase 3 CSV manifest.

    Args:
        csv_path  : Path to train / validation / test CSV
        class_to_idx : dict mapping class_name → integer index
        augment   : True only for training
        shuffle   : True only for training

    Returns:
        tf.data.Dataset of (image_tensor, label_int) batches
    """
    df = pd.read_csv(csv_path)

    # Build absolute path strings
    abs_paths = [str(REPO_ROOT / row["image_path"]) for _, row in df.iterrows()]
    labels    = [int(class_to_idx[row["class_name"]]) for _, row in df.iterrows()]

    path_ds  = tf.data.Dataset.from_tensor_slices(abs_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds       = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    # Decode + resize (deterministic)
    ds = ds.map(_decode_and_resize, num_parallel_calls=NUM_WORKERS)

    # Augment training data only
    if augment:
        ds = ds.map(_augment, num_parallel_calls=NUM_WORKERS)

    ds = ds.batch(BATCH_SIZE)

    if PREFETCH:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_train_dataset(class_to_idx):
    return _build_dataset(SPLITS_DIR / "train.csv",      class_to_idx, augment=True,  shuffle=True)

def get_val_dataset(class_to_idx):
    return _build_dataset(SPLITS_DIR / "validation.csv", class_to_idx, augment=False, shuffle=False)

def get_test_dataset(class_to_idx):
    # NOTE: test dataset should only be used in Phase 5 evaluation
    return _build_dataset(SPLITS_DIR / "test.csv",       class_to_idx, augment=False, shuffle=False)


# ── Class Weights ─────────────────────────────────────────────────────────────

def compute_class_weights(class_to_idx):
    """
    Computes sklearn-style balanced class weights from TRAINING DATA ONLY.
    Formula: w_c = total_samples / (n_classes * samples_in_class)
    """
    from sklearn.utils.class_weight import compute_class_weight

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    y_train  = [class_to_idx[cn] for cn in train_df["class_name"]]
    classes  = np.array(sorted(class_to_idx.values()))

    weights  = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights.tolist()))
