"""
Train Leaf / Non-Leaf Gate Classifier
AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Trains a lightweight binary classifier:
- Class 0: NON_LEAF
- Class 1: LEAF

Model Architecture:
- MobileNetV2 (ImageNet weights)
- GlobalAveragePooling2D
- Dropout(0.3)
- Dense(2, activation='softmax')

Saves:
- ml/models/leaf_detector_model.keras
- ml/models/leaf_detector_metadata.json
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "ml" / "data" / "leaf_detection" / "splits"
MODELS_DIR = REPO_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "leaf_detector_model.keras"
METADATA_OUTPUT_PATH = MODELS_DIR / "leaf_detector_metadata.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

def load_dataset_from_csv(csv_path, is_training=False):
    df = pd.read_csv(csv_path)
    file_paths = [str(REPO_ROOT / p) for p in df["file_path"].values]
    labels = df["label"].values.astype(np.int32)
    
    def process_path(file_path, label):
        img_raw = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        # Augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ], name="augmentation")
        
        ds = ds.shuffle(buffer_size=min(len(df), 1000), seed=SEED)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(BATCH_SIZE)
        
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds, len(df)

def build_leaf_detector():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze for Stage 1
    
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout_regularization")(x)
    outputs = layers.Dense(2, activation="softmax", name="classification_head")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="LeafNonLeafGateDetector")
    return model, base_model

def train():
    print("=" * 60)
    print("Training Leaf / Non-Leaf Gate Classifier")
    print("=" * 60)
    
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"
    
    train_ds, n_train = load_dataset_from_csv(train_csv, is_training=True)
    val_ds, n_val = load_dataset_from_csv(val_csv, is_training=False)
    test_ds, n_test = load_dataset_from_csv(test_csv, is_training=False)
    
    print(f"Loaded datasets: Train={n_train}, Val={n_val}, Test={n_test}")
    
    model, base_model = build_leaf_detector()
    
    # ── Stage 1: Feature Extraction ───────────────────────────────────────────
    print("\n--- Stage 1: Feature Extraction (Backbone Frozen) ---")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=1
    )
    
    # ── Stage 2: Fine-Tuning ──────────────────────────────────────────────────
    print("\n--- Stage 2: Fine-Tuning (Unfreezing layers 100+) ---")
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[early_stop],
        verbose=1
    )
    
    # ── Save Trained Model ────────────────────────────────────────────────────
    print(f"\nSaving model to: {MODEL_OUTPUT_PATH}")
    model.save(str(MODEL_OUTPUT_PATH))
    
    # ── Test Evaluation ───────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4%}")
    
    metadata = {
        "model_name": "LeafNonLeafGateDetector",
        "architecture": "MobileNetV2 (ImageNet transfer learning)",
        "input_shape": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        "classes": {
            0: "NON_LEAF",
            1: "LEAF"
        },
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "rejection_threshold": 0.50,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {METADATA_OUTPUT_PATH}")
    print("Gate training complete!")

if __name__ == "__main__":
    train()
