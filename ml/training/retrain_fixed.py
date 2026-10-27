"""
Retrain Fixed Script -- Enhanced Augmentation and More Epochs
"""

import os, sys, json, random, datetime, shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tensorflow as tf
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR   = Path(__file__).resolve().parent
ML_DIR       = SCRIPT_DIR.parent
REPO_ROOT    = ML_DIR.parent
DATA_DIR     = ML_DIR / "data"
SPLITS_DIR   = DATA_DIR / "splits"
CI_PATH      = DATA_DIR / "class_indices.json"
MODELS_DIR   = ML_DIR / "models"
EVAL_DIR     = ML_DIR / "evaluation"
HISTORY_DIR  = EVAL_DIR / "training_history"
BEST_MODEL_PATH  = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"
BACKUP_PATH      = MODELS_DIR / "best_model_backup.keras"
METADATA_PATH    = MODELS_DIR / "model_metadata.json"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_H = 224; IMAGE_W = 224; BATCH_SIZE = 32; NUM_WORKERS = 4
STAGE1_EPOCHS = 12; STAGE1_LR = 1e-3
STAGE2_EPOCHS = 10; STAGE2_LR = 5e-5; STAGE2_UNFREEZE_FROM = 80

def load_class_mapping():
    with open(CI_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["class_to_idx"]

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.20, seed=SEED)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20, seed=SEED)
    image = tf.image.random_saturation(image, lower=0.60, upper=1.40, seed=SEED)
    image = tf.image.random_hue(image, max_delta=0.08, seed=SEED)
    shape = tf.shape(image)
    crop_h = tf.cast(tf.cast(shape[0], tf.float32) * tf.random.uniform([], 0.80, 1.0, seed=SEED), tf.int32)
    crop_w = tf.cast(tf.cast(shape[1], tf.float32) * tf.random.uniform([], 0.80, 1.0, seed=SEED), tf.int32)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3], seed=SEED)
    image = tf.image.resize(image, [IMAGE_H, IMAGE_W])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def _decode_and_resize(image_path, label):
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMAGE_H, IMAGE_W])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def build_dataset(csv_path, class_to_idx, augment=False, shuffle=False):
    df = pd.read_csv(csv_path)
    abs_paths = [str(REPO_ROOT / row["image_path"]) for _, row in df.iterrows()]
    labels    = [int(class_to_idx[row["class_name"]]) for _, row in df.iterrows()]
    ds = tf.data.Dataset.from_tensor_slices((abs_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_and_resize, num_parallel_calls=NUM_WORKERS)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=NUM_WORKERS)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def compute_class_weights(class_to_idx):
    from sklearn.utils.class_weight import compute_class_weight
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    y_train  = [class_to_idx[cn] for cn in train_df["class_name"]]
    classes  = np.array(sorted(class_to_idx.values()))
    weights  = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights.tolist()))
    apple_classes = [i for cls, i in class_to_idx.items() if cls.startswith("Apple")]
    for idx in apple_classes:
        weight_dict[idx] *= 1.5
        print(f"  Apple class idx {idx} boosted weight: {weight_dict[idx]:.3f}")
    return weight_dict

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_H, IMAGE_W, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs  = tf.keras.Input(shape=(IMAGE_H, IMAGE_W, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="MobileNetV2_FixedRetrain"), base

def make_callbacks(stage):
    patience = 3 if stage == 1 else 4
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(BEST_MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]

def main():
    start = datetime.datetime.now()
    print("=" * 70)
    print("RETRAIN FIXED -- Enhanced Augmentation + More Epochs")
    print("=" * 70)
    if BEST_MODEL_PATH.exists():
        shutil.copy(BEST_MODEL_PATH, BACKUP_PATH)
        print(f"  Backed up existing model -> {BACKUP_PATH}")
    class_to_idx = load_class_mapping()
    num_classes  = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"  Classes: {num_classes}")
    print("\n  Computing class weights ...")
    class_weights = compute_class_weights(class_to_idx)
    print(f"  Weight range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}")
    print("\n  Building datasets ...")
    train_ds = build_dataset(SPLITS_DIR / "train.csv",      class_to_idx, augment=True,  shuffle=True)
    val_ds   = build_dataset(SPLITS_DIR / "validation.csv", class_to_idx, augment=False, shuffle=False)
    print(f"  Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")
    print("\n" + "=" * 70)
    print(f"STAGE 1 -- Head Training ({STAGE1_EPOCHS} epochs max, lr={STAGE1_LR})")
    print("=" * 70)
    model, base = build_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=STAGE1_EPOCHS, class_weight=class_weights, callbacks=make_callbacks(stage=1), verbose=1)
    h1_dict = {k: [float(v) for v in vals] for k, vals in h1.history.items()}
    best_s1_acc  = max(h1_dict["val_accuracy"])
    best_s1_loss = min(h1_dict["val_loss"])
    print(f"\n  Stage 1 best val_accuracy: {best_s1_acc:.4f} | val_loss: {best_s1_loss:.4f}")
    print("\n" + "=" * 70)
    print(f"STAGE 2 -- Fine-tuning (unfreeze from idx {STAGE2_UNFREEZE_FROM}, lr={STAGE2_LR})")
    print("=" * 70)
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for i, l in enumerate(layer.layers):
                l.trainable = i >= STAGE2_UNFREEZE_FROM
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS, class_weight=class_weights, callbacks=make_callbacks(stage=2), verbose=1)
    h2_dict = {k: [float(v) for v in vals] for k, vals in h2.history.items()}
    best_s2_acc  = max(h2_dict["val_accuracy"])
    best_s2_loss = min(h2_dict["val_loss"])
    print(f"\n  Stage 2 best val_accuracy: {best_s2_acc:.4f} | val_loss: {best_s2_loss:.4f}")
    model.save(FINAL_MODEL_PATH)
    print(f"\n  Final model -> {FINAL_MODEL_PATH}")
    print(f"  Best model  -> {BEST_MODEL_PATH}")
    print("\n" + "=" * 70)
    print("PER-CLASS ACCURACY ON VALIDATION SET")
    print("=" * 70)
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total   = {i: 0 for i in range(num_classes)}
    for imgs, labels in val_ds:
        preds = best_model(imgs, training=False).numpy()
        pred_idx = np.argmax(preds, axis=-1)
        for true_lbl, pred_lbl in zip(labels.numpy(), pred_idx):
            per_class_total[int(true_lbl)] += 1
            if int(true_lbl) == int(pred_lbl):
                per_class_correct[int(true_lbl)] += 1
    for i in range(num_classes):
        total = per_class_total[i]; correct = per_class_correct[i]
        acc = correct / total * 100 if total > 0 else 0
        flag = " *** LOW ***" if acc < 70 else ""
        print(f"  [{i:2d}] {idx_to_class[i]:<45} {correct:4d}/{total:4d}  ({acc:.1f}%){flag}")
    duration = round((datetime.datetime.now() - start).total_seconds() / 60, 1)
    metadata = {
        "model_name": "MobileNetV2", "retrain_version": "fixed_v2",
        "input_size": [IMAGE_H, IMAGE_W, 3], "num_classes": num_classes,
        "class_to_idx": class_to_idx, "preprocessing": "RGB, resize 224x224, normalize [0,1]",
        "training_seed": SEED, "batch_size": BATCH_SIZE,
        "stage1_epochs_run": len(h1_dict["loss"]), "stage1_lr": STAGE1_LR,
        "stage2_epochs_run": len(h2_dict["loss"]), "stage2_lr": STAGE2_LR,
        "stage2_unfreeze_from": STAGE2_UNFREEZE_FROM, "apple_weight_boost": 1.5,
        "stage1_best_val_accuracy": round(best_s1_acc, 6), "stage1_best_val_loss": round(best_s1_loss, 6),
        "stage2_best_val_accuracy": round(best_s2_acc, 6), "stage2_best_val_loss": round(best_s2_loss, 6),
        "training_duration_minutes": duration, "trained_at": start.isoformat(),
        "best_model_path": str(BEST_MODEL_PATH), "final_model_path": str(FINAL_MODEL_PATH),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved -> {METADATA_PATH}")
    print("\n" + "=" * 70)
    print(f"RETRAINING COMPLETE in {duration} minutes")
    print(f"  Stage 1: {best_s1_acc*100:.2f}%  |  Stage 2: {best_s2_acc*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
