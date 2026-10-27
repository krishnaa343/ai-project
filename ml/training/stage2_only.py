import os, sys, json, datetime, shutil
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
import random; random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

ML_DIR       = Path(r'C:\Users\agarw\OneDrive\Desktop\ai project\ml')
REPO_ROOT    = ML_DIR.parent
DATA_DIR     = ML_DIR / "data"
SPLITS_DIR   = DATA_DIR / "splits"
CI_PATH      = DATA_DIR / "class_indices.json"
MODELS_DIR   = ML_DIR / "models"
BEST_MODEL   = MODELS_DIR / "best_model.keras"
FINAL_MODEL  = MODELS_DIR / "final_model.keras"
METADATA     = MODELS_DIR / "model_metadata.json"

IMAGE_H = 224; IMAGE_W = 224; BATCH_SIZE = 32; NUM_WORKERS = 4
STAGE2_EPOCHS = 10; STAGE2_LR = 5e-5; UNFREEZE_FROM = 80

def load_class_mapping():
    with open(CI_PATH, "r") as f: data = json.load(f)
    return data["class_to_idx"]

def _decode_and_resize(image_path, label):
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMAGE_H, IMAGE_W])
    return tf.cast(img, tf.float32) / 255.0, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.20, seed=SEED)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20, seed=SEED)
    image = tf.image.random_saturation(image, lower=0.60, upper=1.40, seed=SEED)
    image = tf.image.random_hue(image, max_delta=0.08, seed=SEED)
    shape = tf.shape(image)
    ch = tf.cast(tf.cast(shape[0], tf.float32) * tf.random.uniform([], 0.80, 1.0, seed=SEED), tf.int32)
    cw = tf.cast(tf.cast(shape[1], tf.float32) * tf.random.uniform([], 0.80, 1.0, seed=SEED), tf.int32)
    image = tf.image.random_crop(image, [ch, cw, 3], seed=SEED)
    image = tf.image.resize(image, [IMAGE_H, IMAGE_W])
    return tf.clip_by_value(image, 0.0, 1.0), label

def build_dataset(csv_path, class_to_idx, augment=False, shuffle=False):
    df = pd.read_csv(csv_path)
    abs_paths = [str(REPO_ROOT / r["image_path"]) for _, r in df.iterrows()]
    labels    = [int(class_to_idx[r["class_name"]]) for _, r in df.iterrows()]
    ds = tf.data.Dataset.from_tensor_slices((abs_paths, labels))
    if shuffle: ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_and_resize, num_parallel_calls=NUM_WORKERS)
    if augment: ds = ds.map(augment_image, num_parallel_calls=NUM_WORKERS)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def compute_class_weights(class_to_idx):
    from sklearn.utils.class_weight import compute_class_weight
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    y_train  = [class_to_idx[cn] for cn in train_df["class_name"]]
    classes  = np.array(sorted(class_to_idx.values()))
    weights  = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    w = dict(zip(classes, weights.tolist()))
    for cls, i in class_to_idx.items():
        if cls.startswith("Apple"): w[i] *= 1.5
    return w

start = datetime.datetime.now()
print("="*70)
print(f"STAGE 2 ONLY -- Fine-Tuning from Stage 1 checkpoint")
print(f"Unfreeze from layer {UNFREEZE_FROM}, lr={STAGE2_LR}, epochs={STAGE2_EPOCHS}")
print("="*70)

class_to_idx = load_class_mapping()
num_classes  = len(class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}

class_weights = compute_class_weights(class_to_idx)
print(f"Class weight range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}")
for cls, i in class_to_idx.items():
    if cls.startswith("Apple"): print(f"  Apple weight [{i}] {cls}: {class_weights[i]:.3f}")

train_ds = build_dataset(SPLITS_DIR / "train.csv",      class_to_idx, augment=True,  shuffle=True)
val_ds   = build_dataset(SPLITS_DIR / "validation.csv", class_to_idx, augment=False, shuffle=False)
print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")

print(f"\nLoading Stage 1 checkpoint: {BEST_MODEL}")
model = tf.keras.models.load_model(BEST_MODEL)

# Unfreeze base model layers from UNFREEZE_FROM
for layer in model.layers:
    if hasattr(layer, 'layers'):
        frozen = sum(1 for i, l in enumerate(layer.layers) if i < UNFREEZE_FROM)
        unfrozen = sum(1 for i, l in enumerate(layer.layers) if i >= UNFREEZE_FROM)
        for i, l in enumerate(layer.layers):
            l.trainable = i >= UNFREEZE_FROM
        print(f"Base model: {frozen} layers frozen, {unfrozen} layers trainable")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(BEST_MODEL), monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]

print(f"\nStarting Stage 2 training...")
h2 = model.fit(train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS,
               class_weight=class_weights, callbacks=callbacks, verbose=1)
h2_dict = {k: [float(v) for v in vals] for k, vals in h2.history.items()}
best_acc  = max(h2_dict["val_accuracy"])
best_loss = min(h2_dict["val_loss"])
print(f"\nStage 2 best val_accuracy: {best_acc:.4f} | val_loss: {best_loss:.4f}")

model.save(FINAL_MODEL)

# Per-class accuracy
print("\n" + "="*70)
print("PER-CLASS ACCURACY ON VALIDATION SET")
print("="*70)
best_model = tf.keras.models.load_model(BEST_MODEL)
correct = {i: 0 for i in range(num_classes)}
total   = {i: 0 for i in range(num_classes)}
for imgs, labels in val_ds:
    preds = best_model(imgs, training=False).numpy()
    pred_idx = np.argmax(preds, axis=-1)
    for t, p in zip(labels.numpy(), pred_idx):
        total[int(t)] += 1
        if int(t) == int(p): correct[int(t)] += 1
for i in range(num_classes):
    n = total[i]; c = correct[i]
    acc = c/n*100 if n > 0 else 0
    flag = " *** LOW ***" if acc < 70 else ""
    print(f"  [{i:2d}] {idx_to_class[i]:<45} {c:4d}/{n:4d}  ({acc:.1f}%){flag}")

# Quick test on user Apple images
import cv2
print("\n" + "="*70)
print("TEST ON YOUR UPLOADED APPLE IMAGES")
print("="*70)
test_imgs = [
    (r'C:\Users\agarw\.gemini\antigravity-ide\brain\142fb487-0eb0-4d80-b8a4-8b6dd10fa41e\.user_uploaded\media_1793007083287.jpg', 'Apple Rust (orange spots)'),
    (r'C:\Users\agarw\.gemini\antigravity-ide\brain\142fb487-0eb0-4d80-b8a4-8b6dd10fa41e\.user_uploaded\media_1793007078686.jpg', 'Apple Scab (dark spots)'),
]
for img_path, lbl in test_imgs:
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = tf.expand_dims(tf.image.resize(tf.convert_to_tensor(rgb, dtype=tf.float32), [224,224]) / 255.0, 0).numpy()
    probs = best_model.predict(t, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    print(f"\n{lbl}:")
    for i in top3: print(f"  {idx_to_class[i]}: {probs[i]*100:.2f}%")
    print(f"  WINNER: {idx_to_class[int(np.argmax(probs))]} ({max(probs)*100:.2f}%)")

duration = round((datetime.datetime.now() - start).total_seconds() / 60, 1)
print(f"\nTotal time: {duration} min")
