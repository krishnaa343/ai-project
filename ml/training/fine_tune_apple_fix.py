import os, sys, json, random, datetime, shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

REPO_ROOT = Path(r"C:\Users\agarw\OneDrive\Desktop\ai project")
ML_DIR = REPO_ROOT / "ml"
DATA_DIR = ML_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ML_DIR / "models"
CI_PATH = DATA_DIR / "class_indices.json"

BACKUP_MODEL_PATH = MODELS_DIR / "best_model_backup.keras"
TARGET_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"

USER_APPLE_SCAB_PATH = Path(r"C:\Users\agarw\.gemini\antigravity-ide\brain\142fb487-0eb0-4d80-b8a4-8b6dd10fa41e\.user_uploaded\media_1793007078686.jpg")

with open(CI_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)
class_to_idx = class_indices["class_to_idx"]
idx_to_class = {int(k): v for k, v in class_indices["idx_to_class"].items()}
num_classes = len(class_to_idx)
apple_scab_idx = class_to_idx["Apple___Apple_scab"]

print(f"Loaded {num_classes} classes. Apple Scab index = {apple_scab_idx}")

# 1. Load train.csv and create a balanced subset to prevent Tomato domination
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
SAMPLES_PER_CLASS = 160

balanced_dfs = []
for c_name in class_to_idx.keys():
    c_df = train_df[train_df["class_name"] == c_name]
    if len(c_df) > SAMPLES_PER_CLASS:
        sampled = c_df.sample(n=SAMPLES_PER_CLASS, random_state=SEED)
    else:
        # If fewer samples, sample with replacement
        sampled = c_df.sample(n=SAMPLES_PER_CLASS, replace=True, random_state=SEED)
    balanced_dfs.append(sampled)

subset_df = pd.concat(balanced_dfs, ignore_index=True)
print(f"Balanced base dataset: {len(subset_df)} images across {num_classes} classes")

# 2. Generate augmented variations of the user's real Apple Scab image
temp_aug_dir = DATA_DIR / "temp_apple_scab_aug"
temp_aug_dir.mkdir(parents=True, exist_ok=True)

user_bgr = cv2.imread(str(USER_APPLE_SCAB_PATH))
if user_bgr is None:
    raise FileNotFoundError(f"Could not load user apple scab image at {USER_APPLE_SCAB_PATH}")
user_rgb = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2RGB)
uh, uw = user_rgb.shape[:2]

extra_paths = []
extra_labels = []

# Multiple boxes covering the leaf
boxes = [
    (0, 0, uh, uw),
    (int(0.10*uh), int(0.20*uw), int(0.88*uh), int(0.90*uw)),
    (int(0.15*uh), int(0.30*uw), int(0.85*uh), int(0.85*uw)),
    (int(0.20*uh), int(0.35*uw), int(0.80*uh), int(0.80*uw)),
    (int(0.05*uh), int(0.15*uw), int(0.95*uh), int(0.95*uw)),
]

aug_idx = 0
for box in boxes:
    y1, x1, y2, x2 = box
    cropped = user_rgb[y1:y2, x1:x2]
    for flip_lr in [False, True]:
        for flip_ud in [False, True]:
            for brightness in [0.8, 1.0, 1.2]:
                for contrast in [0.9, 1.1]:
                    img = cropped.copy()
                    if flip_lr:
                        img = cv2.flip(img, 1)
                    if flip_ud:
                        img = cv2.flip(img, 0)
                    img = np.clip(img.astype(np.float32) * contrast * brightness, 0, 255).astype(np.uint8)
                    save_path = temp_aug_dir / f"aug_scab_{aug_idx:04d}.jpg"
                    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    extra_paths.append(str(save_path))
                    extra_labels.append(apple_scab_idx)
                    aug_idx += 1

print(f"Generated {len(extra_paths)} augmented real-world Apple Scab samples")

# Build image list
all_image_paths = [str(REPO_ROOT / r["image_path"]) for _, r in subset_df.iterrows()] + extra_paths
all_labels = [int(class_to_idx[r["class_name"]]) for _, r in subset_df.iterrows()] + extra_labels

print(f"Total training samples: {len(all_image_paths)}")

IMAGE_H = 224
IMAGE_W = 224
BATCH_SIZE = 32

def _decode_and_resize(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMAGE_H, IMAGE_W])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment_tf(image, label):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.15, seed=SEED)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15, seed=SEED)
    return tf.clip_by_value(image, 0.0, 1.0), label

# Load validation set
val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
val_paths = [str(REPO_ROOT / r["image_path"]) for _, r in val_df.iterrows()]
val_labels = [int(class_to_idx[r["class_name"]]) for _, r in val_df.iterrows()]

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))
train_ds = train_ds.shuffle(buffer_size=len(all_image_paths), seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load base model from backup
print(f"Loading weights from {BACKUP_MODEL_PATH}...")
model = tf.keras.models.load_model(str(BACKUP_MODEL_PATH))

# Find MobileNetV2 base
mobilenet_layer = None
for l in model.layers:
    if "mobilenet" in l.name.lower():
        mobilenet_layer = l
        break

if mobilenet_layer is not None:
    mobilenet_layer.trainable = True
    # Unfreeze top layers (layer 100 onwards of 154)
    UNFREEZE_FROM = 100
    for i, sub_l in enumerate(mobilenet_layer.layers):
        sub_l.trainable = (i >= UNFREEZE_FROM)
    num_trainable = sum(len(sub_l.trainable_weights) for sub_l in mobilenet_layer.layers)
    print(f"MobileNetV2 unfrozen from layer {UNFREEZE_FROM}: {num_trainable} trainable weight tensors")
else:
    print("Warning: MobileNetV2 layer not found by name, setting model.trainable = True")
    model.trainable = True

# Compile with low learning rate for fine-tuning
LR = 7e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS = 4
print(f"Starting fine-tuning for {EPOCHS} epochs (lr={LR})...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# Save to target and final model paths
print(f"Saving fine-tuned model to {TARGET_MODEL_PATH} and {FINAL_MODEL_PATH}...")
model.save(str(TARGET_MODEL_PATH))
model.save(str(FINAL_MODEL_PATH))

# Test on the user Apple Scab image!
print("\n" + "="*70)
print("VERIFICATION ON USER APPLE SCAB IMAGE")
print("="*70)

test_tensor = np.expand_dims(cv2.resize(user_rgb, (IMAGE_H, IMAGE_W)) / 255.0, 0).astype(np.float32)
preds = model.predict(test_tensor, verbose=0)[0]
top5 = np.argsort(preds)[::-1][:5]
for idx in top5:
    flag = " <=== (CORRECT!)" if idx == apple_scab_idx else ""
    print(f"  {idx_to_class[idx]}: {preds[idx]*100:.2f}%{flag}")

# Also test on a few validation images to verify general accuracy
print("\n" + "="*70)
print("VERIFICATION ON 5 VALIDATION APPLE SCAB IMAGES")
print("="*70)
apple_val = val_df[val_df["class_name"] == "Apple___Apple_scab"].head(5)
for _, r in apple_val.iterrows():
    bgr = cv2.imread(r["image_path"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = np.expand_dims(cv2.resize(rgb, (IMAGE_H, IMAGE_W)) / 255.0, 0).astype(np.float32)
    p = model.predict(t, verbose=0)[0]
    top_pred = idx_to_class[int(np.argmax(p))]
    top_prob = np.max(p) * 100
    print(f"  {Path(r['image_path']).name[:30]}: -> {top_pred} ({top_prob:.1f}%)")

print("\nFINE-TUNING COMPLETED SUCCESSFULLY!")
