"""
Main Training Script -- Two-Stage Transfer Learning
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Usage:
    python ml/training/train.py

Stages:
    1. Sanity check (2 epochs on small batch subset)
    2. Stage 1: Frozen base, head-only training (up to STAGE1_EPOCHS)
    3. Stage 2: Partial fine-tuning of top base layers (up to STAGE2_EPOCHS)
    4. Save best model, history, metadata, and sanity predictions
"""

import os, sys, json, random, datetime
from pathlib import Path

# Force UTF-8 on Windows terminal/subprocesses to prevent charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- no display needed
import matplotlib.pyplot as plt

# -- Reproducibility seeds -----------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -- Import project modules ----------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import (
        REPO_ROOT, MODELS_DIR, TRAINING_HISTORY_DIR, BEST_MODEL_PATH,
        FINAL_MODEL_PATH, METADATA_PATH,
        INPUT_SHAPE, BATCH_SIZE, MODEL_NAME, RANDOM_SEED,
        STAGE1_EPOCHS, STAGE1_LEARNING_RATE, STAGE1_EARLY_STOP_PAT, STAGE1_LR_REDUCE_PAT,
        STAGE1_LR_REDUCE_FACTOR, STAGE1_MIN_LR,
        STAGE2_EPOCHS, STAGE2_LEARNING_RATE, STAGE2_EARLY_STOP_PAT, STAGE2_LR_REDUCE_PAT,
        STAGE2_LR_REDUCE_FACTOR, STAGE2_MIN_LR, STAGE2_UNFREEZE_FROM,
        LOSS_FUNCTION, OPTIMIZER_NAME,
    )
    from data_loader import load_class_mapping, get_train_dataset, get_val_dataset, compute_class_weights
    from model_builder import build_stage1_model, prepare_stage2_model, print_model_summary
except ImportError:
    from ml.training.config import (
        REPO_ROOT, MODELS_DIR, TRAINING_HISTORY_DIR, BEST_MODEL_PATH,
        FINAL_MODEL_PATH, METADATA_PATH,
        INPUT_SHAPE, BATCH_SIZE, MODEL_NAME, RANDOM_SEED,
        STAGE1_EPOCHS, STAGE1_LEARNING_RATE, STAGE1_EARLY_STOP_PAT, STAGE1_LR_REDUCE_PAT,
        STAGE1_LR_REDUCE_FACTOR, STAGE1_MIN_LR,
        STAGE2_EPOCHS, STAGE2_LEARNING_RATE, STAGE2_EARLY_STOP_PAT, STAGE2_LR_REDUCE_PAT,
        STAGE2_LR_REDUCE_FACTOR, STAGE2_MIN_LR, STAGE2_UNFREEZE_FROM,
        LOSS_FUNCTION, OPTIMIZER_NAME,
    )
    from ml.training.data_loader import load_class_mapping, get_train_dataset, get_val_dataset, compute_class_weights
    from ml.training.model_builder import build_stage1_model, prepare_stage2_model, print_model_summary


# -----------------------------------------------------------------------------
# 0.  Environment Report
# -----------------------------------------------------------------------------
def report_environment():
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    import platform, psutil
    print(f"OS       : {platform.platform()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"TF       : {tf.__version__}")
    print(f"NumPy    : {np.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs     : {len(gpus)}")
    for g in gpus:
        print(f"  {g.name}")
    ram = psutil.virtual_memory()
    print(f"RAM      : {ram.total/1024**3:.1f} GB total  |  {ram.available/1024**3:.1f} GB free")
    if not gpus:
        print("NOTE     : No GPU detected. Training on CPU -- this will take longer.")
    print()


# -----------------------------------------------------------------------------
# 1.  Sanity Check  (fast -- 2 steps, verifies pipeline integrity)
# -----------------------------------------------------------------------------
def run_sanity_check(model, train_ds, val_ds, num_classes):
    print("=" * 70)
    print("SANITY CHECK  (2-step mini-run)")
    print("=" * 70)

    for imgs, labels in train_ds.take(1):
        assert imgs.shape[1:] == INPUT_SHAPE, f"Bad image shape: {imgs.shape}"
        assert labels.numpy().max() < num_classes, "Label index out of range!"
        preds = model(imgs[:2], training=False)
        assert preds.shape == (2, num_classes), f"Bad output shape: {preds.shape}"
        row_sums = tf.reduce_sum(preds, axis=-1).numpy()
        assert all(abs(s - 1.0) < 1e-4 for s in row_sums), "Softmax sums not close to 1"
        print(f"  Image batch shape   : {imgs.shape}")
        print(f"  Label range         : 0 - {labels.numpy().max()}")
        print(f"  Prediction shape    : {preds.shape}")
        print(f"  Softmax row sums    : {row_sums[:2]}")
    print("  [PASS] Sanity check complete -- pipeline is valid.\n")


# -----------------------------------------------------------------------------
# 2.  Callbacks factory
# -----------------------------------------------------------------------------
def make_callbacks(stage: int):
    patience_es = STAGE1_EARLY_STOP_PAT if stage == 1 else STAGE2_EARLY_STOP_PAT
    patience_lr = STAGE1_LR_REDUCE_PAT  if stage == 1 else STAGE2_LR_REDUCE_PAT
    lr_factor   = STAGE1_LR_REDUCE_FACTOR if stage == 1 else STAGE2_LR_REDUCE_FACTOR
    min_lr      = STAGE1_MIN_LR           if stage == 1 else STAGE2_MIN_LR

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy", save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lr_factor,
            patience=patience_lr, min_lr=min_lr, verbose=1
        ),
    ]


# -----------------------------------------------------------------------------
# 3.  History helpers
# -----------------------------------------------------------------------------
def save_history(history_dict, filename):
    path = TRAINING_HISTORY_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  History saved  -> {path}")
    return path


def plot_history(history_dict, stage_label):
    epochs = range(1, len(history_dict["accuracy"]) + 1)

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history_dict["accuracy"],     label="Train accuracy", marker="o")
    ax.plot(epochs, history_dict["val_accuracy"], label="Val accuracy",   marker="s")
    ax.set_title(f"{stage_label}: Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    acc_path = TRAINING_HISTORY_DIR / f"{stage_label.lower().replace(' ', '_')}_accuracy.png"
    fig.savefig(acc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Accuracy plot  -> {acc_path}")

    # Loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history_dict["loss"],     label="Train loss", marker="o")
    ax.plot(epochs, history_dict["val_loss"], label="Val loss",   marker="s")
    ax.set_title(f"{stage_label}: Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = TRAINING_HISTORY_DIR / f"{stage_label.lower().replace(' ', '_')}_loss.png"
    fig.savefig(loss_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss plot      -> {loss_path}")


# -----------------------------------------------------------------------------
# 4.  Validation sanity predictions
# -----------------------------------------------------------------------------
def run_val_sanity_predictions(model, val_ds, class_to_idx, n_samples=10):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    results = []
    count   = 0

    for imgs, labels in val_ds:
        preds      = model(imgs, training=False).numpy()
        pred_idx   = np.argmax(preds, axis=-1)
        confidence = np.max(preds, axis=-1)

        for i in range(len(imgs)):
            if count >= n_samples:
                break
            actual    = idx_to_class[int(labels[i])]
            predicted = idx_to_class[int(pred_idx[i])]
            conf_pct  = round(float(confidence[i]) * 100, 2)
            match     = bool(actual == predicted)
            results.append({
                "sample_idx": count + 1,
                "actual":    actual,
                "predicted": predicted,
                "confidence_pct": conf_pct,
                "correct": match
            })
            count += 1
        if count >= n_samples:
            break

    print("\n" + "=" * 70)
    print(f"VALIDATION SANITY PREDICTIONS (n={n_samples})")
    print("=" * 70)
    for r in results:
        status = "OK" if r["correct"] else "WRONG"
        print(f"  [{status:<5}] Actual: {r['actual']:<40} Predicted: {r['predicted']:<40} Conf: {r['confidence_pct']}%")

    path = TRAINING_HISTORY_DIR / "val_sanity_predictions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Predictions saved -> {path}")
    return results


# -----------------------------------------------------------------------------
# 5.  Main Training Orchestrator
# -----------------------------------------------------------------------------
def main():
    start_time = datetime.datetime.now()
    report_environment()

    # -- Load class mapping ---------------------------------------------------
    class_to_idx = load_class_mapping()
    num_classes  = len(class_to_idx)
    print(f"Classes loaded : {num_classes}  (from class_indices.json)")
    print(f"Sample mapping : {list(class_to_idx.items())[:3]} ...\n")

    # -- Build tf.data datasets -----------------------------------------------
    print("Building data pipelines ...")
    train_ds = get_train_dataset(class_to_idx)
    val_ds   = get_val_dataset(class_to_idx)
    print(f"  Train batches : {len(train_ds)}")
    print(f"  Val   batches : {len(val_ds)}\n")

    # -- Class weights (training data only) -----------------------------------
    print("Computing class weights from training data ...")
    class_weights = compute_class_weights(class_to_idx)
    print(f"  Weights computed. Range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}\n")

    stage1_hist_path = TRAINING_HISTORY_DIR / "stage1_history.json"

    # Check if Stage 1 is already completed
    if BEST_MODEL_PATH.exists() and stage1_hist_path.exists():
        print("=" * 70)
        print("FOUND COMPLETED STAGE 1 CHECKPOINT & HISTORY")
        print("=" * 70)
        print(f"  Loading best Stage 1 model from: {BEST_MODEL_PATH}")
        with open(stage1_hist_path, "r", encoding="utf-8") as f:
            h1_dict = json.load(f)
        best_s1_val_acc  = max(h1_dict["val_accuracy"])
        best_s1_val_loss = min(h1_dict["val_loss"])
        print(f"  Stage 1 best val_accuracy : {best_s1_val_acc:.4f}")
        print(f"  Stage 1 best val_loss     : {best_s1_val_loss:.4f}\n")
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
    else:
        # -- Build Stage 1 model ----------------------------------------------
        print("=" * 70)
        print("STAGE 1 -- Classifier Head Training (Base frozen)")
        print("=" * 70)
        model = build_stage1_model(num_classes)
        print_model_summary(model)

        # -- Sanity check before run ------------------------------------------
        run_sanity_check(model, train_ds, val_ds, num_classes)

        # -- Stage 1 Training -------------------------------------------------
        print(f"Training for up to {STAGE1_EPOCHS} epochs (lr={STAGE1_LEARNING_RATE}, batch={BATCH_SIZE}) ...\n")
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE1_EPOCHS,
            class_weight=class_weights,
            callbacks=make_callbacks(stage=1),
            verbose=1
        )

        h1_dict = {k: [float(v) for v in vals] for k, vals in h1.history.items()}
        save_history(h1_dict, "stage1_history.json")
        plot_history(h1_dict, "Stage 1")

        best_s1_val_acc  = max(h1_dict["val_accuracy"])
        best_s1_val_loss = min(h1_dict["val_loss"])
        print(f"\n  Stage 1 best val_accuracy : {best_s1_val_acc:.4f}")
        print(f"  Stage 1 best val_loss     : {best_s1_val_loss:.4f}\n")

    # -- Stage 2 -- Fine-Tuning -----------------------------------------------
    print("=" * 70)
    print(f"STAGE 2 -- Fine-Tuning (unfreeze base layers from idx {STAGE2_UNFREEZE_FROM})")
    print("=" * 70)
    model = prepare_stage2_model(model)
    print_model_summary(model)

    print(f"Training for up to {STAGE2_EPOCHS} epochs (lr={STAGE2_LEARNING_RATE}, batch={BATCH_SIZE}) ...\n")
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(stage=2),
        verbose=1
    )

    h2_dict = {k: [float(v) for v in vals] for k, vals in h2.history.items()}
    save_history(h2_dict, "stage2_history.json")
    plot_history(h2_dict, "Stage 2")

    best_s2_val_acc  = max(h2_dict["val_accuracy"])
    best_s2_val_loss = min(h2_dict["val_loss"])
    print(f"\n  Stage 2 best val_accuracy : {best_s2_val_acc:.4f}")
    print(f"  Stage 2 best val_loss     : {best_s2_val_loss:.4f}\n")

    # -- Save final model -----------------------------------------------------
    model.save(FINAL_MODEL_PATH)
    print(f"  Final model saved -> {FINAL_MODEL_PATH}")
    print(f"  Best model saved  -> {BEST_MODEL_PATH}  (saved by ModelCheckpoint callback)\n")

    # -- Validation sanity predictions ----------------------------------------
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    predictions = run_val_sanity_predictions(best_model, val_ds, class_to_idx)

    # -- Save model metadata --------------------------------------------------
    end_time = datetime.datetime.now()
    duration_mins = round((end_time - start_time).total_seconds() / 60, 1)

    metadata = {
        "model_name":           MODEL_NAME,
        "architecture_summary": {
            "base_model": "MobileNetV2 (ImageNet weights)",
            "pooling": "GlobalAveragePooling2D",
            "dropout": 0.3,
            "classification_head": f"Dense({num_classes}, softmax)"
        },
        "input_size":           list(INPUT_SHAPE),
        "num_classes":          num_classes,
        "class_to_idx":         class_to_idx,
        "preprocessing":        "RGB, resize 224x224, normalize [0,1]",
        "training_seed":        RANDOM_SEED,
        "batch_size":           BATCH_SIZE,
        "stage1_epochs_run":    len(h1_dict["loss"]),
        "stage1_lr":            STAGE1_LEARNING_RATE,
        "stage2_epochs_run":    len(h2_dict["loss"]),
        "stage2_lr":            STAGE2_LEARNING_RATE,
        "stage2_unfreeze_from": STAGE2_UNFREEZE_FROM,
        "optimizer":            OPTIMIZER_NAME,
        "loss_function":        LOSS_FUNCTION,
        "class_weights_used":   True,
        "augmentation":         ["random_horizontal_flip", "random_brightness(+-10%)", "random_contrast(+-10%)"],
        "stage1_best_val_accuracy": round(float(best_s1_val_acc), 6),
        "stage1_best_val_loss":     round(float(best_s1_val_loss), 6),
        "stage2_best_val_accuracy": round(float(best_s2_val_acc), 6),
        "stage2_best_val_loss":     round(float(best_s2_val_loss), 6),
        "training_duration_minutes": duration_mins,
        "trained_at": start_time.isoformat(),
        "best_model_path":  str(BEST_MODEL_PATH),
        "final_model_path": str(FINAL_MODEL_PATH),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved -> {METADATA_PATH}")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {duration_mins} minutes")
    print(f"  Stage 1 Best Val Accuracy: {best_s1_val_acc*100:.2f}% (loss: {best_s1_val_loss:.4f})")
    print(f"  Stage 2 Best Val Accuracy: {best_s2_val_acc*100:.2f}% (loss: {best_s2_val_loss:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
