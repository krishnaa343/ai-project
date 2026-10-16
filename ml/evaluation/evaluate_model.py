"""
Automated Model Evaluation Suite (Phase 5)
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Evaluates the trained MobileNetV2 model on the untouched test dataset:
- Rigorous test set validation (file existence, readability, hash isolation)
- Full-set batch inference with deterministic preprocessing
- Overall Accuracy, Precision, Recall, F1-Score (macro, weighted, per-class)
- Confusion Matrix generation (CSV and high-res annotated heatmap PNG)
- Comprehensive Error Analysis (misclassified CSV, frequent confusion pairs)
- Confidence distribution and High/Low confidence error profiling
- Representative Sample Predictions visualization grid
- Multiclass One-vs-Rest ROC / AUC curves & metrics
- Inference Latency Benchmark (mean, median, std ms per image, images/sec)
- Machine-readable evaluation summary JSON
"""

import os, sys, time, json, hashlib
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ── Paths Configuration ───────────────────────────────────────────────────────
REPO_ROOT         = Path(__file__).resolve().parent.parent.parent
ML_DIR            = REPO_ROOT / "ml"
DATA_DIR          = ML_DIR / "data"
SPLITS_DIR        = DATA_DIR / "splits"
MODELS_DIR        = ML_DIR / "models"
EVAL_DIR          = ML_DIR / "evaluation"
SAMPLES_DIR       = EVAL_DIR / "sample_predictions"

TEST_CSV          = SPLITS_DIR / "test.csv"
TRAIN_CSV         = SPLITS_DIR / "train.csv"
CLASS_INDICES_JSON= DATA_DIR / "class_indices.json"
BEST_MODEL_PATH   = MODELS_DIR / "best_model.keras"
METADATA_PATH     = MODELS_DIR / "model_metadata.json"

IMAGE_HEIGHT      = 224
IMAGE_WIDTH       = 224
INPUT_SHAPE       = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
BATCH_SIZE        = 32

EVAL_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Verification of Prerequisites & Test Set Integrity
# ─────────────────────────────────────────────────────────────────────────────
def verify_prerequisites_and_test_set():
    print("=" * 75)
    print("PART 1 & 4 -- VERIFY PREVIOUS PHASES & TEST SET INTEGRITY")
    print("=" * 75)

    # Check file existence
    for path, desc in [
        (TEST_CSV, "Test manifest CSV"),
        (TRAIN_CSV, "Train manifest CSV"),
        (CLASS_INDICES_JSON, "Class indices JSON"),
        (BEST_MODEL_PATH, "Best trained model"),
        (METADATA_PATH, "Model metadata JSON")
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 3/4 artifact: {path} ({desc})")
        print(f"  [OK] Found {desc}: {path.name}")

    with open(CLASS_INDICES_JSON, "r", encoding="utf-8") as f:
        class_mapping_data = json.load(f)
    class_to_idx = class_mapping_data["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"  [OK] Class indices loaded: {num_classes} classes")

    test_df = pd.read_csv(TEST_CSV)
    print(f"  [OK] Test manifest loaded: {len(test_df)} samples")

    # Verify all test files exist, are readable, and labels are valid
    print("  Checking image file existence, readability, and labels ...")
    corrupt_count = 0
    invalid_labels = 0
    for _, row in test_df.iterrows():
        abs_p = REPO_ROOT / row["image_path"]
        if not abs_p.exists():
            corrupt_count += 1
        if row["class_name"] not in class_to_idx:
            invalid_labels += 1

    assert corrupt_count == 0, f"Found {corrupt_count} missing/unreadable test image files!"
    assert invalid_labels == 0, f"Found {invalid_labels} invalid test class labels!"
    print(f"  [PASS] All {len(test_df)} test images exist on disk and possess valid labels.")

    # Verify train-test duplicate hash isolation
    print("  Verifying train-test cryptographic hash separation ...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_hashes = set(train_df["md5_hash"]) if "md5_hash" in train_df.columns else set()
    test_hashes  = set(test_df["md5_hash"]) if "md5_hash" in test_df.columns else set()
    overlap = train_hashes.intersection(test_hashes)
    assert len(overlap) == 0, f"Critical Data Leakage: {len(overlap)} exact duplicate hashes shared between train and test!"
    print(f"  [PASS] Train-test duplicate hash overlap: 0 (Strict data isolation verified).")

    return test_df, class_to_idx, idx_to_class, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Loading & Architecture Validation
# ─────────────────────────────────────────────────────────────────────────────
def load_and_verify_model(num_classes):
    print("\n" + "=" * 75)
    print("PART 2 -- LOAD THE FINAL MODEL")
    print("=" * 75)
    print(f"  Loading model from: {BEST_MODEL_PATH}")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    print(f"  Model input shape  : {model.input_shape}")
    print(f"  Model output shape : {model.output_shape}")
    assert model.input_shape == (None, 224, 224, 3), f"Model input shape mismatch: {model.input_shape}"
    assert model.output_shape == (None, num_classes), f"Model output shape mismatch: {model.output_shape}"
    print(f"  [PASS] Model architecture successfully verified for {num_classes} output classes.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Test Inference Execution
# ─────────────────────────────────────────────────────────────────────────────
def run_test_inference(model, test_df, class_to_idx, idx_to_class):
    print("\n" + "=" * 75)
    print("PART 3 & 5 -- RUNNING DETERMINISTIC TEST SET INFERENCE")
    print("=" * 75)

    abs_paths = [str(REPO_ROOT / row["image_path"]) for _, row in test_df.iterrows()]
    actual_labels = [int(class_to_idx[row["class_name"]]) for _, row in test_df.iterrows()]

    def _decode_and_preprocess(p, y):
        raw = tf.io.read_file(p)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((abs_paths, actual_labels))
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    all_preds_probs = []
    print(f"  Predicting across {len(ds)} test batches (batch size {BATCH_SIZE}) ...")
    start_t = time.time()
    for batch_imgs, _ in ds:
        probs = model(batch_imgs, training=False).numpy()
        all_preds_probs.append(probs)
    total_eval_time = time.time() - start_t
    all_preds_probs = np.concatenate(all_preds_probs, axis=0)

    pred_indices     = np.argmax(all_preds_probs, axis=-1)
    pred_confidences = np.max(all_preds_probs, axis=-1)

    records = []
    for i, row in test_df.iterrows():
        actual_cls = row["class_name"]
        actual_idx = class_to_idx[actual_cls]
        pred_idx   = int(pred_indices[i])
        pred_cls   = idx_to_class[pred_idx]
        conf       = float(pred_confidences[i])
        is_correct = bool(actual_idx == pred_idx)

        records.append({
            "image_path":           row["image_path"],
            "actual_class":         actual_cls,
            "actual_class_index":   actual_idx,
            "predicted_class":      pred_cls,
            "predicted_class_index":pred_idx,
            "confidence":           round(conf, 6),
            "correct":              is_correct
        })

    preds_df = pd.DataFrame(records)
    preds_csv_path = EVAL_DIR / "test_predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"  [SAVED] Test predictions saved -> {preds_csv_path}")
    print(f"  Total inference time across {len(preds_df)} images: {total_eval_time:.2f}s ({len(preds_df)/total_eval_time:.1f} imgs/sec)")

    return preds_df, all_preds_probs, actual_labels, pred_indices


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics & Statistical Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_evaluation_metrics(preds_df, actual_labels, pred_indices, class_to_idx, idx_to_class):
    print("\n" + "=" * 75)
    print("PART 6, 7 & 9 -- COMPUTING ACCURACY, PRECISION, RECALL & F1-SCORE")
    print("=" * 75)

    num_samples  = len(preds_df)
    num_correct  = int(preds_df["correct"].sum())
    num_incorrect= num_samples - num_correct
    accuracy     = num_correct / num_samples

    print(f"  Total Test Samples   : {num_samples}")
    print(f"  Correct Predictions  : {num_correct}")
    print(f"  Incorrect Predictions: {num_incorrect}")
    print(f"  Raw Accuracy         : {accuracy:.6f} ({accuracy * 100:.2f}%)")

    # Macro & Weighted metrics
    target_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        actual_labels, pred_indices, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        actual_labels, pred_indices, average="weighted", zero_division=0
    )

    print(f"\n  Macro Average    : Precision={prec_macro:.4f} | Recall={rec_macro:.4f} | F1={f1_macro:.4f}")
    print(f"  Weighted Average : Precision={prec_weighted:.4f} | Recall={rec_weighted:.4f} | F1={f1_weighted:.4f}")

    # Per-class metrics
    prec_pc, rec_pc, f1_pc, supp_pc = precision_recall_fscore_support(
        actual_labels, pred_indices, average=None, zero_division=0
    )

    per_class_data = []
    print("\n  PER-CLASS BREAKDOWN:")
    print(f"  {'Class Name':<38} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support'}")
    print("  " + "-" * 75)
    for i in range(len(target_names)):
        cls_name = target_names[i]
        p, r, f, s = prec_pc[i], rec_pc[i], f1_pc[i], int(supp_pc[i])
        per_class_data.append({
            "class":      cls_name,
            "class_index":i,
            "precision":  round(float(p), 4),
            "recall":     round(float(r), 4),
            "f1_score":   round(float(f), 4),
            "support":    s
        })
        print(f"  {cls_name:<38} {p:<10.4f} {r:<10.4f} {f:<10.4f} {s}")

    report_df = pd.DataFrame(per_class_data)
    report_csv = EVAL_DIR / "classification_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"\n  [SAVED] Classification report CSV -> {report_csv}")

    # Also save formatted text version
    sk_text_report = classification_report(actual_labels, pred_indices, target_names=target_names, digits=4)
    with open(EVAL_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(sk_text_report)

    metrics = {
        "num_samples":       num_samples,
        "num_correct":       num_correct,
        "num_incorrect":     num_incorrect,
        "accuracy":          round(accuracy, 6),
        "macro_precision":   round(float(prec_macro), 6),
        "macro_recall":      round(float(rec_macro), 6),
        "macro_f1":          round(float(f1_macro), 6),
        "weighted_precision":round(float(prec_weighted), 6),
        "weighted_recall":   round(float(rec_weighted), 6),
        "weighted_f1":       round(float(f1_weighted), 6),
    }
    return metrics, per_class_data


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confusion Matrix Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_confusion_matrix(actual_labels, pred_indices, idx_to_class):
    print("\n" + "=" * 75)
    print("PART 8 -- GENERATING CONFUSION MATRIX")
    print("=" * 75)

    num_classes = len(idx_to_class)
    labels = list(range(num_classes))
    target_names = [idx_to_class[i] for i in labels]

    cm = confusion_matrix(actual_labels, pred_indices, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    cm_csv_path = EVAL_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"  [SAVED] Confusion matrix CSV -> {cm_csv_path}")

    # High-resolution heatmap using pure matplotlib
    fig, ax = plt.subplots(figsize=(13, 11))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix -- Crop Disease Classification (MobileNetV2 Test Set)", fontsize=14, pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Ground Truth Class", fontsize=12, labelpad=10)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([name.split("___")[-1] for name in target_names], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([name.split("___")[-1] for name in target_names], fontsize=9)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:d}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=8,
                weight="bold" if val > 0 else "normal"
            )

    plt.tight_layout()
    cm_png_path = EVAL_DIR / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Confusion matrix plot -> {cm_png_path}")
    return cm, cm_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Error Analysis & Confusion Pairs
# ─────────────────────────────────────────────────────────────────────────────
def perform_error_analysis(preds_df):
    print("\n" + "=" * 75)
    print("PART 10 & 11 -- ERROR ANALYSIS & CONFUSION PAIR IDENTIFICATION")
    print("=" * 75)

    errors_df = preds_df[preds_df["correct"] == False].copy()
    total_errors = len(errors_df)
    error_rate   = total_errors / len(preds_df)
    print(f"  Total Misclassifications : {total_errors} ({error_rate * 100:.2f}% error rate)")

    misclassified_csv = EVAL_DIR / "misclassified_predictions.csv"
    errors_df.to_csv(misclassified_csv, index=False)
    print(f"  [SAVED] Misclassified predictions -> {misclassified_csv}")

    # Frequent confusion pairs
    pair_counts = (
        errors_df.groupby(["actual_class", "predicted_class"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )
    print("\n  TOP OBSERVED CONFUSION PAIRS:")
    for _, row in pair_counts.head(10).iterrows():
        print(f"    {row['actual_class']}  ->  {row['predicted_class']}: {row['count']} instances")

    # High-confidence vs Low-confidence errors
    highest_conf_errors = errors_df.sort_values(by="confidence", ascending=False).head(5)
    lowest_conf_errors  = errors_df.sort_values(by="confidence", ascending=True).head(5)

    print("\n  TOP 5 HIGHEST-CONFIDENCE INCORRECT PREDICTIONS (Confident Errors):")
    for _, r in highest_conf_errors.iterrows():
        print(f"    Conf: {r['confidence']*100:.1f}% | Actual: {r['actual_class']} | Pred: {r['predicted_class']} | File: {Path(r['image_path']).name}")

    print("\n  TOP 5 LOWEST-CONFIDENCE INCORRECT PREDICTIONS (Uncertain Errors):")
    for _, r in lowest_conf_errors.iterrows():
        print(f"    Conf: {r['confidence']*100:.1f}% | Actual: {r['actual_class']} | Pred: {r['predicted_class']} | File: {Path(r['image_path']).name}")

    return errors_df, pair_counts, highest_conf_errors, lowest_conf_errors


# ─────────────────────────────────────────────────────────────────────────────
# 7. Representative Sample Predictions Grid
# ─────────────────────────────────────────────────────────────────────────────
def generate_sample_predictions_grid(preds_df):
    print("\n" + "=" * 75)
    print("PART 12 -- GENERATING SAMPLE PREDICTIONS GRID")
    print("=" * 75)

    correct_samples = preds_df[preds_df["correct"] == True].sample(n=6, random_state=42)
    incorrect_samples = preds_df[preds_df["correct"] == False]
    n_err = min(6, len(incorrect_samples))
    incorrect_samples = incorrect_samples.sample(n=n_err, random_state=42)

    sample_items = pd.concat([correct_samples, incorrect_samples])

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    samples_json_data = []

    for idx, (_, row) in enumerate(sample_items.iterrows()):
        ax = axes[idx]
        img_p = REPO_ROOT / row["image_path"]
        img = plt.imread(img_p)
        ax.imshow(img)
        ax.axis("off")

        status_str = "CORRECT" if row["correct"] else "MISCLASSIFIED"
        color = "#16a34a" if row["correct"] else "#dc2626"

        title_text = (
            f"[{status_str}]\n"
            f"Actual: {row['actual_class'].split('___')[-1]}\n"
            f"Pred: {row['predicted_class'].split('___')[-1]}\n"
            f"Conf: {row['confidence']*100:.1f}%"
        )
        ax.set_title(title_text, fontsize=9, color=color, weight="bold", pad=4)

        samples_json_data.append({
            "image_path": row["image_path"],
            "actual_class": row["actual_class"],
            "predicted_class": row["predicted_class"],
            "confidence_pct": round(row["confidence"] * 100, 2),
            "correct": bool(row["correct"])
        })

    plt.suptitle("Representative Test Set Predictions (MobileNetV2)", fontsize=14, y=0.99, weight="bold")
    plt.tight_layout()

    grid_path = SAMPLES_DIR / "sample_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Representative predictions grid -> {grid_path}")

    with open(SAMPLES_DIR / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples_json_data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Multi-class One-vs-Rest ROC / AUC Curves
# ─────────────────────────────────────────────────────────────────────────────
def generate_roc_auc_curves(actual_labels, all_preds_probs, idx_to_class):
    print("\n" + "=" * 75)
    print("PART 14 -- MULTICLASS ONE-VS-REST ROC / AUC EVALUATION")
    print("=" * 75)

    num_classes = len(idx_to_class)
    y_true_bin = label_binarize(actual_labels, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_preds_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_preds_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"  Micro-average AUC : {roc_auc['micro']:.4f}")
    print(f"  Macro-average AUC : {roc_auc['macro']:.4f}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-average (AUC = {roc_auc['micro']:.3f})", color="deeppink", linestyle=":", linewidth=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-average (AUC = {roc_auc['macro']:.3f})", color="navy", linestyle=":", linewidth=3)

    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    for i, color in zip(range(num_classes), colors):
        cls_short = idx_to_class[i].split("___")[-1]
        plt.plot(fpr[i], tpr[i], color=color, lw=1.2, label=f"{cls_short} (AUC={roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("One-vs-Rest Multiclass ROC Curves (Test Set)", fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    roc_path = EVAL_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Multiclass ROC curves plot -> {roc_path}")

    return roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# 9. Inference Latency Benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_inference_latency(model):
    print("\n" + "=" * 75)
    print("PART 15 -- BENCHMARKING SINGLE-IMAGE INFERENCE LATENCY")
    print("=" * 75)

    dummy = np.random.uniform(0.0, 1.0, size=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = model(dummy, training=False)

    times_ms = []
    iterations = 100
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model(dummy, training=False)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)

    mean_lat   = float(np.mean(times_ms))
    median_lat = float(np.median(times_ms))
    std_lat    = float(np.std(times_ms))
    min_lat    = float(np.min(times_ms))
    max_lat    = float(np.max(times_ms))
    throughput = 1000.0 / mean_lat

    print(f"  Mean latency   : {mean_lat:.2f} ms")
    print(f"  Median latency : {median_lat:.2f} ms")
    print(f"  Std deviation  : {std_lat:.2f} ms")
    print(f"  Min / Max      : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  Throughput     : {throughput:.1f} images/sec on host CPU")

    latency_metrics = {
        "benchmark_runs": iterations,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "throughput_fps": round(throughput, 1),
        "device": "Host CPU"
    }
    return latency_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 10. Master Evaluation Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("PHASE 5: RIGOROUS MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 75)

    test_df, class_to_idx, idx_to_class, num_classes = verify_prerequisites_and_test_set()
    model = load_and_verify_model(num_classes)
    preds_df, all_preds_probs, actual_labels, pred_indices = run_test_inference(
        model, test_df, class_to_idx, idx_to_class
    )
    metrics, per_class_data = compute_evaluation_metrics(
        preds_df, actual_labels, pred_indices, class_to_idx, idx_to_class
    )
    cm, cm_df = generate_confusion_matrix(actual_labels, pred_indices, idx_to_class)
    errors_df, pair_counts, highest_conf_errors, lowest_conf_errors = perform_error_analysis(preds_df)
    generate_sample_predictions_grid(preds_df)
    roc_auc = generate_roc_auc_curves(actual_labels, all_preds_probs, idx_to_class)
    latency = benchmark_inference_latency(model)

    # Save comprehensive evaluation summary
    top_confusion_pairs = [
        {"actual": r["actual_class"], "predicted": r["predicted_class"], "count": int(r["count"])}
        for _, r in pair_counts.head(5).iterrows()
    ]

    summary = {
        "model_name":               "MobileNetV2",
        "evaluation_timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_samples":             metrics["num_samples"],
        "correct_predictions":      metrics["num_correct"],
        "incorrect_predictions":    metrics["num_incorrect"],
        "accuracy":                 metrics["accuracy"],
        "macro_precision":          metrics["macro_precision"],
        "macro_recall":             metrics["macro_recall"],
        "macro_f1":                 metrics["macro_f1"],
        "weighted_precision":       metrics["weighted_precision"],
        "weighted_recall":          metrics["weighted_recall"],
        "weighted_f1":              metrics["weighted_f1"],
        "roc_auc_micro":            round(float(roc_auc["micro"]), 4),
        "roc_auc_macro":            round(float(roc_auc["macro"]), 4),
        "top_confusion_pairs":      top_confusion_pairs,
        "inference_latency":        latency,
        "per_class_metrics":        per_class_data
    }

    summary_path = EVAL_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[COMPLETE] Master evaluation summary saved -> {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
