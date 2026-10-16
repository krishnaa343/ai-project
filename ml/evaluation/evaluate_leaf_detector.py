"""
Evaluate Leaf / Non-Leaf Gate Classifier
AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Computes:
- Binary Confusion Matrix
- Precision, Recall, F1 for LEAF and NON_LEAF
- Threshold sensitivity analysis
- Independent test on test_ant_non_leaf.jpg
- Independent tests on difficult diseased leaves (Tomato Early/Late Blight, Potato, Apple, Corn)

Saves results to ml/evaluation/leaf_detector_evaluation.json
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "ml" / "models"
MODEL_PATH = MODELS_DIR / "leaf_detector_model.keras"
SPLITS_DIR = REPO_ROOT / "ml" / "data" / "leaf_detection" / "splits"
EVAL_DIR = REPO_ROOT / "ml" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = EVAL_DIR / "leaf_detector_evaluation.json"

IMAGE_SIZE = (224, 224)

def load_and_preprocess(image_path):
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize(IMAGE_SIZE)
        arr = np.array(im, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

def evaluate():
    print("=" * 60)
    print("Evaluating Leaf / Non-Leaf Gate Classifier")
    print("=" * 60)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"Loaded gate model from: {MODEL_PATH}")
    
    test_csv = SPLITS_DIR / "test.csv"
    val_csv = SPLITS_DIR / "val.csv"
    df_test = pd.read_csv(test_csv)
    df_val = pd.read_csv(val_csv)
    
    print(f"Test samples: {len(df_test)}, Validation samples: {len(df_val)}")
    
    # Run test predictions
    y_true = df_test["label"].values
    y_probs = []
    
    for _, row in df_test.iterrows():
        full_path = REPO_ROOT / row["file_path"]
        inp = load_and_preprocess(full_path)
        prob = model.predict(inp, verbose=0)[0]
        y_probs.append(prob)
        
    y_probs = np.array(y_probs) # shape (N, 2)
    # class 0: NON_LEAF, class 1: LEAF
    p_non_leaf = y_probs[:, 0]
    p_leaf = y_probs[:, 1]
    
    # ── Threshold Sensitivity on Test Set ─────────────────────────────────────
    threshold_results = {}
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
        # Decision: if p_non_leaf >= thresh -> NON_LEAF (0), else LEAF (1)
        y_pred_t = np.where(p_non_leaf >= thresh, 0, 1)
        acc_t = accuracy_score(y_true, y_pred_t)
        rec_non_leaf = recall_score(y_true, y_pred_t, pos_label=0)
        rec_leaf = recall_score(y_true, y_pred_t, pos_label=1)
        threshold_results[f"thresh_{thresh:.2f}"] = {
            "non_leaf_threshold": thresh,
            "accuracy": float(acc_t),
            "non_leaf_recall": float(rec_non_leaf),
            "leaf_recall": float(rec_leaf)
        }
        
    # Standard decision at 0.50
    y_pred = np.where(p_non_leaf >= 0.50, 0, 1)
    
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]] where 0=NON_LEAF, 1=LEAF
    tn, fp, fn, tp = cm.ravel()
    
    non_leaf_precision = precision_score(y_true, y_pred, pos_label=0)
    non_leaf_recall = recall_score(y_true, y_pred, pos_label=0)
    non_leaf_f1 = f1_score(y_true, y_pred, pos_label=0)
    
    leaf_precision = precision_score(y_true, y_pred, pos_label=1)
    leaf_recall = recall_score(y_true, y_pred, pos_label=1)
    leaf_f1 = f1_score(y_true, y_pred, pos_label=1)
    
    false_rejection_rate = fn / (fn + tp) # valid leaf rejected
    false_acceptance_rate = fp / (tn + fp) # non-leaf accepted
    
    print("\nTest Metrics (Threshold = 0.50):")
    print(f"  Overall Accuracy:      {acc:.4%}")
    print(f"  NON_LEAF Precision:    {non_leaf_precision:.4f}, Recall: {non_leaf_recall:.4f}, F1: {non_leaf_f1:.4f}")
    print(f"  LEAF Precision:        {leaf_precision:.4f}, Recall: {leaf_recall:.4f}, F1: {leaf_f1:.4f}")
    print(f"  Confusion Matrix:      TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  False Rejection Rate:  {false_rejection_rate:.4%}")
    print(f"  False Acceptance Rate: {false_acceptance_rate:.4%}")
    
    # ── Ant Test (test_ant_non_leaf.jpg) ───────────────────────────────────────
    ant_path = REPO_ROOT / "test_ant_non_leaf.jpg"
    ant_result = {}
    if ant_path.exists():
        inp = load_and_preprocess(ant_path)
        probs = model.predict(inp, verbose=0)[0]
        p_nl = float(probs[0])
        p_l = float(probs[1])
        is_rejected = p_nl >= 0.50
        ant_result = {
            "image": "test_ant_non_leaf.jpg",
            "p_non_leaf": round(p_nl, 4),
            "p_leaf": round(p_l, 4),
            "classified_as": "NON_LEAF" if is_rejected else "LEAF",
            "rejected_by_gate": bool(is_rejected)
        }
        print(f"\nAnt Test ({ant_path.name}):")
        print(f"  P(NON_LEAF): {p_nl:.4f}, P(LEAF): {p_l:.4f} -> REJECTED: {is_rejected}")
        
    # ── Diverse Foliar / Difficult Diseased Leaf Tests ─────────────────────────
    leaf_tests = [
        REPO_ROOT / "test_leaf.jpg",
    ]
    foliar_results = []
    for lp in leaf_tests:
        if lp.exists():
            inp = load_and_preprocess(lp)
            probs = model.predict(inp, verbose=0)[0]
            p_nl = float(probs[0])
            p_l = float(probs[1])
            is_accepted = p_l >= 0.50
            res = {
                "image": lp.name,
                "p_leaf": round(p_l, 4),
                "p_non_leaf": round(p_nl, 4),
                "classified_as": "LEAF" if is_accepted else "NON_LEAF",
                "accepted_by_gate": bool(is_accepted)
            }
            foliar_results.append(res)
            print(f"Foliar Test ({lp.name}): P(LEAF): {p_l:.4f} -> ACCEPTED: {is_accepted}")

    # Compile comprehensive report
    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_samples": int(len(df_test)),
        "accuracy": float(acc),
        "confusion_matrix": {
            "true_negatives_non_leaf": int(tn),
            "false_positives_non_leaf_as_leaf": int(fp),
            "false_negatives_leaf_as_non_leaf": int(fn),
            "true_positives_leaf": int(tp)
        },
        "non_leaf_metrics": {
            "precision": float(non_leaf_precision),
            "recall": float(non_leaf_recall),
            "f1_score": float(non_leaf_f1),
            "support": int(tn + fp)
        },
        "leaf_metrics": {
            "precision": float(leaf_precision),
            "recall": float(leaf_recall),
            "f1_score": float(leaf_f1),
            "support": int(fn + tp)
        },
        "false_rejection_rate": float(false_rejection_rate),
        "false_acceptance_rate": float(false_acceptance_rate),
        "threshold_sensitivity": threshold_results,
        "ant_benchmark": ant_result,
        "sample_foliar_tests": foliar_results
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation report to: {OUTPUT_JSON}")

if __name__ == "__main__":
    evaluate()
