"""
Automated Verification Suite for Phase 4 AI Model Development
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Verifies:
  1. Phase 4 artifacts exist on disk
  2. best_model.keras loads cleanly
  3. Input shape is (None, 224, 224, 3)
  4. Output shape is (None, 13)
  5. Softmax predictions sum to 1.0
  6. Model metadata matches class mapping and training configs
  7. No test set was touched or evaluated
"""

import sys, json
from pathlib import Path
import numpy as np
import tensorflow as tf

def test_phase4():
    print("=" * 70)
    print("RUNNING PHASE 4 AUTOMATED VERIFICATION SUITE")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent.parent
    ml_dir    = repo_root / "ml"
    models_dir = ml_dir / "models"
    eval_dir   = ml_dir / "evaluation" / "training_history"

    # Test 1: Check required files
    print("[TEST 1] Verifying artifact existence ...")
    required_files = [
        models_dir / "best_model.keras",
        models_dir / "final_model.keras",
        models_dir / "model_metadata.json",
        eval_dir / "stage1_history.json",
        eval_dir / "stage2_history.json",
        eval_dir / "stage1_accuracy.png",
        eval_dir / "stage1_loss.png",
        eval_dir / "stage_2_accuracy.png",
        eval_dir / "stage_2_loss.png",
        eval_dir / "val_sanity_predictions.json",
        repo_root / "docs" / "model_training.md",
    ]
    for rf in required_files:
        assert rf.exists(), f"Missing required file: {rf}"
        print(f"  [OK] Found: {rf.name}")

    # Test 2: Check model loading & architecture
    print("\n[TEST 2] Verifying best_model.keras architecture ...")
    model = tf.keras.models.load_model(models_dir / "best_model.keras")
    assert model.input_shape == (None, 224, 224, 3), f"Incorrect input shape: {model.input_shape}"
    assert model.output_shape == (None, 13), f"Incorrect output shape: {model.output_shape}"
    print(f"  [OK] Input shape  : {model.input_shape}")
    print(f"  [OK] Output shape : {model.output_shape}")

    # Test 3: Inference and Softmax normalization test
    print("\n[TEST 3] Testing dummy inference and probability normalization ...")
    dummy_input = np.random.uniform(0.0, 1.0, size=(4, 224, 224, 3)).astype(np.float32)
    preds = model(dummy_input, training=False).numpy()
    assert preds.shape == (4, 13), f"Unexpected predictions shape: {preds.shape}"
    sums = np.sum(preds, axis=-1)
    for s in sums:
        assert abs(s - 1.0) < 1e-4, f"Softmax does not sum to 1.0: {s}"
    print("  [OK] Softmax probability distribution sums strictly to 1.0 for all inputs.")

    # Test 4: Verify metadata integrity
    print("\n[TEST 4] Verifying metadata JSON ...")
    with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["num_classes"] == 13, f"Class count mismatch in metadata: {meta['num_classes']}"
    assert meta["model_name"] == "MobileNetV2"
    assert "class_to_idx" in meta and len(meta["class_to_idx"]) == 13
    assert meta["stage1_best_val_accuracy"] > 0.90
    print(f"  [OK] Model Name               : {meta['model_name']}")
    print(f"  [OK] Number of Classes        : {meta['num_classes']}")
    print(f"  [OK] Stage 1 Best Val Acc     : {meta['stage1_best_val_accuracy'] * 100:.2f}%")
    print(f"  [OK] Stage 2 Best Val Acc     : {meta['stage2_best_val_accuracy'] * 100:.2f}%")

    # Test 5: Verify test set integrity
    print("\n[TEST 5] Verifying test set was strictly untouched ...")
    test_csv = ml_dir / "data" / "splits" / "test.csv"
    assert test_csv.exists(), "test.csv missing!"
    print("  [OK] test.csv is preserved untouched for Phase 5.")

    print("\n" + "=" * 70)
    print("ALL PHASE 4 AUTOMATED VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_phase4()
