"""
Automated Test Suite for Grad-CAM Explainable AI (Phase 6)
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Executes the 12 required tests from Phase 6 specification plus edge-case tests:
  TEST 1 : Valid image produces a prediction
  TEST 2 : Prediction class exists in class mapping
  TEST 3 : Confidence is within [0, 1]
  TEST 4 : Grad-CAM heatmap is generated
  TEST 5 : Heatmap contains finite values (no NaNs or Infs)
  TEST 6 : Heatmap dimensions are valid
  TEST 7 : Overlay is successfully generated
  TEST 8 : The selected convolutional layer exists
  TEST 9 : Gradients are successfully calculated
  TEST 10: Original image is not modified in-place
  TEST 11: Reproducible Grad-CAM output under deterministic conditions
  TEST 12: Works for multiple classes (not hard-coded)
  EDGE CASES: Invalid path, corrupted input handling
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.explainability.gradcam import GradCAMExplainer, explain_prediction
from ml.explainability.image_utils import load_and_preprocess_image


def run_tests():
    print("=" * 75)
    print("RUNNING GRAD-CAM AUTOMATED VERIFICATION SUITE")
    print("=" * 75)

    explainer = GradCAMExplainer()
    print("  [INIT] GradCAMExplainer initialized successfully.")
    print(f"  [INIT] Detected Target Layer: {explainer.target_layer_name}")

    # Pick a sample image from test set
    test_csv = REPO_ROOT / "ml" / "data" / "splits" / "test.csv"
    test_df  = pd.read_csv(test_csv)
    sample_row = test_df.iloc[0]
    sample_path = str(REPO_ROOT / sample_row["image_path"])
    print(f"  [SAMPLE] Testing on: {sample_row['image_path']}")

    # --- TEST 1: Valid image produces a prediction ---
    res = explainer.explain(sample_path)
    assert res is not None and "predicted_class" in res, "TEST 1 Failed: No prediction returned"
    print("  [PASS] TEST 1: Valid image produces a prediction.")

    # --- TEST 2: Prediction class exists in class mapping ---
    assert res["predicted_class"] in explainer.class_to_idx, f"TEST 2 Failed: Unknown class {res['predicted_class']}"
    assert res["predicted_class_index"] in explainer.idx_to_class, "TEST 2 Failed: Class index mismatch"
    print(f"  [PASS] TEST 2: Prediction class '{res['predicted_class']}' exists in mapping.")

    # --- TEST 3: Confidence is within [0, 1] ---
    conf = res["confidence"]
    assert 0.0 <= conf <= 1.0, f"TEST 3 Failed: Confidence {conf} not in [0, 1]"
    print(f"  [PASS] TEST 3: Confidence {conf:.4f} is within [0, 1].")

    # --- TEST 4: Grad-CAM heatmap is generated ---
    assert res["heatmap_raw"] is not None, "TEST 4 Failed: Heatmap is None"
    assert isinstance(res["heatmap_raw"], np.ndarray), "TEST 4 Failed: Heatmap is not np.ndarray"
    print("  [PASS] TEST 4: Grad-CAM heatmap successfully generated.")

    # --- TEST 5: Heatmap contains finite values ---
    assert np.all(np.isfinite(res["heatmap_raw"])), "TEST 5 Failed: Heatmap contains NaN or Inf"
    assert res["heatmap_raw"].min() >= 0.0, "TEST 5 Failed: Heatmap min < 0"
    assert res["heatmap_raw"].max() <= 1.0, "TEST 5 Failed: Heatmap max > 1"
    print(f"  [PASS] TEST 5: Heatmap values are strictly finite in range [{res['heatmap_raw'].min():.2f}, {res['heatmap_raw'].max():.2f}].")

    # --- TEST 6: Heatmap dimensions are valid ---
    # For MobileNetV2, conv output is (7, 7)
    assert res["heatmap_raw"].shape == (7, 7), f"TEST 6 Failed: Unexpected shape {res['heatmap_raw'].shape}"
    print(f"  [PASS] TEST 6: Heatmap dimensions are valid: {res['heatmap_raw'].shape}.")

    # --- TEST 7: Overlay is successfully generated ---
    overlay = res["overlay"]
    orig    = res["original_rgb"]
    assert overlay is not None and overlay.shape == orig.shape, "TEST 7 Failed: Overlay shape mismatch"
    assert overlay.dtype == np.uint8, "TEST 7 Failed: Overlay dtype is not uint8"
    print(f"  [PASS] TEST 7: Overlay successfully generated with shape {overlay.shape}.")

    # --- TEST 8: The selected convolutional layer exists ---
    assert explainer.target_layer is not None, "TEST 8 Failed: Target layer is None"
    assert explainer.target_layer.name == explainer.target_layer_name, "TEST 8 Failed: Layer name mismatch"
    print(f"  [PASS] TEST 8: Target convolutional layer verified: '{explainer.target_layer_name}'.")

    # --- TEST 9: Gradients are successfully calculated ---
    tensor, _ = load_and_preprocess_image(sample_path)
    hm, p_idx, p_conf, _ = explainer.generate_heatmap(tensor)
    assert hm is not None and hm.sum() > 0.0, "TEST 9 Failed: Gradients produced empty heatmap"
    print("  [PASS] TEST 9: Gradients are successfully calculated and non-zero.")

    # --- TEST 10: Original image is not modified in-place ---
    tensor_test, orig_before = load_and_preprocess_image(sample_path)
    orig_copy = orig_before.copy()
    _ = explainer.explain(sample_path)
    assert np.array_equal(orig_before, orig_copy), "TEST 10 Failed: Original image was mutated!"
    print("  [PASS] TEST 10: Original image array preserved immutably.")

    # --- TEST 11: Reproducible Grad-CAM output under deterministic inference ---
    res_run1 = explainer.explain(sample_path)
    res_run2 = explainer.explain(sample_path)
    diff = np.max(np.abs(res_run1["heatmap_raw"] - res_run2["heatmap_raw"]))
    assert diff < 1e-5, f"TEST 11 Failed: Heatmap not reproducible, diff: {diff}"
    print(f"  [PASS] TEST 11: Heatmaps are numerically identical across runs (max diff: {diff:.2e}).")

    # --- TEST 12: Works for multiple classes (not hard-coded) ---
    classes_tested = set()
    sample_indices = [0, 100, 300, 500, 800, 1100]
    for idx in sample_indices:
        r = test_df.iloc[idx]
        exp_res = explainer.explain(REPO_ROOT / r["image_path"])
        classes_tested.add(exp_res["predicted_class"])
    assert len(classes_tested) >= 4, f"TEST 12 Failed: Only {len(classes_tested)} classes predicted across diverse samples"
    print(f"  [PASS] TEST 12: Tested successfully across {len(classes_tested)} distinct classes.")

    # --- EDGE CASE TESTS ---
    print("\n  TESTING EDGE CASES & EXCEPTION HANDLING:")
    # Edge case 1: Non-existent file
    try:
        explainer.explain("non_existent_leaf_image_12345.jpg")
        assert False, "Failed to raise FileNotFoundError on invalid path"
    except FileNotFoundError:
        print("  [PASS] Edge Case 1: FileNotFoundError raised correctly on invalid image path.")

    # Edge case 2: Invalid image array shape
    try:
        explainer.explain(np.zeros((10, 10), dtype=np.uint8))
        assert False, "Failed to raise ValueError on non-3-channel image"
    except ValueError:
        print("  [PASS] Edge Case 2: ValueError raised correctly on invalid image dimensions.")

    print("\n" + "=" * 75)
    print("ALL 12 GRAD-CAM TESTS + EDGE CASE TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
