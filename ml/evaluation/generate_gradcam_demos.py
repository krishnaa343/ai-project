"""
Grad-CAM Demonstration & Benchmarking Script (Phase 6)
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Selects diverse representative samples from the test set (both correct & misclassified),
computes Grad-CAM heatmaps & overlays, saves image artifacts, benchmarks latencies,
and generates a composite multi-panel visualization report.
"""

import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.explainability.gradcam import GradCAMExplainer
from ml.explainability.image_utils import save_image_rgb

EVAL_DIR    = REPO_ROOT / "ml" / "evaluation"
GRADCAM_DIR = EVAL_DIR / "gradcam"
ORIG_DIR    = GRADCAM_DIR / "original"
HEAT_DIR    = GRADCAM_DIR / "heatmaps"
OVER_DIR    = GRADCAM_DIR / "overlays"

for d in [ORIG_DIR, HEAT_DIR, OVER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_demonstration():
    print("=" * 75)
    print("PHASE 6 -- GENERATING GRAD-CAM DEMONSTRATION & VISUAL REPORTS")
    print("=" * 75)

    explainer = GradCAMExplainer()
    test_preds_csv = EVAL_DIR / "test_predictions.csv"
    preds_df = pd.read_csv(test_preds_csv)

    # 1. Benchmark latencies on 30 runs
    print("Benchmarking Grad-CAM Latency Breakdown (30 runs) ...")
    bench_sample = str(REPO_ROOT / preds_df.iloc[0]["image_path"])
    pred_times = []
    gradcam_times = []
    total_times = []

    # Warmup
    _ = explainer.explain(bench_sample)

    for _ in range(30):
        t0 = time.perf_counter()
        tensor, orig_rgb = explainer.explain.__wrapped__ if hasattr(explainer.explain, "__wrapped__") else (None, None)
        # Detailed timings:
        t_start = time.perf_counter()
        res = explainer.explain(bench_sample)
        t_end = time.perf_counter()
        total_times.append((t_end - t_start) * 1000.0)

    avg_total_ms = float(np.mean(total_times))
    print(f"  Average total explanation time : {avg_total_ms:.2f} ms per image on CPU")

    # 2. Select 8 representative images across crops and classifications:
    # 5 correct (Apple, Corn, Potato, Tomato healthy/disease)
    # 3 misclassified (interesting border blights)
    correct_candidates = [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Corn_(maize)___Common_rust_",
        "Potato___Early_blight",
        "Tomato___healthy"
    ]
    misclassified_candidates = [
        ("Tomato___Late_blight", "Tomato___Early_blight"),
        ("Tomato___Early_blight", "Tomato___Bacterial_spot"),
        ("Potato___Late_blight", "Tomato___Late_blight")
    ]

    selected_rows = []
    for cls in correct_candidates:
        match = preds_df[(preds_df["actual_class"] == cls) & (preds_df["correct"] == True)]
        if len(match) > 0:
            selected_rows.append(match.iloc[0])

    for act, pred in misclassified_candidates:
        match = preds_df[(preds_df["actual_class"] == act) & (preds_df["predicted_class"] == pred)]
        if len(match) > 0:
            selected_rows.append(match.iloc[0])

    print(f"\nProcessing {len(selected_rows)} selected representative test images ...")

    records = []
    plot_items = []

    for idx, row in enumerate(selected_rows, start=1):
        img_rel = row["image_path"]
        img_abs = str(REPO_ROOT / img_rel)
        actual_cls = row["actual_class"]
        is_corr = bool(row["correct"])

        exp = explainer.explain(img_abs)
        pred_cls = exp["predicted_class"]
        conf_pct = exp["confidence_pct"]

        # Filename stems
        file_stem = f"sample_{idx:02d}_{actual_cls.split('___')[-1]}"
        orig_path = ORIG_DIR / f"{file_stem}_orig.png"
        heat_path = HEAT_DIR / f"{file_stem}_heatmap.png"
        over_path = OVER_DIR / f"{file_stem}_overlay.png"

        save_image_rgb(exp["original_rgb"], orig_path)
        save_image_rgb(exp["heatmap_colored"], heat_path)
        save_image_rgb(exp["overlay"], over_path)

        record = {
            "sample_id":            idx,
            "image_path":           img_rel,
            "actual_class":         actual_cls,
            "predicted_class":      pred_cls,
            "confidence_pct":       conf_pct,
            "correct":              is_corr,
            "target_layer":         exp["target_layer_name"],
            "original_file":        str(orig_path.relative_to(REPO_ROOT)),
            "heatmap_file":         str(heat_path.relative_to(REPO_ROOT)),
            "overlay_file":         str(over_path.relative_to(REPO_ROOT)),
        }
        records.append(record)

        status_str = "CORRECT" if is_corr else "MISCLASSIFIED"
        print(f"  [{status_str:<13}] Actual: {actual_cls:<35} Pred: {pred_cls:<35} Conf: {conf_pct}%")

        plot_items.append({
            "orig": exp["original_rgb"],
            "heat": exp["heatmap_colored"],
            "over": exp["overlay"],
            "actual": actual_cls,
            "pred": pred_cls,
            "conf": conf_pct,
            "correct": is_corr
        })

    # Save summary JSON
    summary_data = {
        "model_name":           "MobileNetV2",
        "target_layer":         explainer.target_layer_name,
        "sample_count":         len(records),
        "mean_latency_ms":      round(avg_total_ms, 2),
        "device":               "Host CPU",
        "demonstration_samples":records
    }
    with open(GRADCAM_DIR / "gradcam_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 3. Generate Composite Multi-Panel Report Image
    # 8 rows x 3 columns (Original | Heatmap | Overlay)
    n_samples = len(plot_items)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3.5 * n_samples))

    for i, item in enumerate(plot_items):
        # Column 1: Original
        axes[i, 0].imshow(item["orig"])
        axes[i, 0].set_title(f"Original Leaf Image\nActual: {item['actual'].split('___')[-1]}", fontsize=9, pad=4)
        axes[i, 0].axis("off")

        # Column 2: Heatmap
        axes[i, 1].imshow(item["heat"])
        axes[i, 1].set_title(f"Grad-CAM Heatmap ({explainer.target_layer_name})", fontsize=9, pad=4)
        axes[i, 1].axis("off")

        # Column 3: Overlay
        axes[i, 2].imshow(item["over"])
        status_txt = "CORRECT" if item["correct"] else "MISCLASSIFIED"
        color = "#16a34a" if item["correct"] else "#dc2626"
        axes[i, 2].set_title(
            f"[{status_txt}] Pred: {item['pred'].split('___')[-1]}\nConf: {item['conf']}%",
            fontsize=9, color=color, weight="bold", pad=4
        )
        axes[i, 2].axis("off")

    plt.suptitle("Grad-CAM Explainable AI Demonstration (MobileNetV2)", fontsize=14, y=0.995, weight="bold")
    plt.tight_layout()

    report_path = GRADCAM_DIR / "gradcam_report.png"
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] Composite Grad-CAM visual report -> {report_path}")
    print(f"  [SAVED] Grad-CAM summary JSON           -> {GRADCAM_DIR / 'gradcam_summary.json'}")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
