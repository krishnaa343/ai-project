"""
Training Configuration
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Single source of truth for all training hyperparameters.
All values used in training must match what is documented in model_metadata.json.
"""

from pathlib import Path
import datetime

# ── Path Resolution ──────────────────────────────────────────────────────────
TRAINING_DIR  = Path(__file__).resolve().parent
ML_DIR        = TRAINING_DIR.parent
REPO_ROOT     = ML_DIR.parent

DATA_DIR             = ML_DIR / "data"
SPLITS_DIR           = DATA_DIR / "splits"
CLASS_INDICES_PATH   = DATA_DIR / "class_indices.json"
SELECTED_CLASSES_PATH= DATA_DIR / "selected_classes.json"

MODELS_DIR           = ML_DIR / "models"
EVAL_DIR             = ML_DIR / "evaluation"
TRAINING_HISTORY_DIR = EVAL_DIR / "training_history"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Input Specification ───────────────────────────────────────────────────────
IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
IMAGE_CHANNELS = 3
INPUT_SHAPE   = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# ── Model Selection ───────────────────────────────────────────────────────────
# MobileNetV2 chosen for Phase 4:
#   - Lightweight: ~3.5M params, fast CPU inference
#   - Well-supported: official tf.keras.applications with ImageNet weights
#   - Suitable for limited VRAM / CPU-only environments
#   - Easily deployable in Flask inference server
#   - Can be benchmarked against EfficientNet in Phase 5
MODEL_NAME = "MobileNetV2"

# ── Data Loading ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
PREFETCH    = True   # tf.data prefetch for pipeline efficiency
NUM_WORKERS = 4      # parallel map workers

# ── Stage 1 – Classifier Head Training (Base frozen) ─────────────────────────
STAGE1_EPOCHS          = 5
STAGE1_LEARNING_RATE   = 1e-3
STAGE1_EARLY_STOP_PAT  = 2    # EarlyStopping patience (val_loss)
STAGE1_LR_REDUCE_PAT   = 2    # ReduceLROnPlateau patience
STAGE1_LR_REDUCE_FACTOR= 0.5
STAGE1_MIN_LR          = 1e-6

# ── Stage 2 – Fine-Tuning (Top layers unfrozen) ───────────────────────────────
# Unfreeze the top ~30% of MobileNetV2 base layers
STAGE2_UNFREEZE_FROM   = 100  # Layer index in MobileNetV2 (of ~155 total)
STAGE2_EPOCHS          = 4
STAGE2_LEARNING_RATE   = 1e-5  # Must be << Stage 1 LR to preserve pretrained features
STAGE2_EARLY_STOP_PAT  = 2
STAGE2_LR_REDUCE_PAT   = 2
STAGE2_LR_REDUCE_FACTOR= 0.5
STAGE2_MIN_LR          = 1e-7

# ── Loss & Optimiser ──────────────────────────────────────────────────────────
# SparseCategoricalCrossentropy: labels are integer class indices (not one-hot)
LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER_NAME = "Adam"

# ── Model Checkpoints ─────────────────────────────────────────────────────────
BEST_MODEL_PATH  = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"
METADATA_PATH    = MODELS_DIR / "model_metadata.json"
