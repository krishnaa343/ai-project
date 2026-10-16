"""
Preprocessing & Dataset Splitting Configuration
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture
"""

from pathlib import Path

# Base Paths (Portable resolution relative to repository root)
ML_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ML_DIR.parent
BASE_DIR = REPO_ROOT
DATA_DIR = ML_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "plantvillage"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
SELECTED_CLASSES_PATH = DATA_DIR / "selected_classes.json"
CLASS_INDICES_PATH = DATA_DIR / "class_indices.json"

# Deterministic Seed
RANDOM_SEED = 42

# Target Split Proportions (Sum to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Image Standardization Parameters (for Transfer Learning: MobileNetV2 / EfficientNet)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
TARGET_IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

# Data Augmentation Hyperparameters (TRAINING DATA ONLY)
AUGMENTATION_CONFIG = {
    "rotation_range_degrees": 15,          # Subtle rotations without inverting leaf orientation
    "horizontal_flip": True,               # Biologically invariant for foliar lesions
    "vertical_flip": False,                # Keep vertical orientation natural for leaves
    "zoom_range": (0.9, 1.1),              # Up to +/- 10% zoom
    "translation_shift_fraction": 0.08,    # Max 8% height/width shift
    "brightness_delta": 0.10,              # Subtle illumination changes (+/- 10%)
    "contrast_delta": (0.9, 1.1)           # Subtle contrast adjustment
}
