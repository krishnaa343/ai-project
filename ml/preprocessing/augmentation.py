"""
Image Preprocessing & On-The-Fly Augmentation Pipeline
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Provides deterministic image standardization (224x224 RGB) for inference,
validation, and test splits, and bounded on-the-fly augmentation for training data ONLY.
Zero augmented copies are saved to disk.
"""

import random
import numpy as np
from PIL import Image, ImageEnhance
from config import TARGET_IMAGE_SIZE, AUGMENTATION_CONFIG, RANDOM_SEED

def load_and_preprocess_image(image_path, target_size=TARGET_IMAGE_SIZE):
    """
    Deterministic image standardization pipeline.
    Used for Validation, Testing, and Live Inference.
    
    Steps:
    1. Open image from disk.
    2. Convert to 3-channel RGB (safely dropping alpha or expanding grayscale).
    3. Resize to 224x224 using Bilinear interpolation.
    4. Return normalized float32 array in [0.0, 1.0].
    
    Args:
        image_path (str or Path): Path to image file.
        target_size (tuple): (width, height), default (224, 224).
        
    Returns:
        np.ndarray: Standardized array with shape (224, 224, 3) and dtype float32.
    """
    with Image.open(image_path) as img:
        # Guarantee RGB 3-channel format
        rgb_img = img.convert("RGB")
        
        # High quality bilinear resampling
        resized_img = rgb_img.resize(target_size, resample=Image.Resampling.BILINEAR)
        
        # Convert to float32 array normalized to [0, 1]
        img_array = np.asarray(resized_img, dtype=np.float32) / 255.0
        
    return img_array

def apply_training_augmentation(pil_image, config=AUGMENTATION_CONFIG, seed=None):
    """
    Applies biologically sound, bounded random augmentations.
    STRICTLY FOR TRAINING DATA ONLY.
    
    Transformations:
    - Random rotation (-15 to +15 deg)
    - Random horizontal flip (50% probability)
    - Subtle zoom (+/- 10%)
    - Subtle brightness (+/- 10%)
    - Subtle contrast (+/- 10%)
    
    Args:
        pil_image (PIL.Image): Input RGB PIL Image.
        config (dict): Augmentation parameters.
        seed (int, optional): Random seed for reproducible verification.
        
    Returns:
        PIL.Image: Augmented RGB image.
    """
    rng = random.Random(seed) if seed is not None else random

    # 1. Random Rotation (-15 to +15 deg)
    rot_range = config.get("rotation_range_degrees", 15)
    angle = rng.uniform(-rot_range, rot_range)
    # Rotate with reflection/border fill to prevent black border artifacts
    augmented = pil_image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

    # 2. Random Horizontal Flip
    if config.get("horizontal_flip", True) and rng.random() > 0.5:
        augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # 3. Random Zoom (Crop and resize)
    zoom_min, zoom_max = config.get("zoom_range", (0.9, 1.1))
    zoom_factor = rng.uniform(zoom_min, zoom_max)
    w, h = augmented.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    if zoom_factor > 1.0:
        # Crop center
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        zoomed = augmented.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        augmented = zoomed.crop((left, top, left + w, top + h))
    elif zoom_factor < 1.0:
        # Pad center
        pad_img = Image.new("RGB", (w, h), (0, 0, 0))
        zoomed = augmented.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        pad_img.paste(zoomed, (left, top))
        augmented = pad_img

    # 4. Subtle Brightness Adjustment (+/- 10%)
    b_delta = config.get("brightness_delta", 0.10)
    b_factor = rng.uniform(1.0 - b_delta, 1.0 + b_delta)
    augmented = ImageEnhance.Brightness(augmented).enhance(b_factor)

    # 5. Subtle Contrast Adjustment (+/- 10%)
    c_min, c_max = config.get("contrast_delta", (0.9, 1.1))
    c_factor = rng.uniform(c_min, c_max)
    augmented = ImageEnhance.Contrast(augmented).enhance(c_factor)

    return augmented

def load_and_augment_training_image(image_path, target_size=TARGET_IMAGE_SIZE):
    """
    Loads, augments on-the-fly, and standardizes a single training image.
    STRICTLY FOR TRAINING DATA ONLY.
    """
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        augmented_img = apply_training_augmentation(rgb_img)
        resized_img = augmented_img.resize(target_size, resample=Image.Resampling.BILINEAR)
        img_array = np.asarray(resized_img, dtype=np.float32) / 255.0
    return img_array
