"""
MobileNetV2 Transfer-Learning Model Builder
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Builds the two-stage transfer-learning model:
  Stage 1: frozen MobileNetV2 base + trainable classification head
  Stage 2: top layers of base unfrozen for fine-tuning
"""

import tensorflow as tf
try:
    from config import (
        INPUT_SHAPE,
        STAGE2_UNFREEZE_FROM,
        STAGE1_LEARNING_RATE,
        STAGE2_LEARNING_RATE,
        LOSS_FUNCTION,
    )
except ImportError:
    from ml.training.config import (
        INPUT_SHAPE,
        STAGE2_UNFREEZE_FROM,
        STAGE1_LEARNING_RATE,
        STAGE2_LEARNING_RATE,
        LOSS_FUNCTION,
    )


def build_stage1_model(num_classes: int) -> tf.keras.Model:
    """
    Constructs MobileNetV2 with frozen base for Stage 1 (head-only) training.

    Architecture:
      Input (224, 224, 3)
        → MobileNetV2 backbone (ImageNet, frozen)
        → GlobalAveragePooling2D
        → Dropout(0.3)
        → Dense(num_classes, activation='softmax')

    Args:
        num_classes: Exact number of output classes from class_indices.json

    Returns:
        Compiled tf.keras.Model
    """
    # Load pretrained MobileNetV2 (no top classification head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    # Freeze all pretrained layers for Stage 1
    base_model.trainable = False

    # Build classification head
    inputs  = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
    x       = base_model(inputs, training=False)      # training=False keeps BN in inference mode
    x       = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x       = tf.keras.layers.Dropout(0.3, name="dropout_head")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_CropDisease_Stage1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=LOSS_FUNCTION,
        metrics=["accuracy"]
    )
    return model


def prepare_stage2_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Modifies an existing Stage 1 model for Stage 2 fine-tuning.

    Unfreezes MobileNetV2 layers from index STAGE2_UNFREEZE_FROM onwards
    (top ~30% of base layers) while keeping earlier layers frozen to preserve
    low-level ImageNet features and prevent catastrophic forgetting.

    Args:
        model: The trained Stage 1 model

    Returns:
        Re-compiled tf.keras.Model with partial unfreezing applied
    """
    # Access MobileNetV2 base (second layer after Input)
    base_model = model.layers[1]
    base_model.trainable = True

    # Selectively freeze layers up to STAGE2_UNFREEZE_FROM
    for layer in base_model.layers[:STAGE2_UNFREEZE_FROM]:
        layer.trainable = False

    trainable_layers = sum(1 for l in base_model.layers if l.trainable)
    frozen_layers    = sum(1 for l in base_model.layers if not l.trainable)
    print(f"  Stage 2: {trainable_layers} base layers trainable, {frozen_layers} frozen")

    # Recompile with much lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
        loss=LOSS_FUNCTION,
        metrics=["accuracy"]
    )
    return model


def print_model_summary(model: tf.keras.Model):
    total     = sum(tf.size(v).numpy() for v in model.trainable_variables)
    non_train = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
    print(f"  Total params      : {(total + non_train):,}")
    print(f"  Trainable params  : {total:,}")
    print(f"  Non-trainable     : {non_train:,}")
