"""
model.py
--------
Builds, compiles, and returns the classification model.

Architecture: MobileNetV2 (transfer learning) + custom classification head.

Why MobileNetV2?
  ✔ Pre-trained on ImageNet → already knows low-level visual features
  ✔ Very lightweight (<14 MB) → trains fast on CPU or free Colab GPU
  ✔ Proven in medical imaging research papers
  ✔ Easily swappable with ResNet50, EfficientNet, etc.

Transfer Learning Strategy
  Phase 1 (feature extraction):
    Freeze all MobileNetV2 layers → train only the new classification head.
    Fast convergence, prevents over-fitting on small medical datasets.

  Phase 2 (optional fine-tuning):
    Unfreeze the top N layers of MobileNetV2 → jointly train with lower LR.
    Squeezes out extra accuracy once the head has converged.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, regularizers
from tensorflow.keras.applications import MobileNetV2

from src.config import (
    INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, SAVED_MODEL_PATH
)


# ─── Model factory ───────────────────────────────────────────────────────────

def build_model(fine_tune_at: int = 0) -> Model:
    """
    Build and compile the MobileNetV2-based classifier.

    Parameters
    ----------
    fine_tune_at : int
        If > 0, unfreeze layers from this index onward (fine-tuning phase).
        If 0, only the classification head is trainable.

    Returns
    -------
    tf.keras.Model – compiled, ready for model.fit()
    """
    # ── 1. Base model (pre-trained on ImageNet, no top classifier) ──────────
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,          # remove ImageNet's 1000-class head
        weights="imagenet",
    )

    # ── 2. Freeze / unfreeze strategy ────────────────────────────────────────
    base_model.trainable = fine_tune_at > 0
    if fine_tune_at > 0:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False   # freeze early layers; unfreeze top layers
        print(f"[INFO] Fine-tuning from layer {fine_tune_at} / "
              f"{len(base_model.layers)}")
    else:
        base_model.trainable = False
        print("[INFO] Feature-extraction mode: base model frozen.")

    # ── 3. Custom classification head ────────────────────────────────────────
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")

    # Pass through frozen base
    x = base_model(inputs, training=False)

    # Global average pooling converts (7,7,1280) → (1280,)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dropout for regularisation (reduces over-fitting on small medical datasets)
    x = layers.Dropout(0.3, name="dropout_1")(x)

    # Dense bottleneck
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_128",
    )(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    # Output layer
    if NUM_CLASSES == 2:
        # Binary classification → single sigmoid neuron
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss    = "binary_crossentropy"
        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        # Multi-class → softmax
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
        loss    = "categorical_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    model = Model(inputs, outputs, name="MedicalImageClassifier")

    # ── 4. Compile ────────────────────────────────────────────────────────────
    lr = LEARNING_RATE if fine_tune_at == 0 else LEARNING_RATE / 10
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )

    return model


def load_model(path: str = SAVED_MODEL_PATH) -> Model:
    """Load a previously saved model from disk."""
    print(f"[INFO] Loading model from {path}")
    return tf.keras.models.load_model(path)


def get_model_summary(model: Model) -> None:
    """Print model layer summary."""
    model.summary()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    print(f"\nTrainable params     : {trainable:,}")
    print(f"Non-trainable params : {non_trainable:,}")
