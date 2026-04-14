"""
visualize.py
------------
Visualization utilities:

  1. Preprocessing pipeline stages  → outputs/graphs/preprocessing_stages.png
  2. Grad-CAM heatmap               → outputs/predictions/gradcam_<name>.png
  3. Dataset sample grid            → outputs/graphs/dataset_samples.png

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights WHICH
regions of the X-ray the model focuses on when making a prediction –
directly simulating how a radiologist draws attention to abnormal zones.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tensorflow as tf

from src.config import (
    CLASS_NAMES, GRAPHS_DIR, PREDICTIONS_DIR, SAMPLE_DIR,
    IMG_HEIGHT, IMG_WIDTH
)
from src.preprocessing import (
    get_preprocessing_stages, load_image, resize_image, preprocess_for_model
)


# ─── 1. Preprocessing stages ─────────────────────────────────────────────────

def plot_preprocessing_stages(image_path: str) -> str:
    """
    Show 4-panel: original → CLAHE enhanced → resized → normalized.
    """
    stages = get_preprocessing_stages(image_path)
    titles = ["Original", "CLAHE Enhanced", "Resized (224×224)", "Normalised [0,1]"]
    images = [
        stages["original"],
        stages["enhanced"],
        stages["resized"],
        stages["normalized"],
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Preprocessing Pipeline", fontsize=15, fontweight="bold")

    for ax, img, title in zip(axes, images, titles):
        # normalized image is float [0,1], others are uint8
        ax.imshow(img if img.dtype != np.float32 else img,
                  cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    out_path = os.path.join(GRAPHS_DIR, "preprocessing_stages.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Preprocessing stages saved → {out_path}")
    return out_path


# ─── 2. Grad-CAM ─────────────────────────────────────────────────────────────

def _make_gradcam_heatmap(model: tf.keras.Model,
                          img_array: np.ndarray,
                          last_conv_layer_name: str = "Conv_1") -> np.ndarray:
    """
    Compute Grad-CAM heatmap for img_array.

    Parameters
    ----------
    model               : tf.keras.Model
    img_array           : np.ndarray  shape (1, H, W, 3)  float32 [0,1]
    last_conv_layer_name: str  – name of last conv layer in MobileNetV2

    Returns
    -------
    np.ndarray  float32  shape (H', W')  values in [0, 1]
    """
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary classification (sigmoid) use the output directly
        loss = predictions[:, 0]

    grads       = tape.gradient(loss, conv_outputs)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))      # global avg pool
    conv_out    = conv_outputs[0]
    heatmap     = conv_out @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam(image_path: str,
                 model: tf.keras.Model,
                 label: str = "",
                 last_conv_layer: str = "Conv_1") -> str:
    """
    Generate and save Grad-CAM overlay on the original image.

    The heatmap is upsampled to match the input image size, colour-mapped
    with 'jet', and blended (alpha=0.4) with the original X-ray.
    """
    img_array = preprocess_for_model(image_path, apply_enhancement=True)

    try:
        heatmap = _make_gradcam_heatmap(model, img_array, last_conv_layer)
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return ""

    # Up-sample heatmap to image dimensions
    heatmap_resized = cv2.resize(
        heatmap, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_map       = cm.get_cmap("jet")
    jet_heatmap   = jet_map(heatmap_uint8)[:, :, :3]   # drop alpha channel
    jet_heatmap   = np.uint8(jet_heatmap * 255)

    # Load and resize original
    original = load_image(image_path)
    original = resize_image(original)

    # Blend
    overlay = cv2.addWeighted(original, 0.6, jet_heatmap, 0.4, 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM Analysis  |  {label}", fontsize=14, fontweight="bold")

    axes[0].imshow(original, cmap="gray"); axes[0].set_title("Original X-Ray")
    axes[1].imshow(heatmap_resized, cmap="jet"); axes[1].set_title("Activation Heatmap")
    axes[2].imshow(overlay); axes[2].set_title("Grad-CAM Overlay")

    for ax in axes:
        ax.axis("off")

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(PREDICTIONS_DIR, f"gradcam_{basename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Grad-CAM saved → {out_path}")
    return out_path


# ─── 3. Dataset sample grid ───────────────────────────────────────────────────

def plot_dataset_samples(dataset_dir: str,
                         n_per_class: int = 4) -> str:
    """
    Plot a grid of sample images from the dataset (n_per_class per class).

    dataset_dir must contain sub-folders matching CLASS_NAMES.
    """
    fig, axes = plt.subplots(
        len(CLASS_NAMES), n_per_class,
        figsize=(n_per_class * 3, len(CLASS_NAMES) * 3)
    )
    fig.suptitle("Dataset Samples", fontsize=15, fontweight="bold")

    for row, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(dataset_dir, cls)
        paths   = (
            glob.glob(os.path.join(cls_dir, "*.jpeg")) +
            glob.glob(os.path.join(cls_dir, "*.jpg")) +
            glob.glob(os.path.join(cls_dir, "*.png"))
        )[:n_per_class]

        for col, path in enumerate(paths):
            img = load_image(path)
            img = resize_image(img)
            ax  = axes[row][col] if len(CLASS_NAMES) > 1 else axes[col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_title(cls, fontweight="bold", fontsize=12)

    out_path = os.path.join(GRAPHS_DIR, "dataset_samples.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Sample grid saved → {out_path}")
    return out_path
