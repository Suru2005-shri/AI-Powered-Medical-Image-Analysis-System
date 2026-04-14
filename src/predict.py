"""
predict.py
----------
Single-image and batch prediction utilities.

Usage
-----
  # Via CLI (main.py)
  python main.py --mode predict --image path/to/xray.jpg

  # Direct import
  from src.predict import predict_single, predict_batch
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CLASS_NAMES, NUM_CLASSES, CONFIDENCE_THRESHOLD,
    SAVED_MODEL_PATH, PREDICTIONS_DIR
)
from src.model import load_model
from src.preprocessing import preprocess_for_model, load_image, resize_image


# ─── Single-image prediction ──────────────────────────────────────────────────

def predict_single(image_path: str,
                   model=None,
                   model_path: str = SAVED_MODEL_PATH,
                   save_output: bool = True) -> dict:
    """
    Predict class for a single medical image.

    Parameters
    ----------
    image_path  : str           – path to input image
    model       : tf.keras.Model (pass in to avoid reloading)
    model_path  : str           – path to saved model (used if model is None)
    save_output : bool          – save annotated prediction image to disk

    Returns
    -------
    dict with keys: 'class', 'confidence', 'label', 'image_path'
    """
    if model is None:
        model = load_model(model_path)

    # Pre-process
    img_array = preprocess_for_model(image_path, apply_enhancement=True)

    # Inference
    prediction = model.predict(img_array, verbose=0)

    # Decode
    if NUM_CLASSES == 2:
        confidence = float(prediction[0][0])
        class_idx  = 1 if confidence >= CONFIDENCE_THRESHOLD else 0
    else:
        class_idx  = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_idx])

    label      = CLASS_NAMES[class_idx]
    is_disease = class_idx == 1          # index 1 = PNEUMONIA / disease class

    result = {
        "image_path" : image_path,
        "class_index": class_idx,
        "label"      : label,
        "confidence" : round(confidence * 100, 2),
        "is_disease" : is_disease,
        "raw_scores" : prediction[0].tolist(),
    }

    print(f"\n{'─'*50}")
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print(f"  Risk flag  : {'⚠ DISEASE DETECTED' if is_disease else '✓ NORMAL'}")
    print(f"{'─'*50}")

    if save_output:
        _save_prediction_image(image_path, result)

    return result


def _save_prediction_image(image_path: str, result: dict) -> str:
    """Save annotated image with prediction overlay."""
    img = load_image(image_path)
    img = resize_image(img)

    colour = "#e74c3c" if result["is_disease"] else "#2ecc71"
    status = "⚠ DISEASE DETECTED" if result["is_disease"] else "✓ NORMAL"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(
        f"{result['label']}   {result['confidence']:.1f}% confidence\n{status}",
        fontsize=13,
        fontweight="bold",
        color=colour,
        pad=10,
    )

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(PREDICTIONS_DIR, f"pred_{basename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#111111")
    plt.close(fig)
    print(f"[INFO] Prediction image saved → {out_path}")
    return out_path


# ─── Batch prediction ─────────────────────────────────────────────────────────

def predict_batch(image_dir: str,
                  model=None,
                  model_path: str = SAVED_MODEL_PATH,
                  extensions: tuple = ("*.jpg", "*.jpeg", "*.png")) -> list:
    """
    Predict all images in a directory.

    Parameters
    ----------
    image_dir  : str  – folder containing images
    model      : pre-loaded model (optional)
    model_path : str  – fallback model path
    extensions : tuple – glob patterns

    Returns
    -------
    list of result dicts (one per image)
    """
    if model is None:
        model = load_model(model_path)

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_paths:
        print(f"[WARN] No images found in {image_dir}")
        return []

    print(f"[INFO] Running batch prediction on {len(image_paths)} images …")
    results = []
    for path in sorted(image_paths):
        r = predict_single(path, model=model, save_output=True)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    normal_count  = sum(1 for r in results if not r["is_disease"])
    disease_count = sum(1 for r in results if r["is_disease"])
    print(f"\n{'='*50}")
    print(f"  Batch complete: {len(results)} images")
    print(f"  Normal   : {normal_count}")
    print(f"  Disease  : {disease_count}")
    print(f"{'='*50}\n")

    _save_batch_summary_grid(results[:16])   # show first 16 in a grid

    return results


def _save_batch_summary_grid(results: list) -> str:
    """Save a grid of predicted images (up to 16)."""
    n    = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Batch Predictions", fontsize=16, fontweight="bold",
                 color="white")
    fig.patch.set_facecolor("#111111")
    axes = axes.flatten() if n > 1 else [axes]

    for i, result in enumerate(results):
        img = load_image(result["image_path"])
        img = resize_image(img)
        colour = "#e74c3c" if result["is_disease"] else "#2ecc71"

        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(
            f"{result['label']}\n{result['confidence']:.1f}%",
            color=colour, fontsize=9,
        )

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    out_path = os.path.join(PREDICTIONS_DIR, "batch_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#111111")
    plt.close(fig)
    print(f"[INFO] Batch grid saved → {out_path}")
    return out_path
