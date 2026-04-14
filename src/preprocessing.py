"""
preprocessing.py
----------------
Standalone image preprocessing utilities.

These functions work on individual images (NumPy arrays or file paths),
independent of the Keras ImageDataGenerator pipeline.

Used by:
  - predict.py  (pre-processing a single user image before inference)
  - visualize.py (showing raw vs pre-processed side-by-side)
"""

import cv2
import numpy as np
from PIL import Image

from src.config import IMG_HEIGHT, IMG_WIDTH


# ─── Core preprocessing steps ────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk as an RGB NumPy array (H, W, 3).

    Parameters
    ----------
    image_path : str  – absolute or relative path to the image file

    Returns
    -------
    np.ndarray  uint8, shape (H, W, 3)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR → convert to RGB
    return img


def resize_image(img: np.ndarray,
                 height: int = IMG_HEIGHT,
                 width: int = IMG_WIDTH) -> np.ndarray:
    """
    Resize to target (height, width) using bilinear interpolation.

    For medical images we avoid aggressive interpolation that could hide
    diagnostic features.
    """
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] float32.

    Dividing by 255 matches the ImageDataGenerator rescale=1./255 used
    during training, ensuring train/inference parity.
    """
    return img.astype(np.float32) / 255.0


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    luminance channel of the image.

    CLAHE is commonly used in radiology pre-processing to improve contrast
    in X-ray / CT images without over-amplifying noise.

    Parameters
    ----------
    img : np.ndarray  uint8 RGB

    Returns
    -------
    np.ndarray  uint8 RGB  – CLAHE-enhanced image
    """
    # Convert to LAB colour space; CLAHE is applied only to the L channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def preprocess_for_model(image_path: str,
                         apply_enhancement: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image file.

    Steps:
      1. Load (RGB uint8)
      2. CLAHE contrast enhancement (optional but recommended for X-rays)
      3. Resize to model input size
      4. Normalize to [0, 1]
      5. Add batch dimension → shape (1, H, W, 3)

    Parameters
    ----------
    image_path       : str  – path to image
    apply_enhancement: bool – whether to apply CLAHE

    Returns
    -------
    np.ndarray  float32, shape (1, IMG_HEIGHT, IMG_WIDTH, 3)
    """
    img = load_image(image_path)

    if apply_enhancement:
        img = apply_clahe(img)

    img = resize_image(img)
    img = normalize_image(img)

    # Model expects a batch: (1, H, W, C)
    return np.expand_dims(img, axis=0)


def get_preprocessing_stages(image_path: str) -> dict:
    """
    Return intermediate preprocessing stages for visualization.

    Returns a dict with keys:
      'original'  – raw loaded image (uint8 RGB)
      'enhanced'  – after CLAHE
      'resized'   – after resize
      'normalized'– after /255
    """
    original = load_image(image_path)
    enhanced = apply_clahe(original)
    resized  = resize_image(enhanced)
    normalized = normalize_image(resized)

    return {
        "original"  : original,
        "enhanced"  : enhanced,
        "resized"   : resized,
        "normalized": normalized,
    }
