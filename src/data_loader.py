"""
data_loader.py
--------------
Handles:
  1. Downloading the Chest X-Ray dataset from Kaggle.
  2. Building tf.data / Keras ImageDataGenerator pipelines.
  3. Returning train / validation / test generators.

HOW TO GET THE KAGGLE DATASET
──────────────────────────────
Method A (recommended – Kaggle API):
  1. Create a free account at kaggle.com
  2. Go to Account → API → Create New Token → download kaggle.json
  3. Place kaggle.json at  ~/.kaggle/kaggle.json  (Linux/Mac)
                       or  C:\\Users\\<you>\\.kaggle\\kaggle.json  (Windows)
  4. Run:  python -c "from src.data_loader import download_dataset; download_dataset()"

Method B (manual):
  1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  2. Click Download
  3. Unzip into  data/raw/chest_xray/
  The structure should be:
    data/raw/chest_xray/
      train/NORMAL/   train/PNEUMONIA/
      val/NORMAL/     val/PNEUMONIA/
      test/NORMAL/    test/PNEUMONIA/
"""

import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    RAW_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE,
    CLASS_NAMES, KAGGLE_DATASET, USE_AUGMENTATION, RANDOM_SEED
)

DATASET_SUBDIR = os.path.join(RAW_DIR, "chest_xray")


# ─── Download ─────────────────────────────────────────────────────────────────

def download_dataset():
    """Download dataset via Kaggle API (requires kaggle.json configured)."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError("Install kaggle:  pip install kaggle")

    if os.path.exists(DATASET_SUBDIR):
        print(f"[INFO] Dataset already exists at {DATASET_SUBDIR}. Skipping download.")
        return

    print("[INFO] Downloading dataset from Kaggle …")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip")
    print(f"[INFO] Dataset downloaded to {RAW_DIR}")


# ─── Generators ───────────────────────────────────────────────────────────────

def _make_train_datagen():
    """ImageDataGenerator with optional augmentation for training."""
    if USE_AUGMENTATION:
        return ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    return ImageDataGenerator(rescale=1.0 / 255)


def _make_val_test_datagen():
    """ImageDataGenerator WITHOUT augmentation for val/test."""
    return ImageDataGenerator(rescale=1.0 / 255)


def get_generators(dataset_dir: str = DATASET_SUBDIR):
    """
    Build and return (train_gen, val_gen, test_gen).

    Parameters
    ----------
    dataset_dir : str
        Root folder that contains 'train/', 'val/', 'test/' sub-directories.

    Returns
    -------
    train_gen, val_gen, test_gen : DirectoryIterator objects
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir   = os.path.join(dataset_dir, "val")
    test_dir  = os.path.join(dataset_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"[ERROR] Directory not found: {d}\n"
                "Did you download and unzip the dataset? See data_loader.py header."
            )

    target_size   = (IMG_HEIGHT, IMG_WIDTH)
    class_mode    = "binary" if len(CLASS_NAMES) == 2 else "categorical"
    color_mode    = "rgb"

    train_gen = _make_train_datagen().flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        color_mode=color_mode,
        class_mode=class_mode,
        classes=CLASS_NAMES,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = _make_val_test_datagen().flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        color_mode=color_mode,
        class_mode=class_mode,
        classes=CLASS_NAMES,
        shuffle=False,
    )

    test_gen = _make_val_test_datagen().flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        color_mode=color_mode,
        class_mode=class_mode,
        classes=CLASS_NAMES,
        shuffle=False,
    )

    print(f"[INFO] Train samples : {train_gen.samples}")
    print(f"[INFO] Val   samples : {val_gen.samples}")
    print(f"[INFO] Test  samples : {test_gen.samples}")
    print(f"[INFO] Classes       : {train_gen.class_indices}")

    return train_gen, val_gen, test_gen
