"""
config.py
---------
Central configuration for the AI Medical Image Analysis project.
Edit these values to adapt the project to a different dataset or task.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
SAMPLE_DIR      = os.path.join(DATA_DIR, "sample_images")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR      = os.path.join(OUTPUTS_DIR, "graphs")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
REPORTS_DIR     = os.path.join(OUTPUTS_DIR, "reports")

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Chest X-Ray Images (Pneumonia) from Kaggle
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
KAGGLE_DATASET  = "paultimothymooney/chest-xray-pneumonia"
CLASS_NAMES     = ["NORMAL", "PNEUMONIA"]   # must match subfolder names
NUM_CLASSES     = len(CLASS_NAMES)

# ─── Image ────────────────────────────────────────────────────────────────────
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
IMG_CHANNELS    = 3
INPUT_SHAPE     = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 20
LEARNING_RATE   = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED     = 42

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "mobilenet_chest_xray"
SAVED_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras")

# ─── Augmentation ─────────────────────────────────────────────────────────────
USE_AUGMENTATION = True

# ─── Thresholds ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5   # binary classification decision boundary

# ─── Ensure directories exist ─────────────────────────────────────────────────
for _dir in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR,
             GRAPHS_DIR, PREDICTIONS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
