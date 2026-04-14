"""
train.py
--------
Orchestrates the full training pipeline:

  Phase 1 – Feature extraction (frozen base, train head only)
  Phase 2 – Optional fine-tuning (unfreeze top layers, lower LR)

Usage
-----
  python main.py --mode train

or standalone:
  python -m src.train
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

from src.config import (
    EPOCHS, BATCH_SIZE, SAVED_MODEL_PATH, MODELS_DIR,
    GRAPHS_DIR, REPORTS_DIR, CLASS_NAMES, NUM_CLASSES
)
from src.data_loader import get_generators
from src.model import build_model, get_model_summary


# ─── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(checkpoint_path: str) -> list:
    """
    Standard set of training callbacks:
      - ModelCheckpoint : save best weights (val_accuracy)
      - EarlyStopping   : halt if val_loss stagnates for 5 epochs
      - ReduceLROnPlateau: halve LR if val_loss doesn't improve for 3 epochs
      - CSVLogger       : log metrics to CSV for later plotting
    """
    csv_log_path = os.path.join(REPORTS_DIR, "training_log.csv")

    return [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(csv_log_path, append=False),
    ]


# ─── Class weight computation (handles class imbalance) ───────────────────────

def compute_class_weights(train_gen) -> dict:
    """
    Compute class weights inversely proportional to class frequency.
    Helps with the NORMAL vs PNEUMONIA imbalance in the Kaggle dataset.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes
    unique = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique, y=labels)
    cw = dict(zip(unique.tolist(), weights.tolist()))
    print(f"[INFO] Class weights: {cw}")
    return cw


# ─── Main training function ───────────────────────────────────────────────────

def train(dataset_dir: str = None,
          run_fine_tuning: bool = True) -> dict:
    """
    Run the complete training pipeline.

    Parameters
    ----------
    dataset_dir     : str  – override default dataset location
    run_fine_tuning : bool – whether to run Phase 2 fine-tuning

    Returns
    -------
    dict with 'history_phase1' and optionally 'history_phase2'
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    kwargs = {"dataset_dir": dataset_dir} if dataset_dir else {}
    train_gen, val_gen, test_gen = get_generators(**kwargs)

    # ── Class weights ──────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_gen)

    # ── PHASE 1: Feature extraction ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 — Feature Extraction (base model frozen)")
    print("="*60)

    model = build_model(fine_tune_at=0)
    get_model_summary(model)

    checkpoint_p1 = os.path.join(MODELS_DIR, "best_phase1.keras")
    history_p1 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(checkpoint_p1),
        class_weight=class_weights,
        verbose=1,
    )

    # ── PHASE 2: Fine-tuning ──────────────────────────────────────────────────
    history_p2 = None
    if run_fine_tuning:
        print("\n" + "="*60)
        print("  PHASE 2 — Fine-tuning (top layers unfrozen)")
        print("="*60)

        # Unfreeze layers from index 100 onward (top ~20% of MobileNetV2)
        model_ft = build_model(fine_tune_at=100)
        model_ft.load_weights(checkpoint_p1)   # start from best Phase 1 weights

        checkpoint_p2 = os.path.join(MODELS_DIR, "best_phase2.keras")
        history_p2 = model_ft.fit(
            train_gen,
            epochs=10,                         # shorter fine-tuning run
            validation_data=val_gen,
            callbacks=get_callbacks(checkpoint_p2),
            class_weight=class_weights,
            verbose=1,
        )
        model = model_ft
        best_checkpoint = checkpoint_p2
    else:
        best_checkpoint = checkpoint_p1

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(SAVED_MODEL_PATH)
    print(f"\n[INFO] Final model saved → {SAVED_MODEL_PATH}")

    # ── Save histories to JSON for later plotting ──────────────────────────────
    histories = {
        "phase1": {k: [float(v) for v in vals]
                   for k, vals in history_p1.history.items()},
    }
    if history_p2:
        histories["phase2"] = {k: [float(v) for v in vals]
                               for k, vals in history_p2.history.items()}

    history_path = os.path.join(REPORTS_DIR, "histories.json")
    with open(history_path, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"[INFO] Training histories saved → {history_path}")

    return histories


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
