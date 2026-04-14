"""
evaluate.py
-----------
Evaluates a trained model on the test set and generates:

  1. Classification report (precision, recall, F1, support)
  2. Confusion matrix heatmap → outputs/graphs/confusion_matrix.png
  3. ROC-AUC curve           → outputs/graphs/roc_curve.png
  4. Training history plots   → outputs/graphs/training_history.png
  5. JSON metrics summary     → outputs/reports/evaluation_metrics.json

Usage
-----
  python main.py --mode evaluate
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from src.config import (
    CLASS_NAMES, NUM_CLASSES, GRAPHS_DIR, REPORTS_DIR, SAVED_MODEL_PATH
)
from src.data_loader import get_generators
from src.model import load_model


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {path}")
    return path


# ─── Plot functions ───────────────────────────────────────────────────────────

def plot_training_history(histories: dict) -> str:
    """
    Plot accuracy and loss curves for Phase 1 (and Phase 2 if present).
    """
    phases = list(histories.keys())      # ["phase1"] or ["phase1","phase2"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    colours = {"phase1": ("#1f77b4", "#ff7f0e"),
               "phase2": ("#2ca02c", "#d62728")}

    for phase in phases:
        h = histories[phase]
        c_train, c_val = colours.get(phase, ("#555", "#aaa"))
        label_suffix = f" ({phase})"

        # Accuracy
        axes[0].plot(h["accuracy"],     color=c_train,
                     label=f"Train{label_suffix}")
        axes[0].plot(h["val_accuracy"], color=c_val, linestyle="--",
                     label=f"Val{label_suffix}")

        # Loss
        axes[1].plot(h["loss"],     color=c_train,
                     label=f"Train{label_suffix}")
        axes[1].plot(h["val_loss"], color=c_val, linestyle="--",
                     label=f"Val{label_suffix}")

    for ax, title, ylabel in zip(
        axes,
        ["Model Accuracy", "Model Loss"],
        ["Accuracy", "Loss"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    return _save_fig(fig, "training_history.png")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Compute and plot normalised + raw confusion matrix.
    """
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix", fontsize=16, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Raw counts", "Normalised (%)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    return _save_fig(fig, "confusion_matrix.png")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """
    Plot ROC curve and compute AUC.
    Only valid for binary classification.
    """
    if NUM_CLASSES != 2:
        print("[WARN] ROC curve skipped (only supported for binary classification).")
        return ""

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
            label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    return _save_fig(fig, "roc_curve.png")


# ─── Main evaluation function ─────────────────────────────────────────────────

def evaluate(model_path: str = SAVED_MODEL_PATH,
             dataset_dir: str = None,
             histories: dict = None) -> dict:
    """
    Full evaluation on test set.

    Parameters
    ----------
    model_path  : str  – path to saved .keras model
    dataset_dir : str  – override default dataset location
    histories   : dict – training histories (loaded from JSON if None)

    Returns
    -------
    dict of evaluation metrics
    """

    # ── Load model & test data ─────────────────────────────────────────────────
    model = load_model(model_path)

    kwargs = {"dataset_dir": dataset_dir} if dataset_dir else {}
    _, _, test_gen = get_generators(**kwargs)

    # ── Predict ────────────────────────────────────────────────────────────────
    print("[INFO] Running inference on test set …")
    y_prob_raw = model.predict(test_gen, verbose=1)

    y_true = test_gen.classes

    if NUM_CLASSES == 2:
        y_prob = y_prob_raw.flatten()          # shape (N,)
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = y_prob_raw
        y_pred = np.argmax(y_prob, axis=1)

    # ── Classification report ──────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred)

    if NUM_CLASSES == 2:
        plot_roc_curve(y_true, y_prob)

    # ── Training history plot ──────────────────────────────────────────────────
    if histories is None:
        hist_path = os.path.join(REPORTS_DIR, "histories.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories = json.load(f)

    if histories:
        plot_training_history(histories)

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "test_accuracy"  : float(report["accuracy"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro"   : float(report["macro avg"]["recall"]),
        "f1_macro"       : float(report["macro avg"]["f1-score"]),
        "class_report"   : report,
    }

    metrics_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved → {metrics_path}")

    return metrics


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()
