"""Visualization — training curves, confusion matrix, t-SNE."""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src import config as cfg


def plot_training_curves(history, save_dir=None):
    """Plot loss and F1 curves from training history."""
    if save_dir is None:
        save_dir = cfg.LOG_DIR

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # F1
    axes[1].plot(epochs, history["train_f1"], label="Train")
    axes[1].plot(epochs, history["val_f1"], label="Val")
    baseline = 1.0 / cfg.NUM_CLASSES
    axes[1].axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
                     label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro F1-Score")
    axes[1].legend()

    # LR
    axes[2].plot(epochs, history["lr"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")

    plt.tight_layout()
    path = os.path.join(save_dir, f"training_curves_task{cfg.ACTIVE_TASK}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves → {path}")


def plot_confusion_matrix(labels, preds, class_names=None, save_dir=None):
    """Plot and save confusion matrix heatmap."""
    if save_dir is None:
        save_dir = cfg.LOG_DIR
    if class_names is None:
        class_names = cfg.LABEL_NAMES

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Task {cfg.ACTIVE_TASK}")

    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_task{cfg.ACTIVE_TASK}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix → {path}")


def plot_per_class_f1(labels, preds, class_names=None, save_dir=None):
    """Horizontal bar chart of per-class F1 scores."""
    from sklearn.metrics import f1_score

    if save_dir is None:
        save_dir = cfg.LOG_DIR
    if class_names is None:
        class_names = cfg.LABEL_NAMES

    per_class = f1_score(labels, preds, average=None, zero_division=0)
    macro = f1_score(labels, preds, average='macro', zero_division=0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(class_names) * 0.5)))
    y_pos = range(len(class_names))
    ax.barh(y_pos, per_class, color='steelblue')
    ax.axvline(x=macro, color='red', linestyle='--', label=f"Macro F1 = {macro:.3f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("F1-Score")
    ax.set_title(f"Per-Class F1 — Task {cfg.ACTIVE_TASK}")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"per_class_f1_task{cfg.ACTIVE_TASK}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class F1 → {path}")
