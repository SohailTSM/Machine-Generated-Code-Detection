"""
Shared utilities: metrics, logging setup, seed setting, and checkpoint helpers.
"""

import os
import json
import random
import logging
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)

from src.config import SEED, LOGS_DIR


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a configured logger that writes to console and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(os.path.join(LOGS_DIR, log_file))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED):
    """Fix random seeds for reproducibility across numpy, torch, python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_macro_f1(y_true, y_pred) -> float:
    """Compute the official competition metric: Macro F1."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def full_classification_report(y_true, y_pred, label_names=None) -> str:
    """Return a formatted classification report string."""
    return classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )


def print_confusion_matrix(y_true, y_pred, label_names=None, logger=None):
    """Log the confusion matrix in a readable way."""
    cm = confusion_matrix(y_true, y_pred)
    log = logger.info if logger else print
    log("Confusion Matrix:")
    if label_names:
        header = "        " + "  ".join(f"{n[:8]:>8}" for n in label_names)
        log(header)
    for i, row in enumerate(cm):
        name = label_names[i][:8] if label_names else str(i)
        log(f"{name:>8} " + "  ".join(f"{v:>8}" for v in row))


# ---------------------------------------------------------------------------
# Checkpoint / artifact helpers
# ---------------------------------------------------------------------------
def save_json(obj, path: str):
    """Save a dictionary / list to JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_predictions(ids, preds, path: str):
    """Write predictions CSV in the competition format: ID,label."""
    import pandas as pd
    df = pd.DataFrame({"ID": ids, "label": preds})
    df.to_csv(path, index=False)


def get_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
