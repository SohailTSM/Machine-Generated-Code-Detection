"""Unified evaluation helpers for Task A/B/C pipelines.

Primary metric is macro F1, but we always compute and return:
  - macro_f1
  - accuracy
  - macro_precision
  - macro_recall
  - confusion_matrix
  - classification_report
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a complete metric bundle for a prediction vector."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
