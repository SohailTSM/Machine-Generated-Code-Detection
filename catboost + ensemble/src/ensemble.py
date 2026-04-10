"""
Stacking ensemble: combines CatBoost and Transformer probability outputs
through a meta-learner (logistic regression or small MLP).

Workflow:
  1. Both base models produce P(class | x) on a shared dataset.
  2. The probability vectors are concatenated into a single feature vector.
  3. A lightweight meta-learner is trained on these features.
  4. At test time the same pipeline is run: base probs → concat → meta-learner.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.config import (
    ENSEMBLE_DIR,
    ENSEMBLE_META_LEARNER,
    SUBTASK_NUM_LABELS,
    SUBTASK_LABEL_NAMES,
    SEED,
    NUM_WORKERS,
)
from src.utils import (
    get_logger,
    print_confusion_matrix,
    save_json,
)
from src.evaluation import evaluate_predictions

logger = get_logger(__name__, log_file="ensemble.log")


# ---------------------------------------------------------------------------
# Build stacking features
# ---------------------------------------------------------------------------
def build_meta_features(
    catboost_probs: np.ndarray,
    transformer_probs: np.ndarray,
) -> np.ndarray:
    """
    Concatenate probability vectors from both base models.

    Parameters
    ----------
    catboost_probs    : (n_samples, n_classes) from CatBoost
    transformer_probs : (n_samples, n_classes) from Transformer

    Returns
    -------
    (n_samples, 2 * n_classes)  meta-feature matrix
    """
    assert catboost_probs.shape == transformer_probs.shape, (
        f"Shape mismatch: CatBoost {catboost_probs.shape} vs "
        f"Transformer {transformer_probs.shape}"
    )
    return np.hstack([catboost_probs, transformer_probs])


# ---------------------------------------------------------------------------
# Train meta-learner
# ---------------------------------------------------------------------------
def train_meta_learner(
    meta_X: np.ndarray,
    meta_y: np.ndarray,
    task: str,
    learner_type: str | None = None,
) -> object:
    """
    Train a meta-learner on the stacking features.

    Returns the fitted meta-learner.
    """
    learner_type = learner_type or ENSEMBLE_META_LEARNER

    logger.info(
        f"Training {learner_type} meta-learner for Task {task} "
        f"on {meta_X.shape[0]:,} samples, {meta_X.shape[1]} features"
    )

    if learner_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=NUM_WORKERS,
        )
    elif learner_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown learner type: {learner_type}")

    model.fit(meta_X, meta_y)

    # Save checkpoint
    save_dir = os.path.join(ENSEMBLE_DIR, f"ensemble_{task}")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "meta_learner.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Meta-learner saved to {model_path}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_ensemble(
    model,
    meta_X: np.ndarray,
    meta_y: np.ndarray,
    task: str,
) -> dict:
    """
    Evaluate the ensemble on held-out meta-features.

    Returns dict with 'macro_f1' and 'predictions'.
    """
    label_names = SUBTASK_LABEL_NAMES[task]
    y_pred = model.predict(meta_X)

    metrics = evaluate_predictions(meta_y, y_pred, label_names)
    macro_f1 = metrics["macro_f1"]
    report = metrics["report"]

    logger.info(f"Task {task} — Ensemble Macro F1: {macro_f1:.4f}")
    logger.info(f"Task {task} — Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Task {task} — Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Task {task} — Macro Recall: {metrics['macro_recall']:.4f}")
    logger.info(f"\n{report}")
    print_confusion_matrix(meta_y, y_pred, label_names, logger=logger)

    save_dir = os.path.join(ENSEMBLE_DIR, f"ensemble_{task}")
    save_json(metrics, os.path.join(save_dir, "eval_results.json"))

    return {"macro_f1": macro_f1, "predictions": y_pred, "metrics": metrics}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_ensemble(
    model,
    catboost_probs: np.ndarray,
    transformer_probs: np.ndarray,
) -> np.ndarray:
    """Run the full ensemble prediction pipeline."""
    meta_X = build_meta_features(catboost_probs, transformer_probs)
    return model.predict(meta_X)


def predict_proba_ensemble(
    model,
    catboost_probs: np.ndarray,
    transformer_probs: np.ndarray,
) -> np.ndarray:
    """Return ensemble class probabilities."""
    meta_X = build_meta_features(catboost_probs, transformer_probs)
    return model.predict_proba(meta_X)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_meta_learner(task: str):
    """Load a previously saved meta-learner from disk."""
    path = os.path.join(ENSEMBLE_DIR, f"ensemble_{task}", "meta_learner.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved meta-learner at {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded meta-learner from {path}")
    return model
