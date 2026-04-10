"""
CatBoost training, evaluation, checkpoint saving / resuming, and prediction.

Key design decisions:
  - Uses auto_class_weights='Balanced' to handle class imbalance.
  - Saves model snapshot after training so it can be loaded later for
    ensemble or standalone prediction.
  - Feature importances are logged and saved.
"""

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    CATBOOST_PARAMS,
    CATBOOST_DIR,
    SUBTASK_LABEL_NAMES,
    SUBTASK_NUM_LABELS,
    NUM_WORKERS,
    SEED,
)
from src.utils import (
    get_logger,
    print_confusion_matrix,
    save_json,
)
from src.evaluation import evaluate_predictions

logger = get_logger(__name__, log_file="catboost.log")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    task: str,
    params: dict | None = None,
    resume_from: str | None = None,
) -> CatBoostClassifier:
    """
    Train (or resume training) a CatBoost model.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val, y_val     : validation features and labels
    task             : 'A', 'B', or 'C'
    params           : override default CATBOOST_PARAMS
    resume_from      : path to a saved CatBoost model to continue training

    Returns
    -------
    Trained CatBoostClassifier instance.
    """
    p = {**CATBOOST_PARAMS, **(params or {})}

    # Keep MultiClass + TotalF1:Macro for all tasks (including binary)
    # so that early stopping always optimises the competition metric.

    model_path = os.path.join(CATBOOST_DIR, f"catboost_{task}")
    os.makedirs(model_path, exist_ok=True)

    train_pool = Pool(X_train, label=y_train)
    val_pool   = Pool(X_val, label=y_val)

    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming training from {resume_from}")
        model = CatBoostClassifier()
        model.load_model(resume_from)
        # Continue training with more iterations
        model.fit(
            train_pool,
            eval_set=val_pool,
            init_model=resume_from,
            verbose=p.get("verbose", 200),
            early_stopping_rounds=p.get("early_stopping_rounds", 100),
        )
    else:
        logger.info(f"Training CatBoost for task {task} from scratch ...")
        logger.info(f"Params: {p}")
        model = CatBoostClassifier(**p, train_dir=model_path)
        model.fit(train_pool, eval_set=val_pool)

    # Save model checkpoint
    snapshot_path = os.path.join(model_path, "model.cbm")
    model.save_model(snapshot_path)
    logger.info(f"Model saved to {snapshot_path}")

    # Save feature importances
    if hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
        feature_names = X_train.columns.tolist()
        imp_dict = dict(zip(feature_names, importances.tolist()))
        imp_sorted = dict(sorted(imp_dict.items(), key=lambda x: -x[1]))
        save_json(imp_sorted, os.path.join(model_path, "feature_importances.json"))
        logger.info("Top-10 features:")
        for name, score in list(imp_sorted.items())[:10]:
            logger.info(f"  {name:>30s}: {score:.4f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_catboost(
    model: CatBoostClassifier,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    task: str,
) -> dict:
    """
    Evaluate the model on validation data and log results.

    Returns a dict with 'macro_f1' and 'predictions'.
    """
    label_names = SUBTASK_LABEL_NAMES[task]
    y_pred = model.predict(X_val).astype(int).flatten()

    metrics = evaluate_predictions(y_val, y_pred, label_names)
    macro_f1 = metrics["macro_f1"]
    report = metrics["report"]

    logger.info(f"Task {task} — CatBoost Macro F1: {macro_f1:.4f}")
    logger.info(f"Task {task} — Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Task {task} — Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Task {task} — Macro Recall: {metrics['macro_recall']:.4f}")
    logger.info(f"\n{report}")
    print_confusion_matrix(y_val, y_pred, label_names, logger=logger)

    # Save report
    model_path = os.path.join(CATBOOST_DIR, f"catboost_{task}")
    save_json(metrics, os.path.join(model_path, "eval_results.json"))

    return {"macro_f1": macro_f1, "predictions": y_pred, "metrics": metrics}


# ---------------------------------------------------------------------------
# Probability predictions (for ensemble)
# ---------------------------------------------------------------------------
def predict_proba_catboost(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return class probabilities (n_samples × n_classes)."""
    return model.predict_proba(X)


# ---------------------------------------------------------------------------
# Loading a saved model
# ---------------------------------------------------------------------------
def load_catboost(task: str) -> CatBoostClassifier:
    """Load a previously trained CatBoost model from disk."""
    path = os.path.join(CATBOOST_DIR, f"catboost_{task}", "model.cbm")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved CatBoost model at {path}")
    model = CatBoostClassifier()
    model.load_model(path)
    logger.info(f"Loaded CatBoost model from {path}")
    return model
