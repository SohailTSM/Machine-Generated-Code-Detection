#!/usr/bin/env python3
"""
Entry point: Stacking Ensemble (CatBoost + Transformer → meta-learner).

This script assumes that both CatBoost and Transformer models have already
been trained and saved.  It:
  1. Loads validation data + the saved base models.
  2. Generates class-probability predictions from each base model.
  3. Concatenates them into meta-features.
  4. Trains a lightweight meta-learner (logistic regression or MLP).
  5. Evaluates the ensemble on the validation set.

Usage examples:
    # Build ensemble for Subtask A
    python run_ensemble.py --task A

    # Build for all subtasks
    python run_ensemble.py --task A B C

    # Use full data (must match what base models were trained on)
    python run_ensemble.py --task A --full

    # Switch to MLP meta-learner
    python run_ensemble.py --task A --meta_learner mlp
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split

from src.config import TRANSFORMER_DIR, SEED
from src.utils import set_seed, get_logger
from src.data_loader import load_and_subsample
from src.feature_extractor import extract_and_cache
from src.catboost_model import load_catboost, predict_proba_catboost
from src.transformer_model import predict_proba_transformer
from src.ensemble import (
    build_meta_features,
    train_meta_learner,
    evaluate_ensemble,
)

logger = get_logger("run_ensemble", log_file="run_ensemble.log")

META_TRAIN_RATIO = 0.7   # 70 % for meta-training, 30 % for meta-testing


def run_single_task(task: str, use_full_data: bool, meta_learner: str):
    """Build and evaluate the stacking ensemble for one subtask."""
    logger.info(f"{'='*60}")
    logger.info(f" Ensemble Pipeline — Task {task}")
    logger.info(f"{'='*60}")

    # ---- 1. Load validation data ----
    _, val_df = load_and_subsample(task, subsample=not use_full_data)
    y_val = val_df["label"].values
    logger.info(f"Validation set: {len(val_df):,} samples")

    # ---- 2. CatBoost probabilities (on full val set) ----
    logger.info("Generating CatBoost probabilities …")
    catboost_model = load_catboost(task)
    val_features = extract_and_cache(val_df, task, split="val")
    cat_probs = predict_proba_catboost(catboost_model, val_features)
    logger.info(f"CatBoost probs shape: {cat_probs.shape}")

    # ---- 3. Transformer probabilities (on full val set) ----
    logger.info("Generating Transformer probabilities …")
    best_model_dir = os.path.join(TRANSFORMER_DIR, task, "best_model")
    if not os.path.isdir(best_model_dir):
        raise FileNotFoundError(
            f"No saved transformer at {best_model_dir}. "
            f"Run run_transformer.py --task {task} first."
        )
    trans_probs = predict_proba_transformer(
        model_dir=best_model_dir,
        codes=val_df["code"].tolist(),
    )
    logger.info(f"Transformer probs shape: {trans_probs.shape}")

    # ---- 4. Build meta-features, split into meta-train / meta-test ----
    meta_X = build_meta_features(cat_probs, trans_probs)

    idx_train, idx_test = train_test_split(
        range(len(meta_X)),
        train_size=META_TRAIN_RATIO,
        stratify=y_val,
        random_state=SEED,
    )
    meta_X_train, meta_y_train = meta_X[idx_train], y_val[idx_train]
    meta_X_test,  meta_y_test  = meta_X[idx_test],  y_val[idx_test]

    logger.info(
        f"Meta split — train: {len(idx_train):,}  test: {len(idx_test):,}"
    )

    # ---- 5. Train meta-learner on meta-train ----
    meta_model = train_meta_learner(
        meta_X_train, meta_y_train, task, learner_type=meta_learner,
    )

    # ---- 6. Evaluate on held-out meta-test ----
    results = evaluate_ensemble(meta_model, meta_X_test, meta_y_test, task)
    logger.info(f"Task {task} Ensemble Macro F1: {results['macro_f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stacking ensemble pipeline")
    parser.add_argument(
        "--task", nargs="+", choices=["A", "B", "C"], default=["A"],
        help="Subtask(s) to build ensemble for.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full (non-subsampled) data.  Must match base-model training.",
    )
    parser.add_argument(
        "--meta_learner", type=str, default="logistic_regression",
        choices=["logistic_regression", "mlp"],
        help="Type of meta-learner (default: logistic_regression).",
    )
    args = parser.parse_args()

    set_seed()

    for task in args.task:
        run_single_task(task, use_full_data=args.full, meta_learner=args.meta_learner)

    logger.info("All ensemble tasks complete.")


if __name__ == "__main__":
    main()
