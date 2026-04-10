#!/usr/bin/env python3
"""
Entry point: CatBoost pipeline.

Usage examples:
    # Train on a small subsample for Subtask A (default)
    python run_catboost.py --task A

    # Train on full data
    python run_catboost.py --task A --full

    # Resume training from a saved model
    python run_catboost.py --task B --resume

    # Train all three subtasks sequentially
    python run_catboost.py --task A B C
"""

import argparse
import os
import sys

# Ensure the project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CATBOOST_DIR
from src.utils import set_seed, get_logger
from src.data_loader import load_and_subsample
from src.feature_extractor import extract_and_cache
from src.catboost_model import (
    train_catboost,
    evaluate_catboost,
    load_catboost,
)

logger = get_logger("run_catboost", log_file="run_catboost.log")


def run_single_task(task: str, use_full_data: bool, resume: bool):
    """Train + evaluate CatBoost for one subtask."""
    logger.info(f"{'='*60}")
    logger.info(f" CatBoost Pipeline — Task {task}")
    logger.info(f"{'='*60}")

    # ---- 1. Load data (subsample by default) ----
    train_df, val_df = load_and_subsample(task, subsample=not use_full_data)

    # ---- 2. Extract features (cached to disk) ----
    split_suffix = "full" if use_full_data else "sub"
    X_train = extract_and_cache(train_df, task, f"train_{split_suffix}")
    X_val   = extract_and_cache(val_df,   task, "val")

    y_train = train_df["label"].values
    y_val   = val_df["label"].values

    # ---- 3. Train (or resume) ----
    resume_path = None
    if resume:
        resume_path = os.path.join(CATBOOST_DIR, f"catboost_{task}", "model.cbm")
        if not os.path.exists(resume_path):
            logger.warning(f"No checkpoint found at {resume_path} — training from scratch.")
            resume_path = None

    model = train_catboost(
        X_train, y_train,
        X_val, y_val,
        task=task,
        resume_from=resume_path,
    )

    # ---- 4. Evaluate ----
    results = evaluate_catboost(model, X_val, y_val, task)
    logger.info(f"Task {task} done — Macro F1: {results['macro_f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CatBoost training pipeline")
    parser.add_argument(
        "--task", nargs="+", choices=["A", "B", "C"], default=["A"],
        help="Subtask(s) to train. Pass multiple to run sequentially.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Train on the full dataset instead of the subsample.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from a previously saved CatBoost model.",
    )
    args = parser.parse_args()

    set_seed()

    for task in args.task:
        run_single_task(task, use_full_data=args.full, resume=args.resume)

    logger.info("All tasks complete.")


if __name__ == "__main__":
    main()
