#!/usr/bin/env python3
"""
Entry point: Transformer (ModernBERT) fine-tuning pipeline.

Usage examples:
    # Train on a small subsample for Subtask A (default)
    python run_transformer.py --task A

    # Train on full data
    python run_transformer.py --task A --full

    # Resume from the latest checkpoint
    python run_transformer.py --task A --resume

    # Use a different model
    python run_transformer.py --task A --model microsoft/unixcoder-base

    # Adjust batch size and max length for your GPU
    python run_transformer.py --task A --batch_size 4 --max_length 512

    # Train all subtasks sequentially
    python run_transformer.py --task A B C
"""

import argparse
import os
import sys

# Ensure the project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TRANSFORMER_MODEL_NAME, TRANSFORMER_DIR, TRANSFORMER_PARAMS
from src.utils import set_seed, get_logger
from src.data_loader import load_and_subsample
from src.transformer_model import (
    train_transformer,
    evaluate_transformer,
    find_latest_checkpoint,
)

logger = get_logger("run_transformer", log_file="run_transformer.log")


def run_single_task(
    task: str,
    use_full_data: bool,
    resume: bool,
    model_name: str,
    batch_size: int | None,
    max_length: int | None,
):
    """Fine-tune + evaluate a transformer for one subtask."""
    logger.info(f"{'='*60}")
    logger.info(f" Transformer Pipeline — Task {task}")
    logger.info(f" Model: {model_name}")
    logger.info(f"{'='*60}")

    # ---- 1. Load data ----
    train_df, val_df = load_and_subsample(task, subsample=not use_full_data)

    # ---- 2. Build param overrides ----
    param_overrides = {}
    if batch_size is not None:
        param_overrides["batch_size"] = batch_size
    if max_length is not None:
        param_overrides["max_length"] = max_length

    # ---- 3. Check for checkpoint to resume from ----
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(task)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
        else:
            logger.warning("No checkpoint found — training from scratch.")

    # ---- 4. Train ----
    trainer = train_transformer(
        train_df=train_df,
        val_df=val_df,
        task=task,
        model_name=model_name,
        params=param_overrides if param_overrides else None,
        resume_from_checkpoint=checkpoint,
    )

    # ---- 5. Evaluate on full validation set ----
    #   (trainer already has eval_dataset loaded)
    results = evaluate_transformer(trainer, trainer.eval_dataset, task)
    logger.info(f"Task {task} done — Macro F1: {results['macro_f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Transformer fine-tuning pipeline")
    parser.add_argument(
        "--task", nargs="+", choices=["A", "B", "C"], default=["A"],
        help="Subtask(s) to train.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Train on the full dataset instead of the subsample.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--model", type=str, default=TRANSFORMER_MODEL_NAME,
        help=f"HuggingFace model name (default: {TRANSFORMER_MODEL_NAME}).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size (default from config).",
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Override max sequence length (default from config).",
    )
    args = parser.parse_args()

    set_seed()

    for task in args.task:
        run_single_task(
            task=task,
            use_full_data=args.full,
            resume=args.resume,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    logger.info("All tasks complete.")


if __name__ == "__main__":
    main()
