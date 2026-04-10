"""
Data loading, profiling, and stratified subsampling for SemEval-2026 Task 13.

Loads from HuggingFace, uses official train / validation splits, and creates
smaller stratified subsets for fast iteration.
"""

import os
import pandas as pd
from datasets import load_dataset

from src.config import (
    HF_DATASET_NAME,
    SUBSAMPLE_SIZES,
    VAL_SUBSAMPLE_SIZES,
    FEATURES_DIR,
    SEED,
)
from src.utils import get_logger

logger = get_logger(__name__, log_file="data_loader.log")


def _normalize_code_text(code: str) -> str:
    """Normalize line endings/whitespace while preserving indentation signal."""
    if not isinstance(code, str):
        code = "" if code is None else str(code)

    # Normalize line endings first.
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    # Expand tabs deterministically so indent-based features remain consistent.
    code = code.replace("\t", "    ")
    # Strip trailing whitespace per line (keep leading indentation).
    code = "\n".join(line.rstrip() for line in code.split("\n"))
    return code


def _log_duplicate_and_length_diagnostics(train_df: pd.DataFrame, val_df: pd.DataFrame, task: str):
    """Log diagnostics useful for generalization and leakage checks."""
    train_hash = pd.util.hash_pandas_object(train_df["code"], index=False)
    val_hash = pd.util.hash_pandas_object(val_df["code"], index=False)

    train_dup = int(train_hash.duplicated().sum())
    val_dup = int(val_hash.duplicated().sum())
    cross_overlap = int(len(set(train_hash.unique()) & set(val_hash.unique())))

    logger.info(
        f"Task {task} duplicate diagnostics — "
        f"train_internal: {train_dup}, val_internal: {val_dup}, train_val_overlap: {cross_overlap}"
    )

    for split_name, df in [("train", train_df), ("val", val_df)]:
        line_counts = df["code"].str.count("\n") + 1
        short_5 = float((line_counts <= 5).mean() * 100)
        short_10 = float((line_counts <= 10).mean() * 100)
        short_20 = float((line_counts <= 20).mean() * 100)
        logger.info(
            f"Task {task} {split_name} short-snippet distribution — "
            f"<=5 lines: {short_5:.2f}%, <=10 lines: {short_10:.2f}%, <=20 lines: {short_20:.2f}%"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_task_data(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation splits for a given subtask (A, B, or C).

    Returns two DataFrames, each with at least columns: 'code', 'label'.
    Additional metadata columns (language, generator, etc.) are kept.
    """
    logger.info(f"Loading HuggingFace dataset '{HF_DATASET_NAME}', subset '{task}' ...")
    dataset = load_dataset(HF_DATASET_NAME, task)

    train_df = dataset["train"].to_pandas()
    val_df   = dataset["validation"].to_pandas()

    # Basic sanity checks
    for name, df in [("train", train_df), ("val", val_df)]:
        assert "code" in df.columns and "label" in df.columns, (
            f"{name} split must contain 'code' and 'label' columns. "
            f"Found: {df.columns.tolist()}"
        )
        # Drop nulls
        before = len(df)
        df.dropna(subset=["code", "label"], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} null rows from {name} split.")

        # Deterministic text normalization to reduce accidental format leakage.
        df["code"] = df["code"].astype(str).map(_normalize_code_text)

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"]   = val_df["label"].astype(int)

    logger.info(f"Task {task} — train: {len(train_df):,}  val: {len(val_df):,}")
    logger.info(f"Train label distribution:\n{train_df['label'].value_counts().sort_index().to_string()}")
    _log_duplicate_and_length_diagnostics(train_df, val_df, task)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------
def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    label_col: str = "label",
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Return a stratified subsample of *df* with at most *n* rows.

    For heavily imbalanced tasks (like Subtask B), minority classes are kept
    in full and the majority class is down-sampled to fill the budget.
    """
    counts = df[label_col].value_counts()
    total = len(df)
    if n >= total:
        logger.info("Requested subsample >= dataset size — returning full data.")
        return df

    # Strategy: keep all minority-class samples if they fit in the budget,
    # then allocate the remaining budget to the majority class.
    sorted_classes = counts.sort_values().index.tolist()   # smallest first
    budget = n
    keep_per_class = {}

    for cls in sorted_classes:
        cls_count = counts[cls]
        remaining_classes = len(sorted_classes) - len(keep_per_class)
        fair_share = budget // max(remaining_classes, 1)

        if cls_count <= fair_share:
            # Minority class — keep everything
            keep_per_class[cls] = cls_count
        else:
            keep_per_class[cls] = fair_share
        budget -= keep_per_class[cls]

    # Sample each class
    parts = []
    for cls, k in keep_per_class.items():
        cls_df = df[df[label_col] == cls]
        if k >= len(cls_df):
            parts.append(cls_df)
        else:
            parts.append(cls_df.sample(n=k, random_state=seed))

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        f"Subsampled {len(df):,} → {len(result):,} rows.  "
        f"Per-class: {result[label_col].value_counts().sort_index().to_dict()}"
    )
    return result


def load_and_subsample(
    task: str,
    subsample: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: load data + optionally subsample both splits.

    When subsample=True, both train and val are reduced for fast iteration.
    Use subsample=False (--full flag) to train/eval on full data.
    """
    train_df, val_df = load_task_data(task)

    if subsample and SUBSAMPLE_SIZES.get(task):
        logger.info(f"Creating stratified subsample of train split (n={SUBSAMPLE_SIZES[task]:,}) ...")
        train_df = stratified_subsample(train_df, SUBSAMPLE_SIZES[task])

    if subsample and VAL_SUBSAMPLE_SIZES.get(task):
        logger.info(f"Creating stratified subsample of val split (n={VAL_SUBSAMPLE_SIZES[task]:,}) ...")
        val_df = stratified_subsample(val_df, VAL_SUBSAMPLE_SIZES[task])

    return train_df, val_df


# ---------------------------------------------------------------------------
# Profiling (optional — run once to understand the data)
# ---------------------------------------------------------------------------
def profile_dataset(df: pd.DataFrame, name: str = "dataset"):
    """Print useful statistics about a code dataset."""
    logger.info(f"--- Profile: {name} ({len(df):,} rows) ---")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")

    # Code length stats (characters)
    lengths = df["code"].str.len()
    logger.info(
        f"Code length (chars) — min: {lengths.min()}, "
        f"median: {lengths.median():.0f}, mean: {lengths.mean():.0f}, "
        f"max: {lengths.max()}, p95: {lengths.quantile(0.95):.0f}"
    )

    # Rough token count (whitespace split — cheaper than running a tokenizer)
    token_counts = df["code"].str.split().str.len()
    logger.info(
        f"Rough token count — median: {token_counts.median():.0f}, "
        f"mean: {token_counts.mean():.0f}, "
        f"p95: {token_counts.quantile(0.95):.0f}, "
        f"max: {token_counts.max()}"
    )

    if "language" in df.columns:
        logger.info(f"Languages: {df['language'].value_counts().to_dict()}")
    if "generator" in df.columns:
        logger.info(f"Generators: {df['generator'].value_counts().to_dict()}")
