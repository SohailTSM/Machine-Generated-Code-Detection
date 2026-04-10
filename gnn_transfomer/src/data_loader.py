"""Dataset I/O — parquet loading, label remapping, validation, class weights."""

import os
from collections import Counter

import pandas as pd
import torch

from src import config as cfg


def load_task_data(task: str = None, data_dir: str = None):
    """Load train/val parquets. Priority: local cache → Kaggle mount → HuggingFace.

    Returns:
        (train_df, val_df) — val_df may be None if not available.
    """
    task = task or cfg.ACTIVE_TASK
    data_dir = data_dir or cfg.DATA_DIR
    task_cfg = cfg.TASK_CONFIG[task]

    local_dir = os.path.join(data_dir, f"task_{task.lower()}")
    local_train = os.path.join(local_dir, "train.parquet")
    local_val = os.path.join(local_dir, "validation.parquet")

    # Priority 1: Local parquet files
    if os.path.exists(local_train):
        train = pd.read_parquet(local_train)
        val = pd.read_parquet(local_val) if os.path.exists(local_val) else None
        print(f"✓ Loaded Task {task} from local: {len(train):,} train"
              + (f", {len(val):,} val" if val is not None else ""))
        return _postprocess(train, val, task)

    # Priority 2: Kaggle input mount
    for pattern in [
        f"/kaggle/input/sem-eval-2026-task-13-subtask-{task.lower()}",
        f"/kaggle/input/semeval-2026-task13",
    ]:
        kgl_train = os.path.join(pattern, f"Task_{task}", "train.parquet")
        if not os.path.exists(kgl_train):
            kgl_train = os.path.join(pattern, f"task_{task.lower()}",
                                     f"task_{task.lower()}_training_set_1.parquet")
        if os.path.exists(kgl_train):
            train = pd.read_parquet(kgl_train)
            # Try various val paths
            for vp in [
                os.path.join(pattern, f"Task_{task}", "validation.parquet"),
                os.path.join(pattern, f"task_{task.lower()}",
                             f"task_{task.lower()}_validation_set.parquet"),
            ]:
                if os.path.exists(vp):
                    val = pd.read_parquet(vp)
                    break
            else:
                val = None
            print(f"✓ Loaded Task {task} from Kaggle: {len(train):,} train")
            return _postprocess(train, val, task)

    # Priority 3: HuggingFace API
    from datasets import load_dataset
    hf_name = task_cfg.get("hf_dataset", cfg.HF_DATASET_NAME)
    hf_config = task_cfg["hf_config"]
    print(f"  Downloading from HF: {hf_name} config={hf_config}")
    ds = load_dataset(hf_name, hf_config)
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas() if "validation" in ds else None

    # Cache locally — use WORK_DIR (writable) instead of DATA_DIR (may be read-only)
    cache_dir = os.path.join(cfg.WORK_DIR, f"task_{task.lower()}")
    os.makedirs(cache_dir, exist_ok=True)
    train.to_parquet(os.path.join(cache_dir, "train.parquet"), index=False)
    if val is not None:
        val.to_parquet(os.path.join(cache_dir, "validation.parquet"), index=False)

    # Also cache test split as test_sample if available
    if "test" in ds:
        test_sample = ds["test"].to_pandas()
        test_sample_path = os.path.join(cache_dir, "test_sample.parquet")
        test_sample.to_parquet(test_sample_path, index=False)
        print(f"  ✓ Cached test_sample: {len(test_sample)} rows → {test_sample_path}")

    print(f"✓ Downloaded Task {task} from HF: {len(train):,} train")
    return _postprocess(train, val, task)


def _postprocess(train_df, val_df, task):
    """Apply label remapping, cleaning, and DEBUG subsampling."""
    task_cfg = cfg.TASK_CONFIG[task]

    # Task A: remap 11-class → binary
    if task_cfg.get("train_label_remap"):
        train_df["label"] = (train_df["label"] > 0).astype(int)
        print(f"  ⚠ Task {task} train labels remapped to binary (11→2)")

    # Clean: drop null/empty code or label
    for split_name, df in [("train", train_df), ("val", val_df)]:
        if df is None:
            continue
        mask = df["code"].notna() & (df["code"].str.strip() != "") & df["label"].notna()
        dropped = (~mask).sum()
        if dropped > 0:
            if split_name == "train":
                train_df = df[mask].reset_index(drop=True)
            else:
                val_df = df[mask].reset_index(drop=True)
            print(f"  Dropped {dropped} invalid rows from {split_name}")

    # Add orig_id — use ID column from parquet if present, else 0-based
    if "ID" in train_df.columns:
        train_df["orig_id"] = train_df["ID"].values
    else:
        train_df["orig_id"] = range(len(train_df))
    if val_df is not None:
        if "ID" in val_df.columns:
            val_df["orig_id"] = val_df["ID"].values
        else:
            val_df["orig_id"] = range(len(val_df))

    # Per-task sample cap (e.g. Task C: 100K train, 20K val)
    task_cfg = cfg.TASK_CONFIG[cfg.ACTIVE_TASK]
    max_train = task_cfg.get("max_train_samples")
    max_val = task_cfg.get("max_val_samples")
    if max_train and len(train_df) > max_train:
        print(f"  Capping train from {len(train_df):,} → {max_train:,} (stratified)")
        train_df = _stratified_subsample(train_df, max_train)
    if max_val and val_df is not None and len(val_df) > max_val:
        print(f"  Capping val from {len(val_df):,} → {max_val:,} (stratified)")
        val_df = _stratified_subsample(val_df, max_val)

    # DEBUG subsampling
    if cfg.DEBUG:
        train_df = _stratified_subsample(train_df, cfg.DEBUG_TRAIN_SAMPLES)
        if val_df is not None:
            val_df = _stratified_subsample(val_df, cfg.DEBUG_VAL_SAMPLES)
        print(f"  DEBUG mode: {len(train_df)} train, "
              f"{len(val_df) if val_df is not None else 0} val")

    return train_df, val_df


def _stratified_subsample(df, n):
    """Stratified subsample preserving class distribution."""
    if len(df) <= n:
        return df
    pieces = []
    for _, group in df.groupby("label"):
        k = max(1, int(len(group) / len(df) * n))
        pieces.append(group.sample(n=k, random_state=cfg.SEED))
    return pd.concat(pieces, ignore_index=True)


def validate_labels(df, task: str = None, split: str = ""):
    """Assert label range matches TASK_CONFIG."""
    task = task or cfg.ACTIVE_TASK
    num_classes = cfg.TASK_CONFIG[task]["num_classes"]
    expected = set(range(num_classes))
    actual = set(df["label"].dropna().astype(int).unique())
    if not actual.issubset(expected):
        raise ValueError(
            f"[{split}] Unexpected labels: {actual - expected}. "
            f"Expected {expected} for task {task}."
        )
    print(f"  ✓ {split} labels valid: {sorted(actual)} (expected {num_classes} classes)")


def get_class_weights(df, num_classes: int = None, device=None):
    """Compute inverse-frequency class weights for loss function."""
    num_classes = num_classes or cfg.NUM_CLASSES
    device = device or cfg.DEVICE
    counts = Counter(df["label"].astype(int).tolist())
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    w = torch.tensor(weights, dtype=torch.float, device=device)
    return w / w.sum() * num_classes


def check_leakage(train_df, val_df, sample_chars=1000):
    """Check for train/val overlap using code hash."""
    if val_df is None:
        return
    train_hashes = set(train_df["code"].str[:sample_chars].apply(hash))
    val_hashes = set(val_df["code"].str[:sample_chars].apply(hash))
    overlap = train_hashes & val_hashes
    if overlap:
        print(f"  ⚠ Potential leakage: {len(overlap)} overlapping code hashes!")
    else:
        print(f"  ✓ No train/val leakage detected")


def load_test_sample(task: str = None, data_dir: str = None):
    """Load test_sample.parquet for the given task (labeled test data for evaluation).

    Returns DataFrame or None if not found.
    """
    task = task or cfg.ACTIVE_TASK
    data_dir = data_dir or cfg.DATA_DIR
    task_cfg = cfg.TASK_CONFIG[task]

    candidates = [
        os.path.join(data_dir, f"task_{task.lower()}", "test_sample.parquet"),
        os.path.join(cfg.WORK_DIR, f"task_{task.lower()}", "test_sample.parquet"),
    ]
    if cfg.ENV == "kaggle":
        candidates.insert(0, f"/kaggle/input/semeval-2026-task13/task_{task.lower()}/test_sample.parquet")

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Apply same label remapping as training data
            if task_cfg.get("train_label_remap"):
                df["label"] = (df["label"] > 0).astype(int)
            # Clean
            mask = df["code"].notna() & (df["code"].str.strip() != "") & df["label"].notna()
            df = df[mask].reset_index(drop=True)
            if "ID" in df.columns:
                df["orig_id"] = df["ID"].values
            else:
                df["orig_id"] = range(len(df))
            print(f"✓ Loaded test_sample for Task {task}: {len(df)} samples from {path}")
            return df

    print(f"⚠ test_sample not found for Task {task}")
    return None
