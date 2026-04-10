"""
Balance Task B data to mirror the Task A curated dataset approach:
  - Equal samples per programming language (cap majority languages)
  - Equal samples per label within each language (cap Human class)
  - Minority language-label combos kept in full; majority ones capped

Saves balanced train/val splits to TaskB/data/.
Test split is never balanced (predict on all rows).

Usage:
  python balance_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
SRC  = Path("../sem-eval-2026-task-13-subtask-b/Task_B")
DST  = Path("data")


def balance(df: pd.DataFrame, max_per_lang_label: int, seed: int = SEED) -> pd.DataFrame:
    """
    For each (language, label) group, keep at most max_per_lang_label samples.
    Groups with fewer samples are kept in full.
    Returns a shuffled DataFrame.
    """
    groups = []
    for (lang, label), grp in df.groupby(["language", "label"]):
        n = min(len(grp), max_per_lang_label)
        groups.append(grp.sample(n=n, random_state=seed))

    balanced = pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


def print_summary(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"{name}:  {len(df):,} samples")
    print(f"{'='*60}")

    print("\nLabel distribution:")
    lc = df["label"].value_counts().sort_index()
    gen_map = df.groupby("label")["generator"].first()
    for label, count in lc.items():
        pct = count / len(df) * 100
        print(f"  Label {label:2d} ({gen_map[label]:<45})  {count:>6,}  ({pct:5.1f}%)")

    print("\nLanguage distribution:")
    for lang, count in df["language"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {lang:<15}  {count:>6,}  ({pct:5.1f}%)")

    print("\nLanguage x Label (sample counts):")
    ct = df.groupby(["language", "label"]).size().unstack(fill_value=0)
    print(ct.to_string())


def main():
    DST.mkdir(parents=True, exist_ok=True)

    # ── Load raw splits ────────────────────────────────────────────────────
    print("Loading raw data...")
    train_raw = pd.read_parquet(SRC / "train.parquet")
    val_raw   = pd.read_parquet(SRC / "validation.parquet")
    test_raw  = pd.read_parquet(SRC / "test.parquet")

    n_langs  = train_raw["language"].nunique()   # 8
    n_labels = train_raw["label"].nunique()       # 11

    print(f"Raw train: {len(train_raw):,} samples  |  {n_langs} languages  |  {n_labels} labels")

    # ── Determine cap ──────────────────────────────────────────────────────
    # Target ~50k total (matching Task A scale).
    # cap = target / (n_langs * n_labels), rounded to nearest 50.
    target_total   = 50_000
    cap_per_group  = target_total // (n_langs * n_labels)   # 50000 / 88 ≈ 568
    cap_per_group  = max(cap_per_group // 50 * 50, 50)      # round down to nearest 50
    print(f"Cap per (language, label) group: {cap_per_group}")

    # ── Balance splits ─────────────────────────────────────────────────────
    train_bal = balance(train_raw, cap_per_group)
    val_bal   = balance(val_raw,   cap_per_group)
    # Test is never balanced — predict on all rows
    test_out  = test_raw.copy()

    # ── Print summaries ────────────────────────────────────────────────────
    print_summary(train_bal, "Balanced Train")
    print_summary(val_bal,   "Balanced Validation")

    # ── Save ───────────────────────────────────────────────────────────────
    train_bal.to_parquet(DST / "task_b_training_set.parquet",   index=False)
    val_bal.to_parquet(  DST / "task_b_validation_set.parquet", index=False)
    test_out.to_parquet( DST / "task_b_test_set.parquet",       index=False)

    print(f"\n{'='*60}")
    print("Saved:")
    print(f"  {DST}/task_b_training_set.parquet    ({len(train_bal):,} samples)")
    print(f"  {DST}/task_b_validation_set.parquet  ({len(val_bal):,} samples)")
    print(f"  {DST}/task_b_test_set.parquet        ({len(test_out):,} samples — unbalanced)")


if __name__ == "__main__":
    main()
