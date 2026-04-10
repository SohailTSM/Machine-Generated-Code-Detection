"""
Balance Task C data mirroring the Task A curated dataset approach:
  - Equal samples per programming language (cap majority languages)
  - Equal samples per label within each language (cap Human and large-label classes)
  - Minority language-label combos kept in full; majority ones capped

Task C labels:
  0 = Human
  1 = Large proprietary models (GPT-4o, GPT-4o-mini, Qwen-7B-Instruct, ...)
  2 = Large open-source models (Qwen-72B, Qwen-32B, Granite-34B, ...)
  3 = Small/mid models (DeepSeek-coder, Yi-Coder, ...)

Saves balanced train/val splits to TaskC/data/.
Test split is never balanced (predict on all rows).

Usage:
  python balance_data.py
"""

import pandas as pd
from pathlib import Path

SEED = 42
SRC  = Path("../sem-eval-2026-task-13-subtask-c/Task_C")
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

    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)


def print_summary(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"{name}:  {len(df):,} samples")
    print(f"{'='*60}")

    print("\nLabel distribution:")
    lc = df["label"].value_counts().sort_index()
    for label, count in lc.items():
        pct = count / len(df) * 100
        top_gen = df[df["label"] == label]["generator"].value_counts().index[0]
        print(f"  Label {label}  ({top_gen:<45})  {count:>6,}  ({pct:5.1f}%)")

    print("\nLanguage distribution:")
    for lang, count in df["language"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {lang:<15}  {count:>6,}  ({pct:5.1f}%)")

    print("\nLanguage x Label (sample counts):")
    ct = df.groupby(["language", "label"]).size().unstack(fill_value=0)
    print(ct.to_string())


def main():
    DST.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    train_raw = pd.read_parquet(SRC / "train.parquet")
    val_raw   = pd.read_parquet(SRC / "validation.parquet")
    test_raw  = pd.read_parquet(SRC / "test.parquet")

    n_langs  = train_raw["language"].nunique()   # 8
    n_labels = train_raw["label"].nunique()       # 4

    print(f"Raw train: {len(train_raw):,} samples  |  {n_langs} languages  |  {n_labels} labels")

    # Target ~160k total
    # cap = 160000 / (8 langs * 4 labels) = 5000
    target_total  = 160_000
    cap_per_group = target_total // (n_langs * n_labels)
    cap_per_group = max(cap_per_group // 50 * 50, 50)
    print(f"Cap per (language, label) group: {cap_per_group}")

    train_bal = balance(train_raw, cap_per_group)
    val_bal   = balance(val_raw,   cap_per_group)
    test_out  = test_raw.copy()  # never balanced

    print_summary(train_bal, "Balanced Train")
    print_summary(val_bal,   "Balanced Validation")

    train_bal.to_parquet(DST / "task_c_training_set.parquet",   index=False)
    val_bal.to_parquet(  DST / "task_c_validation_set.parquet", index=False)
    test_out.to_parquet( DST / "task_c_test_set.parquet",       index=False)

    print(f"\n{'='*60}")
    print("Saved:")
    print(f"  {DST}/task_c_training_set.parquet    ({len(train_bal):,} samples)")
    print(f"  {DST}/task_c_validation_set.parquet  ({len(val_bal):,} samples)")
    print(f"  {DST}/task_c_test_set.parquet        ({len(test_out):,} samples — unbalanced)")


if __name__ == "__main__":
    main()
