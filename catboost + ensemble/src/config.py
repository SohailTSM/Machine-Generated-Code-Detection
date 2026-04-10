"""
Central configuration for the SemEval-2026 Task 13 pipeline.

All hyperparameters, paths, and constants are defined here so that
every module in the project reads from a single source of truth.
"""

import os
import multiprocessing

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CATBOOST_DIR = os.path.join(OUTPUT_DIR, "catboost")
TRANSFORMER_DIR = os.path.join(OUTPUT_DIR, "transformer")
ENSEMBLE_DIR = os.path.join(OUTPUT_DIR, "ensemble")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create all output directories on import
for _d in [OUTPUT_DIR, CATBOOST_DIR, TRANSFORMER_DIR, ENSEMBLE_DIR,
           FEATURES_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
# Use all available CPU cores for parallel feature extraction / CatBoost
NUM_WORKERS = multiprocessing.cpu_count()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
HF_DATASET_NAME = "DaniilOr/SemEval-2026-Task13"

# Subsample sizes used for fast iteration (set to None to use full data)
# Current values tuned for a quick test run on MacBook M4 Air (24 GB).
# For full training, restore to A:50_000  B:108_000  C:100_000.
SUBSAMPLE_SIZES = {
    "A": 100_000,   # 5K per class  — quick test
    "B": 15_000,   # keeps all LLM minorities + ~7K Human
    "C": 10_000,   # balanced across 4 classes
}

# Validation subsample sizes (set to None to use full val set).
# Should be smaller than or equal to train subsample for quick tests.
VAL_SUBSAMPLE_SIZES = {
    "A": 50_000,
    "B": 5_000,
    "C": 5_000,
}

# ---------------------------------------------------------------------------
# Subtask metadata
# ---------------------------------------------------------------------------
SUBTASK_NUM_LABELS = {
    "A": 2,   # human vs machine
    "B": 11,  # human + 10 LLM families
    "C": 4,   # human / machine / hybrid / adversarial
}

SUBTASK_LABEL_NAMES = {
    "A": ["Human", "Machine"],
    "B": [
        "Human", "DeepSeek-AI", "Qwen", "01-ai", "BigCode",
        "Gemma", "Phi", "Meta-LLaMA", "IBM-Granite", "Mistral", "OpenAI",
    ],
    "C": ["Human", "Machine", "Hybrid", "Adversarial"],
}

# ---------------------------------------------------------------------------
# CatBoost hyper-parameters
# ---------------------------------------------------------------------------
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 8,
    "eval_metric": "TotalF1:average=Macro;use_weights=false",
    "loss_function": "MultiClass",
    "random_seed": 42,
    "verbose": 100,                   # print every 100 iterations
    "early_stopping_rounds": 120,
    "thread_count": NUM_WORKERS,      # use all cores
    "task_type": "CPU",
    "auto_class_weights": "Balanced",  # handles class imbalance
}

# ---------------------------------------------------------------------------
# Transformer hyper-parameters
# ---------------------------------------------------------------------------
# Primary model — ModernBERT with 8192 context
TRANSFORMER_MODEL_NAME = "answerdotai/ModernBERT-base"

TRANSFORMER_PARAMS = {
    "max_length": 512,             # shorter for quick M4 test (restore to 1024+)
    "batch_size": 16,              # M4 24 GB unified memory handles this
    "gradient_accumulation_steps": 2,  # effective batch = 32
    "learning_rate": 2e-5,
    "num_epochs": 2,               # quick test (restore to 5 for full run)
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "fp16": False,                 # MPS does not support fp16 reliably
    "save_steps": 200,
    "eval_steps": 200,
    "logging_steps": 50,
    "save_total_limit": 2,         # keep last 2 checkpoints
    "early_stopping_patience": 2,
    "dataloader_num_workers": 0,   # MPS + multiprocess dataloaders can deadlock
}

# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
ENSEMBLE_META_LEARNER = "logistic_regression"  # or "mlp"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
