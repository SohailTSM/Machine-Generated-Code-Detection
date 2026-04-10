"""
Transformer (ModernBERT / UniXcoder / CodeBERT) fine-tuning for sequence
classification, with:

  - Macro F1 as the primary metric
  - Class-weighted cross-entropy for imbalanced tasks (B, C)
  - Checkpoint saving every N steps so training can be resumed
  - Mixed precision (fp16) for speed
  - Multi-core data loading
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

from src.config import (
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_PARAMS,
    TRANSFORMER_DIR,
    SUBTASK_NUM_LABELS,
    SUBTASK_LABEL_NAMES,
    SEED,
    NUM_WORKERS,
)
from src.utils import (
    get_logger,
    set_seed,
    print_confusion_matrix,
    save_json,
    get_device,
)
from src.evaluation import evaluate_predictions

logger = get_logger(__name__, log_file="transformer.log")


# ======================================================================
# Custom Trainer with class-weighted loss
# ======================================================================
class WeightedTrainer(Trainer):
    """
    Subclass of HuggingFace Trainer that applies class-weight to the
    cross-entropy loss.  When self.class_weights is None it behaves
    identically to the default Trainer.
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float32
            )
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ======================================================================
# Tokenisation helpers
# ======================================================================
def _make_tokenize_fn(tokenizer, max_length: int):
    """Return a function suitable for Dataset.map()."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
            # padding is handled by DataCollatorWithPadding
        )
    return tokenize_fn


# ======================================================================
# Compute metrics callback
# ======================================================================
def _compute_metrics(eval_pred):
    """Metrics logged during training.  Primary metric: macro_f1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": accuracy}


# ======================================================================
# Compute class weights from label distribution
# ======================================================================
def _compute_class_weights(labels: np.ndarray, num_classes: int) -> list[float]:
    """
    Inverse-frequency class weights, normalised so that the mean weight = 1.0.

    This prevents the loss scale from changing dramatically compared to
    unweighted training.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Avoid division by zero for classes with 0 samples
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()  # normalise
    return weights.tolist()


# ======================================================================
# Main training function
# ======================================================================
def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str,
    model_name: str | None = None,
    params: dict | None = None,
    resume_from_checkpoint: str | None = None,
):
    """
    Fine-tune a transformer for the given subtask.

    Parameters
    ----------
    train_df, val_df : DataFrames with 'code' and 'label' columns
    task             : 'A', 'B', or 'C'
    model_name       : HuggingFace model name (defaults to config)
    params           : override TRANSFORMER_PARAMS
    resume_from_checkpoint : path to a HF checkpoint dir to resume from

    Returns
    -------
    trainer : the HuggingFace Trainer (can be used for prediction)
    """
    set_seed(SEED)

    model_name = model_name or TRANSFORMER_MODEL_NAME
    p = {**TRANSFORMER_PARAMS, **(params or {})}
    num_labels = SUBTASK_NUM_LABELS[task]
    output_dir = os.path.join(TRANSFORMER_DIR, f"{task}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Fine-tuning '{model_name}' for Task {task} ({num_labels} classes)")
    logger.info(f"Train: {len(train_df):,}   Val: {len(val_df):,}")
    logger.info(f"Params: {p}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )

    # ---- Datasets ----
    tokenize_fn = _make_tokenize_fn(tokenizer, p["max_length"])

    train_ds = Dataset.from_pandas(train_df[["code", "label"]].rename(columns={"label": "labels"}))
    val_ds   = Dataset.from_pandas(val_df[["code", "label"]].rename(columns={"label": "labels"}))

    train_ds = train_ds.map(tokenize_fn, batched=True, num_proc=min(4, NUM_WORKERS), remove_columns=["code"])
    val_ds   = val_ds.map(tokenize_fn, batched=True, num_proc=min(4, NUM_WORKERS), remove_columns=["code"])

    # ---- Class weights (for tasks B and C) ----
    class_weights = None
    if task in ("B", "C"):
        class_weights = _compute_class_weights(train_df["label"].values, num_labels)
        logger.info(f"Class weights: {[f'{w:.3f}' for w in class_weights]}")

    # ---- Mixed precision: fp16 on CUDA, bf16 on MPS/Apple Silicon ----
    use_fp16 = p["fp16"] and torch.cuda.is_available()
    use_bf16 = (
        not use_fp16
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=p["num_epochs"],
        per_device_train_batch_size=p["batch_size"],
        per_device_eval_batch_size=p["batch_size"] * 2,  # eval can use bigger batch
        gradient_accumulation_steps=p["gradient_accumulation_steps"],
        learning_rate=p["learning_rate"],
        warmup_ratio=p["warmup_ratio"],
        weight_decay=p["weight_decay"],
        lr_scheduler_type=p["lr_scheduler_type"],
        fp16=use_fp16,
        bf16=use_bf16,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=p["logging_steps"],
        eval_strategy="steps",
        eval_steps=p["eval_steps"],
        save_strategy="steps",
        save_steps=p["save_steps"],
        save_total_limit=p["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",        # <-- competition metric
        greater_is_better=True,
        dataloader_num_workers=p["dataloader_num_workers"],
        seed=SEED,
        report_to="none",   # disable W&B / MLflow etc.
        remove_unused_columns=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=p["early_stopping_patience"]),
        ],
    )

    # ---- Train (or resume) ----
    logger.info("Starting training ...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ---- Save best model -----
    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"Best model saved to {best_dir}")

    return trainer


# ======================================================================
# Evaluation
# ======================================================================
def evaluate_transformer(trainer: Trainer, val_ds, task: str) -> dict:
    """
    Run full evaluation, log classification report + confusion matrix.

    Returns dict with 'macro_f1' and 'predictions'.
    """
    label_names = SUBTASK_LABEL_NAMES[task]
    preds_output = trainer.predict(val_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids

    metrics = evaluate_predictions(y_true, y_pred, label_names)
    macro_f1 = metrics["macro_f1"]
    report = metrics["report"]

    logger.info(f"Task {task} — Transformer Macro F1: {macro_f1:.4f}")
    logger.info(f"Task {task} — Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Task {task} — Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Task {task} — Macro Recall: {metrics['macro_recall']:.4f}")
    logger.info(f"\n{report}")
    print_confusion_matrix(y_true, y_pred, label_names, logger=logger)

    output_dir = os.path.join(TRANSFORMER_DIR, task)
    save_json(metrics, os.path.join(output_dir, "eval_results.json"))

    return {"macro_f1": macro_f1, "predictions": y_pred, "metrics": metrics}


# ======================================================================
# Probability predictions (for ensemble)
# ======================================================================
def predict_proba_transformer(
    model_dir: str,
    codes: list[str],
    max_length: int | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Load a saved transformer and return class probabilities for a list of
    code snippets.

    Returns: np.ndarray of shape (n_samples, n_classes).
    """
    max_length = max_length or TRANSFORMER_PARAMS["max_length"]
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    all_probs = []
    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i : i + batch_size]
        inputs = tokenizer(
            batch_codes,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ======================================================================
# Find latest checkpoint for resuming
# ======================================================================
def find_latest_checkpoint(task: str) -> str | None:
    """
    Scan the transformer output directory for the latest checkpoint-XXXX dir.
    Returns the path, or None if no checkpoint exists.
    """
    output_dir = os.path.join(TRANSFORMER_DIR, task)
    if not os.path.isdir(output_dir):
        return None

    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = os.path.join(output_dir, checkpoints[-1])
    logger.info(f"Found latest checkpoint: {latest}")
    return latest
