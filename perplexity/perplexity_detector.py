"""
Perplexity-based Machine-Generated Code Detection.

This module computes perplexity scores using an autoregressive language model
and finds an optimal threshold to classify code as Human vs Machine.

Intuition: AI-generated code may have lower perplexity (higher likelihood)
under a language model because LLMs produce "typical" code patterns.
"""

import os
import math
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERPLEXITY_CONFIG = {
    # Model options (from smallest to largest):
    # - "microsoft/CodeGPT-small-py" (124M, Python-only)
    # - "Salesforce/codegen-350M-multi" (350M, C/C++/Go/Java/JS/Python)
    # - "bigcode/starcoderbase-1b" (1B, multilingual)
    "model_name": "Salesforce/codegen-350M-multi",  # Multi-language model
    "max_length": 512,
    "batch_size": 4,  # Reduced for larger model
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "stride": 256,  # sliding window stride for long sequences
}


# ---------------------------------------------------------------------------
# Perplexity Computation
# ---------------------------------------------------------------------------
def load_model(model_name: str = None, device: str = None):
    """Load the autoregressive model and tokenizer."""
    model_name = model_name or PERPLEXITY_CONFIG["model_name"]
    device = device or PERPLEXITY_CONFIG["device"]
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def compute_perplexity_single(
    code: str,
    model,
    tokenizer,
    max_length: int = 512,
    stride: int = 256,
    device: str = "cpu",
) -> float:
    """
    Compute perplexity of a single code snippet.
    Uses sliding window for sequences longer than max_length.
    """
    if not code or len(code.strip()) == 0:
        return float("inf")
    
    try:
        encodings = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_length * 4,
        )
    except Exception:
        return float("inf")
    
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)
    
    if seq_len == 0:
        return float("inf")
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood.item())
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break
    
    total_nll = sum(nlls)
    num_tokens = prev_end_loc
    
    if num_tokens == 0:
        return float("inf")
    
    return math.exp(total_nll / num_tokens)


def compute_perplexities_batch(
    codes: list,
    model,
    tokenizer,
    max_length: int = 512,
    stride: int = 256,
    device: str = "cpu",
    desc: str = "Computing perplexity",
) -> np.ndarray:
    """Compute perplexity for a batch of code snippets."""
    perplexities = []
    
    for code in tqdm(codes, desc=desc):
        ppl = compute_perplexity_single(
            code, model, tokenizer, max_length, stride, device
        )
        perplexities.append(ppl)
    
    return np.array(perplexities)


# ---------------------------------------------------------------------------
# Threshold Finding
# ---------------------------------------------------------------------------
def find_optimal_threshold(
    perplexities: np.ndarray,
    labels: np.ndarray,
    method: str = "youden",
) -> tuple:
    """
    Find optimal perplexity threshold for binary classification.
    
    Rule: perplexity < threshold → Machine (1)
    """
    valid_mask = np.isfinite(perplexities)
    ppl_valid = perplexities[valid_mask]
    labels_valid = labels[valid_mask]
    
    print(f"Valid samples: {len(ppl_valid)} / {len(perplexities)}")
    
    if len(ppl_valid) == 0:
        return 100.0, {"roc_auc": 0.5, "optimal_threshold": 100.0}
    
    # Low perplexity → Machine, so use negative as score
    scores = -ppl_valid
    
    fpr, tpr, thresholds = roc_curve(labels_valid, scores)
    
    if method == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
    else:
        best_f1 = 0
        best_idx = 0
        for i, thresh in enumerate(thresholds):
            preds = (scores >= thresh).astype(int)
            f1 = f1_score(labels_valid, preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_idx = i
    
    optimal_threshold = -thresholds[best_idx]
    roc_auc = auc(fpr, tpr)
    
    stats = {
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_threshold,
        "tpr_at_threshold": float(tpr[best_idx]),
        "fpr_at_threshold": float(fpr[best_idx]),
        "num_valid_samples": int(len(ppl_valid)),
    }
    
    return optimal_threshold, stats


def classify_with_threshold(perplexities: np.ndarray, threshold: float) -> np.ndarray:
    """Classify: perplexity < threshold → Machine (1), else Human (0)."""
    predictions = (perplexities < threshold).astype(int)
    predictions[~np.isfinite(perplexities)] = 0  # Infinite → Human
    return predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_perplexity_classifier(
    perplexities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    label_names: list = None,
) -> dict:
    """Evaluate perplexity-based classifier and return metrics."""
    label_names = label_names or ["Human", "Machine"]
    
    predictions = classify_with_threshold(perplexities, threshold)
    
    valid_mask = np.isfinite(perplexities)
    ppl_valid = perplexities[valid_mask]
    labels_valid = labels[valid_mask]
    preds_valid = predictions[valid_mask]
    
    if len(ppl_valid) == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0, "error": "No valid samples"}
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_valid, preds_valid, average=None, labels=[0, 1]
    )
    macro_f1 = f1_score(labels_valid, preds_valid, average="macro")
    accuracy = float((preds_valid == labels_valid).mean())
    
    report = classification_report(
        labels_valid, preds_valid,
        target_names=label_names,
        digits=4,
    )
    
    cm = confusion_matrix(labels_valid, preds_valid)
    
    human_ppl = ppl_valid[labels_valid == 0]
    machine_ppl = ppl_valid[labels_valid == 1]
    
    return {
        "macro_f1": float(macro_f1),
        "accuracy": accuracy,
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
        "f1_per_class": [float(f) for f in f1],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
        "perplexity_stats": {
            "human_mean": float(np.mean(human_ppl)) if len(human_ppl) > 0 else 0.0,
            "human_std": float(np.std(human_ppl)) if len(human_ppl) > 0 else 0.0,
            "human_median": float(np.median(human_ppl)) if len(human_ppl) > 0 else 0.0,
            "machine_mean": float(np.mean(machine_ppl)) if len(machine_ppl) > 0 else 0.0,
            "machine_std": float(np.std(machine_ppl)) if len(machine_ppl) > 0 else 0.0,
            "machine_median": float(np.median(machine_ppl)) if len(machine_ppl) > 0 else 0.0,
        },
        "num_valid": int(len(ppl_valid)),
        "num_infinite": int(len(perplexities) - len(ppl_valid)),
    }


def print_results(results: dict, split_name: str = "Evaluation"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f" {split_name} Results")
    print(f"{'='*60}")
    print(f"Threshold: {results['threshold']:.2f}")
    print(f"Macro F1:  {results['macro_f1']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"\nPerplexity Statistics:")
    stats = results["perplexity_stats"]
    print(f"  Human:   mean={stats['human_mean']:.2f}, median={stats['human_median']:.2f}, std={stats['human_std']:.2f}")
    print(f"  Machine: mean={stats['machine_mean']:.2f}, median={stats['machine_median']:.2f}, std={stats['machine_std']:.2f}")
    print(f"\nClassification Report:")
    print(results["classification_report"])
    cm = np.array(results["confusion_matrix"])
    print(f"Confusion Matrix:")
    print(f"                  Pred: Human  Pred: Machine")
    print(f"  True: Human     {cm[0,0]:>10}  {cm[0,1]:>13}")
    print(f"  True: Machine   {cm[1,0]:>10}  {cm[1,1]:>13}")
