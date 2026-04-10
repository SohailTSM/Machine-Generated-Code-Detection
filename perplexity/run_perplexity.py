#!/usr/bin/env python3
"""
Entry point: Perplexity-based detection pipeline.

Usage:
    python run_perplexity.py --task A --samples 2000
    python run_perplexity.py --task A --samples 5000 --model Salesforce/codegen-350M-mono
"""

import argparse
import os
import sys
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from perplexity_detector import (
    PERPLEXITY_CONFIG,
    load_model,
    compute_perplexities_batch,
    find_optimal_threshold,
    evaluate_perplexity_classifier,
    print_results,
)
from src.data_loader import load_task_data, stratified_subsample
from src.utils import set_seed


def save_json(data: dict, path: str):
    """Save dict to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def run_perplexity_experiment(
    task: str = "A",
    n_train_samples: int = 2000,
    n_val_samples: int = 2000,
    model_name: str = None,
):
    """Run the perplexity-based detection experiment."""
    set_seed(42)
    
    model_name = model_name or PERPLEXITY_CONFIG["model_name"]
    device = PERPLEXITY_CONFIG["device"]
    max_length = PERPLEXITY_CONFIG["max_length"]
    stride = PERPLEXITY_CONFIG["stride"]
    
    print(f"\n{'='*60}")
    print(f" Perplexity-Based Detection — Task {task}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Train samples: {n_train_samples}, Val samples: {n_val_samples}")
    print(f"Device: {device}")
    
    # ---- 1. Load data ----
    print("\n[1/5] Loading data...")
    train_df, val_df = load_task_data(task)
    
    # Subsample for speed
    if n_train_samples < len(train_df):
        train_df = stratified_subsample(train_df, n_train_samples)
    if n_val_samples < len(val_df):
        val_df = stratified_subsample(val_df, n_val_samples)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    train_codes = train_df["code"].tolist()
    train_labels = train_df["label"].values
    val_codes = val_df["code"].tolist()
    val_labels = val_df["label"].values
    
    # ---- 2. Load model ----
    print("\n[2/5] Loading language model...")
    model, tokenizer = load_model(model_name, device)
    
    # ---- 3. Compute perplexities ----
    print("\n[3/5] Computing perplexities on training set...")
    train_ppl = compute_perplexities_batch(
        train_codes, model, tokenizer, max_length, stride, device,
        desc="Train perplexity"
    )
    
    print("\n[4/5] Computing perplexities on validation set...")
    val_ppl = compute_perplexities_batch(
        val_codes, model, tokenizer, max_length, stride, device,
        desc="Val perplexity"
    )
    
    # ---- 4. Find optimal threshold on train ----
    print("\n[5/5] Finding optimal threshold...")
    threshold, threshold_stats = find_optimal_threshold(
        train_ppl, train_labels, method="youden"
    )
    
    print(f"\nThreshold Statistics:")
    print(f"  ROC-AUC: {threshold_stats['roc_auc']:.4f}")
    print(f"  Optimal threshold: {threshold:.2f}")
    print(f"  TPR at threshold: {threshold_stats['tpr_at_threshold']:.4f}")
    print(f"  FPR at threshold: {threshold_stats['fpr_at_threshold']:.4f}")
    
    # ---- 5. Evaluate on train and val ----
    train_results = evaluate_perplexity_classifier(
        train_ppl, train_labels, threshold
    )
    print_results(train_results, "Training Set")
    
    val_results = evaluate_perplexity_classifier(
        val_ppl, val_labels, threshold
    )
    print_results(val_results, "Validation Set")
    
    # ---- 6. Save results ----
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "perplexity", f"perplexity_{task}")
    os.makedirs(output_dir, exist_ok=True)
    
    final_results = {
        "model_name": model_name,
        "n_train_samples": n_train_samples,
        "n_val_samples": n_val_samples,
        "threshold": float(threshold),
        "threshold_stats": threshold_stats,
        "train_macro_f1": train_results["macro_f1"],
        "val_macro_f1": val_results["macro_f1"],
        "val_accuracy": val_results["accuracy"],
        "val_classification_report": val_results["classification_report"],
        "val_confusion_matrix": val_results["confusion_matrix"],
        "perplexity_stats_train": train_results["perplexity_stats"],
        "perplexity_stats_val": val_results["perplexity_stats"],
    }
    
    results_path = os.path.join(output_dir, "eval_results.json")
    save_json(final_results, results_path)
    
    # Save raw perplexities for analysis
    np.savez(
        os.path.join(output_dir, "perplexities.npz"),
        train_ppl=train_ppl,
        train_labels=train_labels,
        val_ppl=val_ppl,
        val_labels=val_labels,
    )
    print(f"Saved: {os.path.join(output_dir, 'perplexities.npz')}")
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Optimal Threshold: {threshold:.2f}")
    print(f"Train Macro F1: {train_results['macro_f1']:.4f}")
    print(f"Val Macro F1:   {val_results['macro_f1']:.4f}")
    print(f"Val Accuracy:   {val_results['accuracy']:.4f}")
    print(f"\nPerplexity Separation (Validation):")
    vs = val_results["perplexity_stats"]
    print(f"  Human mean:   {vs['human_mean']:.2f}")
    print(f"  Machine mean: {vs['machine_mean']:.2f}")
    print(f"  Difference:   {vs['human_mean'] - vs['machine_mean']:.2f}")
    
    return val_results


def main():
    parser = argparse.ArgumentParser(description="Perplexity-based detection")
    parser.add_argument("--task", default="A", choices=["A", "B", "C"])
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of samples per split (train/val)")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name for perplexity")
    args = parser.parse_args()
    
    run_perplexity_experiment(
        task=args.task,
        n_train_samples=args.samples,
        n_val_samples=args.samples,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
