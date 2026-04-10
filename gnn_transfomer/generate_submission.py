#!/usr/bin/env python3
"""generate_submission.py — Standalone script to generate submission.csv from a trained model.

Usage:
    # Generate submission for Task A using best checkpoint and test file from final_test_files/
    python generate_submission.py --task A --env local

    # Specify custom checkpoint and test file
    python generate_submission.py --task A --checkpoint checkpoints/best_model_taskA.pt \
        --test-file final_test_files/test_taskA.parquet

    # Use a resume checkpoint (auto-extracts model weights)
    python generate_submission.py --task A --checkpoint checkpoints/resume_taskA.pt

    # On Kaggle
    python generate_submission.py --task A --env kaggle \
        --test-file /kaggle/input/test-data/test_taskA.parquet

This script:
    1. Loads the model checkpoint
    2. Loads vocab from cache (or rebuilds from train.parquet)
    3. Loads TF-IDF model from cache (or refits from train.parquet)
    4. Computes CodeBERT embeddings for the test data
    5. Builds graphs and runs inference
    6. Saves submission.csv to submissions/
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch


# ── HuggingFace artifact downloader ──────────────────────────
HF_ARTIFACT_MAP = {
    "A": {
        "checkpoint":  "taskA/best_model_taskA.pt",
        "vocab":       "taskA/global_vocab_v11_medium.json",
        "tfidf":       "taskA/tfidf_model_taskA.joblib",
        "codebert_emb": "taskA/codebert_emb_task_a_all_31fdf2fe.npy",
        "codebert_idx": "taskA/codebert_idx_task_a_all_31fdf2fe.json",
    },
    "B": {
        "checkpoint":  "taskB/best_model_taskB.pt",
        "vocab":       "taskB/global_vocab_v11_medium.json",
        "tfidf":       "taskB/tfidf_model_taskB.joblib",
        "codebert_emb": "taskB/codebert_emb_task_b_all_b2f3a2aa.npy",
        "codebert_idx": "taskB/codebert_idx_task_b_all_b2f3a2aa.json",
    },
    "C": {
        "checkpoint":  "taskC/best_model_taskC.pt",
        "vocab":       "taskC/global_vocab_v11_medium.json",
        "tfidf":       "taskC/tfidf_model_taskC.joblib",
        "codebert_emb": "taskC/codebert_emb_task_c_all_8acb3e2d.npy",
        "codebert_idx": "taskC/codebert_idx_task_c_all_8acb3e2d.json",
    },
}


def download_hf_artifacts(task, cache_dir, checkpoint_dir):
    """Download trained artifacts from HuggingFace if not present locally."""
    from huggingface_hub import hf_hub_download
    import shutil

    repo_id = "dhruv10050/semeval-gnn-models"
    artifacts = HF_ARTIFACT_MAP.get(task)
    if not artifacts:
        print(f"⚠ No HF artifacts mapped for task {task}")
        return

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    mapping = {
        "checkpoint":  (checkpoint_dir, f"best_model_task{task}.pt"),
        "vocab":       (cache_dir, os.path.basename(artifacts["vocab"])),
        "tfidf":       (cache_dir, f"tfidf_model_task{task}.joblib"),
        "codebert_emb": (cache_dir, os.path.basename(artifacts["codebert_emb"])),
        "codebert_idx": (cache_dir, os.path.basename(artifacts["codebert_idx"])),
    }

    for key, (dest_dir, dest_name) in mapping.items():
        dest_path = os.path.join(dest_dir, dest_name)
        if os.path.exists(dest_path):
            continue
        remote = artifacts[key]
        print(f"  ↓ Downloading {remote} → {dest_path} ...")
        try:
            cached = hf_hub_download(repo_id=repo_id, filename=remote)
            shutil.copy2(cached, dest_path)
            print(f"    ✓ {dest_name}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission CSV from a trained model checkpoint")
    parser.add_argument("--task", type=str, required=True,
                        choices=["A", "B", "C"], help="Task to run")
    parser.add_argument("--env", type=str, default="local",
                        choices=["kaggle", "local"], help="Platform")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: checkpoints/best_model_task{TASK}.pt)")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Path to test parquet file (default: final_test_files/test_task{TASK}.parquet)")
    parser.add_argument("--train-file", type=str, default=None,
                        help="Path to train parquet (needed to rebuild vocab/tfidf if cache missing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV filename (default: submission_task{TASK}.csv)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for inference")
    args = parser.parse_args()

    # ── 1. Configure ──────────────────────────────────────────
    from src import config as cfg
    cfg.ENV = args.env
    cfg.ACTIVE_TASK = args.task
    cfg.DEBUG = False
    cfg.init_config()

    from src.utils import seed_everything, mem_report, init_hf
    seed_everything(cfg.SEED)
    init_hf()

    batch_size = args.batch_size or cfg.BATCH_SIZE
    device = cfg.DEVICE

    print(f"\n{'='*60}")
    print(f"  Submission Generator — Task {args.task}")
    print(f"  ENV={args.env} | DEVICE={device}")
    print(f"{'='*60}\n")

    # ── 1b. Download artifacts from HuggingFace if missing ────
    print("Checking for cached artifacts (downloading from HF if needed)...")
    download_hf_artifacts(args.task, cfg.CACHE_DIR, cfg.CHECKPOINT_DIR)

    # ── 2. Load checkpoint ────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(
        cfg.CHECKPOINT_DIR, f"best_model_task{args.task}.pt")
    if not os.path.exists(ckpt_path):
        print(f"✗ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── 3. Load or rebuild vocab ──────────────────────────────
    from src.tree_sitter_utils import init_parsers
    from src.vocab import build_global_vocab, load_vocab

    init_parsers()

    # Try loading cached vocab
    vocab_path = os.path.join(cfg.CACHE_DIR,
                              f"global_vocab_{cfg.GRAPH_LOGIC_VERSION}.json")
    if os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
        print(f"✓ Loaded vocab from cache: {len(vocab)} types")
    else:
        # Rebuild from train data
        train_path = args.train_file or os.path.join(
            cfg.DATA_DIR, f"task_{args.task.lower()}", "train.parquet")
        if not os.path.exists(train_path):
            print(f"✗ Vocab cache not found and no train file at: {train_path}")
            sys.exit(1)
        train_df = pd.read_parquet(train_path)
        print(f"  Building vocab from {len(train_df)} train samples...")
        vocab = build_global_vocab([train_df])

    # ── 4. Load or rebuild TF-IDF ─────────────────────────────
    from src.tfidf_extractor import CodeTFIDFExtractor

    tfidf_path = os.path.join(cfg.CACHE_DIR, f"tfidf_model_task{args.task}.joblib")
    if os.path.exists(tfidf_path):
        tfidf = CodeTFIDFExtractor.load(tfidf_path)
        print(f"✓ Loaded TF-IDF model from cache")
    else:
        train_path = args.train_file or os.path.join(
            cfg.DATA_DIR, f"task_{args.task.lower()}", "train.parquet")
        if not os.path.exists(train_path):
            print(f"✗ TF-IDF cache not found and no train file at: {train_path}")
            sys.exit(1)
        train_df = pd.read_parquet(train_path)
        print(f"  Fitting TF-IDF on {len(train_df)} train samples...")
        tfidf = CodeTFIDFExtractor()
        tfidf.fit_transform(train_df["code"].tolist())
        tfidf.save(tfidf_path)

    # ── 5. CodeBERT embedder ──────────────────────────────────
    from src.codebert_embedder import CodeBERTEmbedder

    embedder = CodeBERTEmbedder(device=device)

    # ── 6. Load test data ─────────────────────────────────────
    test_path = args.test_file or os.path.join(
        "final_test_files", f"test_task{args.task}.parquet")
    if not os.path.exists(test_path):
        # Try alternate locations
        alt = os.path.join(cfg.DATA_DIR, "final_test_files",
                           f"test_task{args.task}.parquet")
        if os.path.exists(alt):
            test_path = alt
        else:
            print(f"✗ Test file not found: {test_path}")
            sys.exit(1)

    test_df = pd.read_parquet(test_path)
    # Normalize column names
    if "ID" in test_df.columns and "id" not in test_df.columns:
        test_df = test_df.rename(columns={"ID": "id"})
    print(f"✓ Loaded test data: {len(test_df)} samples from {test_path}")

    # ── 7. Precompute CodeBERT embeddings for test data ───────
    t0 = time.time()
    embedder.precompute(test_df, split_name=f"task_{args.task.lower()}_test_final")
    print(f"  CodeBERT embeddings computed in {time.time()-t0:.1f}s")
    embedder.unload()  # Free model memory

    # ── 8. Build model and load weights ───────────────────────
    from src.model import SemanticGraphTransformer

    model = SemanticGraphTransformer(
        vocab_size=len(vocab) + 1,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_data, dict) and "model" in ckpt_data:
        model.load_state_dict(ckpt_data["model"])
        print(f"✓ Loaded model from resume checkpoint: {ckpt_path} "
              f"(epoch {ckpt_data.get('epoch', '?')}, "
              f"best F1={ckpt_data.get('best_f1', '?')})")
    else:
        model.load_state_dict(ckpt_data)
        print(f"✓ Loaded checkpoint: {ckpt_path}")

    print(f"  Model: {model.count_parameters():,} parameters")

    # ── 9. Chunked graph building + inference ──────────────────
    from src.inference import chunked_predict, save_submission

    print(f"\nBuilding graphs and running inference in chunks...")
    t0 = time.time()
    preds, ids = chunked_predict(
        model, test_df, embedder, tfidf, vocab, device,
        batch_size=batch_size, chunk_size=25000,
    )
    print(f"\n  Total: {len(preds):,} predictions in {time.time()-t0:.1f}s")

    # ── 10. Save submission ───────────────────────────────────
    output_name = args.output or f"submission_task{args.task}.csv"
    final_sub_dir = os.path.join(cfg.WORK_DIR, "final_submissions")
    save_submission(ids, preds, task=args.task,
                    filename=output_name, output_dir=final_sub_dir)

    mem_report("final")
    print(f"\n✓ Done. Submission saved to: {os.path.join(final_sub_dir, output_name)}")


if __name__ == "__main__":
    main()
