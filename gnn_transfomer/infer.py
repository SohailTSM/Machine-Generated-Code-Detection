#!/usr/bin/env python3
"""infer.py — Standalone inference script.

Loads all artifacts from cache (vocab, TF-IDF, CodeBERT embeddings, model checkpoint).
Does NOT require training data or graph dataset builds.

Usage (Kaggle cell):
    !python infer.py --task A --env kaggle
    !python infer.py --task B --env kaggle --checkpoint /kaggle/working/checkpoints/best_model_taskB.pt
"""

import argparse
import gc
import os
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Standalone inference")
    parser.add_argument("--task", type=str, default="A",
                        choices=["A", "B", "C"])
    parser.add_argument("--env", type=str, default="kaggle",
                        choices=["kaggle", "local"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: auto-detect)")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Inference chunk size (rows per batch)")
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Configure ──────────────────────────────────────────
    from src import config as cfg
    cfg.ENV = args.env
    cfg.ACTIVE_TASK = args.task
    cfg.DEBUG = False
    cfg.init_config()

    from src.utils import seed_everything, mem_report
    seed_everything(cfg.SEED)

    print(f"\n{'='*60}")
    print(f"  Standalone Inference — Task {args.task}")
    print(f"  ENV={args.env} | DEVICE={cfg.DEVICE}")
    print(f"{'='*60}\n")

    # ── 2. Load vocab from cache ─────────────────────────────
    from src.vocab import load_vocab

    vocab = load_vocab()
    print(f"✓ Vocab: {len(vocab)} types")

    # ── 3. Load TF-IDF from cache ─────────────────────────────
    from src.tfidf_extractor import CodeTFIDFExtractor

    tfidf_path = os.path.join(cfg.CACHE_DIR,
                              f"tfidf_model_task{args.task}.joblib")
    if not os.path.exists(tfidf_path):
        print(f"✗ TF-IDF model not found: {tfidf_path}")
        print("  Run training first to generate this artifact.")
        sys.exit(1)
    tfidf = CodeTFIDFExtractor.load(tfidf_path)
    print(f"✓ TF-IDF loaded: {tfidf_path}")

    # ── 4. CodeBERT embedder (lazy — embeds test texts on demand) ─
    from src.codebert_embedder import CodeBERTEmbedder

    embedder = CodeBERTEmbedder(device=cfg.DEVICE)
    # We'll call precompute on test_df so all test node texts get embeddings

    # ── 5. Load model from checkpoint ─────────────────────────
    from src.model import SemanticGraphTransformer

    ckpt_path = args.checkpoint or os.path.join(
        cfg.CHECKPOINT_DIR, f"best_model_task{args.task}.pt")
    if not os.path.exists(ckpt_path):
        print(f"✗ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = SemanticGraphTransformer(
        vocab_size=len(vocab) + 1,
        num_classes=cfg.NUM_CLASSES,
    ).to(cfg.DEVICE)

    ckpt_data = torch.load(ckpt_path, map_location=cfg.DEVICE,
                           weights_only=False)
    if isinstance(ckpt_data, dict) and "model" in ckpt_data:
        model.load_state_dict(ckpt_data["model"])
        print(f"✓ Model loaded (resume ckpt): epoch {ckpt_data.get('epoch', '?')}, "
              f"best F1={ckpt_data.get('best_f1', '?')}")
    else:
        model.load_state_dict(ckpt_data)
        print(f"✓ Model loaded: {ckpt_path}")
    del ckpt_data
    gc.collect()
    print(f"  Parameters: {model.count_parameters():,}")

    # ── 6. Load test data ─────────────────────────────────────
    from src.inference import load_test_data, prepare_test_graphs, \
        run_inference, save_submission
    from torch_geometric.loader import DataLoader as PyGDataLoader

    test_df = load_test_data(args.task)
    print(f"✓ Test data: {len(test_df):,} samples")

    # ── 7. Precompute CodeBERT embeddings for test data ───────
    embedder.precompute(test_df, split_name=f"task_{args.task.lower()}_test_final")
    embedder.unload()  # Free CodeBERT model, keep embeddings (mmap)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_report("post-CodeBERT")

    # ── 8. Chunked inference ──────────────────────────────────
    CHUNK = args.chunk_size
    all_preds = []
    all_ids = []
    n_chunks = (len(test_df) + CHUNK - 1) // CHUNK

    print(f"\nRunning inference: {len(test_df):,} samples in {n_chunks} chunks")

    for ci in range(n_chunks):
        start = ci * CHUNK
        end = min(start + CHUNK, len(test_df))
        chunk_df = test_df.iloc[start:end].reset_index(drop=True)
        print(f"\n  Chunk {ci+1}/{n_chunks}: rows {start:,}–{end-1:,}")

        chunk_graphs = prepare_test_graphs(
            chunk_df, embedder, tfidf, vocab, cfg.DEVICE)

        if len(chunk_graphs) == 0:
            print(f"    ⚠ No valid graphs — skipping")
            continue

        chunk_loader = PyGDataLoader(
            chunk_graphs, batch_size=cfg.BATCH_SIZE, shuffle=False)
        chunk_preds, chunk_ids = run_inference(model, chunk_loader, cfg.DEVICE)
        all_preds.extend(chunk_preds)
        all_ids.extend(chunk_ids)

        del chunk_graphs, chunk_loader, chunk_preds, chunk_ids, chunk_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    ✓ {len(all_preds):,} predictions so far")

    # ── 9. Save submission ────────────────────────────────────
    print(f"\nTotal: {len(all_preds):,} predictions, {len(all_ids):,} IDs")
    final_sub_dir = os.path.join(cfg.WORK_DIR, "final_submissions")
    save_submission(all_ids, all_preds, task=args.task, output_dir=final_sub_dir,
                    test_df=test_df)

    elapsed = time.time() - t0
    print(f"\n✓ Inference complete in {elapsed/60:.1f} minutes")
    mem_report("final")


if __name__ == "__main__":
    main()
