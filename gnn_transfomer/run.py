#!/usr/bin/env python3
"""run.py — CLI entry point for the SemanticGraphTransformer pipeline."""

import argparse
import gc
import json
import os
import sys
import time

# GPU memory safety: reduce fragmentation on CUDA devices
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


def _cleanup_disk(work_dir: str, task: str, keep_embeddings=True):
    """Delete bulky intermediate files to free disk before HF upload.

    Removes: graph chunk dirs, cached parquets, resume checkpoint.
    Keeps: best_model checkpoint, embeddings (.npy/.json), vocab, tfidf, submissions.
    """
    import shutil
    import glob

    freed = 0
    cache_dir = os.path.join(work_dir, "pt_chunks")

    # Delete graph chunk directories (largest disk consumer)
    for pattern in [f"task_{task.lower()}_train_*", f"task_{task.lower()}_val_*",
                    f"task_{task.lower()}_test_*"]:
        for d in glob.glob(os.path.join(cache_dir, pattern)):
            if os.path.isdir(d):
                size = sum(os.path.getsize(os.path.join(dp, f))
                           for dp, _, fns in os.walk(d) for f in fns)
                shutil.rmtree(d, ignore_errors=True)
                freed += size

    # Delete resume checkpoint (large — best_model is sufficient)
    resume = os.path.join(work_dir, "checkpoints", f"resume_task{task}.pt")
    if os.path.exists(resume):
        freed += os.path.getsize(resume)
        os.remove(resume)

    # Delete cached parquets
    parq_dir = os.path.join(work_dir, f"task_{task.lower()}")
    if os.path.isdir(parq_dir):
        for f in os.listdir(parq_dir):
            fp = os.path.join(parq_dir, f)
            if f.endswith(".parquet"):
                freed += os.path.getsize(fp)
                os.remove(fp)

    if freed > 0:
        print(f"  ♻ Disk cleanup freed {freed / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="SemEval-2026 Task 13 — SemanticGraphTransformer")
    parser.add_argument("--task", type=str, default="A",
                        choices=["A", "B", "C"], help="Task to run")
    parser.add_argument("--env", type=str, default="local",
                        choices=["kaggle", "local"], help="Platform")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (small subsample)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training, run inference on test set")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for inference")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last resume checkpoint")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to resume checkpoint (default: checkpoints/resume_task{TASK}.pt)")
    args = parser.parse_args()

    # ── 1. Configure ──────────────────────────────────────────
    from src import config as cfg
    cfg.ENV = args.env
    cfg.ACTIVE_TASK = args.task
    cfg.DEBUG = args.debug
    cfg.init_config()

    from src.utils import seed_everything, mem_report, disk_report, init_wandb, init_hf, Timer

    seed_everything(cfg.SEED)
    init_hf()
    disk_report(tag="startup")
    print(f"\n{'='*60}")
    print(f"  SemanticGraphTransformer v10_compact — Task {args.task}")
    print(f"  ENV={args.env} | DEBUG={args.debug} | DEVICE={cfg.DEVICE}")
    print(f"{'='*60}\n")

    # ── 2. Load Data ──────────────────────────────────────────
    from src.data_loader import load_task_data, get_class_weights, validate_labels

    with Timer("Data loading"):
        train_df, val_df = load_task_data()
        validate_labels(train_df, split="train")
        if val_df is not None:
            validate_labels(val_df, split="val")

    # Delete cached parquets from working dir to free disk (data is in RAM now)
    _cached_parquet_dir = os.path.join(cfg.WORK_DIR, f"task_{args.task.lower()}")
    if os.path.isdir(_cached_parquet_dir):
        import shutil
        shutil.rmtree(_cached_parquet_dir, ignore_errors=True)
        print(f"  ♻ Freed disk: deleted cached parquets in {_cached_parquet_dir}")
    disk_report(tag="after data load")

    # ── 3. Parsers & Vocab ────────────────────────────────────
    from src.tree_sitter_utils import init_parsers
    from src.vocab import build_global_vocab, load_vocab

    with Timer("Parsers & vocab"):
        init_parsers()
        vocab = build_global_vocab([train_df])

    # ── 4. CodeBERT Embeddings ────────────────────────────────
    import pandas as pd
    from src.codebert_embedder import CodeBERTEmbedder

    with Timer("CodeBERT precompute"):
        embedder = CodeBERTEmbedder(device=cfg.DEVICE)
        # Combine train+val so all unique node texts share one embedding table
        all_df = pd.concat(
            [train_df] + ([val_df] if val_df is not None else []),
            ignore_index=True,
        )
        embedder.precompute(all_df, split_name=f"task_{args.task.lower()}_all")
        # Free CodeBERT model from memory
        embedder.unload()
        # Free the concatenated DF immediately (train_df/val_df still alive)
        del all_df
        gc.collect()

    disk_report(tag="after CodeBERT")

    # ── 5. TF-IDF Features ───────────────────────────────────
    from src.tfidf_extractor import CodeTFIDFExtractor

    with Timer("TF-IDF"):
        tfidf_path = os.path.join(cfg.CACHE_DIR, f"tfidf_model_task{args.task}.joblib")
        if os.path.exists(tfidf_path) and not args.debug:
            tfidf = CodeTFIDFExtractor.load(tfidf_path)
            train_tfidf = tfidf.transform(train_df["code"].tolist())
        else:
            tfidf = CodeTFIDFExtractor()
            train_tfidf = tfidf.fit_transform(train_df["code"].tolist())
            tfidf.save(tfidf_path)

        val_tfidf = None
        if val_df is not None:
            val_tfidf = tfidf.transform(val_df["code"].tolist())

    # ── 6. Build Graph Datasets ───────────────────────────────
    from src.augmentation import CodeAugmenter
    from src.dataset import ChunkedGraphDataset, create_dataloaders, clear_chunk_cache
    from torch_geometric.loader import DataLoader as PyGDataLoader

    with Timer("Graph datasets"):
        augmenter = CodeAugmenter() if cfg.AUGMENT_RATIO > 0 else None

        train_ds = ChunkedGraphDataset(
            train_df, vocab, embedder, train_tfidf,
            split_name=f"task_{args.task.lower()}_train",
            augmenter=augmenter,
            augment_ratio=cfg.AUGMENT_RATIO,
            embeddings=embedder.embeddings,
        )

        val_ds = None
        if val_df is not None:
            val_ds = ChunkedGraphDataset(
                val_df, vocab, embedder, val_tfidf,
                split_name=f"task_{args.task.lower()}_val",
                embeddings=embedder.embeddings,
            )

        train_loader, val_loader = create_dataloaders(train_ds, val_ds)

    # Free TF-IDF dense/sparse arrays (data is now in chunks on disk)
    del train_tfidf
    if val_tfidf is not None:
        del val_tfidf
    gc.collect()

    mem_report()
    disk_report(tag="after graph build")

    # ── 7. Model ──────────────────────────────────────────────
    from src.model import SemanticGraphTransformer

    model = SemanticGraphTransformer(
        vocab_size=len(vocab) + 1,  # +1 for padding idx 0
        num_classes=cfg.NUM_CLASSES,
    ).to(cfg.DEVICE)

    print(f"✓ Model: {model.count_parameters():,} trainable parameters")
    # NOTE: 2nd T4 GPU is available but standard DataParallel doesn't work with
    # PyG graph batches (can't split variable-size node/edge tensors along dim=0).
    # Using single GPU; 2nd GPU can run a separate task in parallel.

    # ── 8. Train or Infer ─────────────────────────────────────
    if args.inference_only:
        # Load checkpoint (supports both best_model and resume checkpoint formats)
        ckpt_path = args.checkpoint or os.path.join(
            cfg.CHECKPOINT_DIR, f"best_model_task{args.task}.pt")
        if not os.path.exists(ckpt_path):
            print(f"✗ Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        ckpt_data = torch.load(ckpt_path, map_location=cfg.DEVICE,
                               weights_only=False)
        if isinstance(ckpt_data, dict) and "model" in ckpt_data:
            model.load_state_dict(ckpt_data["model"])
            print(f"✓ Loaded model from resume checkpoint: {ckpt_path} "
                  f"(epoch {ckpt_data.get('epoch', '?')}, "
                  f"best F1={ckpt_data.get('best_f1', '?')})")
        else:
            model.load_state_dict(ckpt_data)
            print(f"✓ Loaded checkpoint: {ckpt_path}")

        # Run inference on test set (chunked for memory safety)
        from src.inference import load_test_data, prepare_test_graphs, \
            run_inference, save_submission
        from torch_geometric.loader import DataLoader as PyGDataLoader

        # Free training data (not needed for inference-only)
        del train_df, train_ds, train_loader
        if val_df is not None:
            del val_df
        if val_ds is not None:
            del val_ds
        del val_loader
        clear_chunk_cache()
        gc.collect()

        test_df = load_test_data(args.task)
        INFERENCE_CHUNK = 5000
        all_preds = []
        all_ids = []
        n_chunks = (len(test_df) + INFERENCE_CHUNK - 1) // INFERENCE_CHUNK
        for ci in range(n_chunks):
            start = ci * INFERENCE_CHUNK
            end = min(start + INFERENCE_CHUNK, len(test_df))
            chunk_df = test_df.iloc[start:end].reset_index(drop=True)
            chunk_graphs = prepare_test_graphs(
                chunk_df, embedder, tfidf, vocab, cfg.DEVICE)
            if len(chunk_graphs) == 0:
                continue
            chunk_loader = PyGDataLoader(
                chunk_graphs, batch_size=cfg.BATCH_SIZE, shuffle=False)
            chunk_preds, chunk_ids = run_inference(model, chunk_loader, cfg.DEVICE)
            all_preds.extend(chunk_preds)
            all_ids.extend(chunk_ids)
            del chunk_graphs, chunk_loader, chunk_df
            gc.collect()
        save_submission(all_ids, all_preds, task=args.task, test_df=test_df)

    else:
        # WandB init
        if not args.no_wandb:
            init_wandb()

        # Class weights
        class_weights = get_class_weights(train_df, cfg.NUM_CLASSES)

        # Free DataFrames early — data is in disk chunks, only class_weights needed
        train_df = None
        val_df = None
        gc.collect()
        mem_report("freed DFs before training")

        # Train
        from src.training import train_model, eval_epoch
        from src.visualization import (plot_training_curves,
                                        plot_confusion_matrix,
                                        plot_per_class_f1)
        from src.losses import FocalLoss

        # Determine resume path
        resume_path = None
        if args.resume or args.resume_from:
            resume_path = args.resume_from or os.path.join(
                cfg.CHECKPOINT_DIR, f"resume_task{args.task}.pt")

        history = train_model(model, train_loader, val_loader,
                              class_weights, cfg.DEVICE,
                              resume_path=resume_path)

        # Save history
        hist_path = os.path.join(cfg.LOG_DIR, f"training_history_task{args.task}.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        # Plots
        plot_training_curves(history)
        criterion = FocalLoss(gamma=cfg.FOCAL_GAMMA)
        if val_loader is not None:
            val_final = eval_epoch(model, val_loader, criterion,
                                   cfg.DEVICE, cfg.USE_AMP)
            plot_confusion_matrix(val_final["labels"], val_final["preds"])
            plot_per_class_f1(val_final["labels"], val_final["preds"])

        # ── Free training-phase memory before inference ───────
        del train_df, train_loader, train_ds, class_weights
        if val_df is not None:
            del val_df
        if val_loader is not None:
            del val_loader
        if val_ds is not None:
            del val_ds
        clear_chunk_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem_report("post-training cleanup")

        # ── 9. Test Sample Evaluation ─────────────────────────
        from src.data_loader import load_test_sample

        test_sample_df = load_test_sample(args.task)
        if test_sample_df is not None:
            print(f"\n{'='*60}")
            print(f"  Evaluating on test_sample (Task {args.task})")
            print(f"{'='*60}")

            test_sample_tfidf = tfidf.transform(test_sample_df["code"].tolist())
            test_sample_ds = ChunkedGraphDataset(
                test_sample_df, vocab, embedder, test_sample_tfidf,
                split_name=f"task_{args.task.lower()}_test_sample",
                embeddings=embedder.embeddings,
            )
            test_sample_loader = PyGDataLoader(
                test_sample_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                num_workers=cfg.NUM_WORKERS)

            ts_result = eval_epoch(model, test_sample_loader, criterion,
                                   cfg.DEVICE, cfg.USE_AMP)
            from sklearn.metrics import classification_report
            report = classification_report(
                ts_result["labels"], ts_result["preds"],
                target_names=cfg.LABEL_NAMES, zero_division=0)
            print(f"\nTest Sample Results — Task {args.task}:")
            print(f"  Macro F1:      {ts_result['f1']:.4f}")
            print(f"  Balanced Acc:  {ts_result['bal_acc']:.4f}")
            print(f"  Loss:          {ts_result['loss']:.4f}")
            print(f"\n{report}")

            # Free test_sample memory
            del test_sample_ds, test_sample_loader, test_sample_df, test_sample_tfidf
            clear_chunk_cache()
            gc.collect()

        # ── 10. Inference on final test files (chunked for memory) ─
        from src.inference import load_test_data, prepare_test_graphs, \
            run_inference, save_submission

        try:
            test_df = load_test_data(args.task)
            print(f"\n{'='*60}")
            print(f"  Running inference on final test set (Task {args.task})")
            print(f"  {len(test_df)} samples — chunked inference")
            print(f"{'='*60}")

            # Precompute CodeBERT embeddings for test data if needed
            if embedder.text_to_idx is None or embedder.embeddings is None:
                print("  Reloading CodeBERT embeddings for test data...")
                embedder.precompute(test_df, split_name=f"task_{args.task.lower()}_test_final")
                embedder.unload()

            INFERENCE_CHUNK = 5000
            all_preds = []
            all_ids = []
            n_chunks = (len(test_df) + INFERENCE_CHUNK - 1) // INFERENCE_CHUNK

            for ci in range(n_chunks):
                start = ci * INFERENCE_CHUNK
                end = min(start + INFERENCE_CHUNK, len(test_df))
                chunk_df = test_df.iloc[start:end].reset_index(drop=True)
                print(f"\n  Chunk {ci+1}/{n_chunks}: rows {start:,}–{end-1:,}")

                chunk_graphs = prepare_test_graphs(
                    chunk_df, embedder, tfidf, vocab, cfg.DEVICE)

                if len(chunk_graphs) == 0:
                    print(f"    ⚠ No valid graphs — skipping chunk")
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

            print(f"\n  Total: {len(all_preds):,} predictions, {len(all_ids):,} IDs")
            final_sub_dir = os.path.join(cfg.WORK_DIR, "final_submissions")
            save_submission(all_ids, all_preds, task=args.task,
                           output_dir=final_sub_dir, test_df=test_df)
        except FileNotFoundError as e:
            print(f"\n⚠ No final test file found for Task {args.task} — skipping submission generation")
            print(f"  Detail: {e}")
        except Exception as e:
            print(f"\n⚠ Final inference failed for Task {args.task}: {e}")
            import traceback
            traceback.print_exc()

        # ── 11. Push model + inference artifacts to HuggingFace ──
        # Clean up bulky intermediate files to free disk before upload
        _cleanup_disk(cfg.WORK_DIR, args.task)
        disk_report(tag="pre-HF-push")

        # Free GPU memory before HF push (uploads don't need GPU)
        model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem_report("pre-HF-push cleanup")

        if cfg.HF_WRITE_TOKEN and cfg.HF_MODEL_REPO:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=cfg.HF_WRITE_TOKEN)
                api.create_repo(cfg.HF_MODEL_REPO, exist_ok=True, private=False)

                # Collect all files to upload: (local_path, repo_path)
                uploads = []

                # Best model checkpoint
                best_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"best_model_task{args.task}.pt")
                if os.path.exists(best_ckpt):
                    uploads.append((best_ckpt, f"task{args.task}/best_model_task{args.task}.pt"))

                # Resume checkpoint
                resume_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"resume_task{args.task}.pt")
                if os.path.exists(resume_ckpt):
                    uploads.append((resume_ckpt, f"task{args.task}/resume_task{args.task}.pt"))

                # Vocabulary JSON
                import glob
                for vf in glob.glob(os.path.join(cfg.CACHE_DIR, f"global_vocab_{cfg.GRAPH_LOGIC_VERSION}.json")):
                    uploads.append((vf, f"task{args.task}/{os.path.basename(vf)}"))

                # TF-IDF model
                tfidf_file = os.path.join(cfg.CACHE_DIR, f"tfidf_model_task{args.task}.joblib")
                if os.path.exists(tfidf_file):
                    uploads.append((tfidf_file, f"task{args.task}/tfidf_model_task{args.task}.joblib"))

                # CodeBERT embedding cache (.npy + .json index)
                for ext in ["npy", "json"]:
                    for f in glob.glob(os.path.join(cfg.CACHE_DIR, f"codebert_*_task_{args.task.lower()}_*.{ext}")):
                        uploads.append((f, f"task{args.task}/{os.path.basename(f)}"))

                # Submission CSV
                final_sub = os.path.join(cfg.WORK_DIR, "final_submissions", f"submission_task{args.task}.csv")
                if os.path.exists(final_sub):
                    uploads.append((final_sub, f"task{args.task}/submission_task{args.task}.csv"))

                # Training history
                hist_file = os.path.join(cfg.LOG_DIR, f"training_history_task{args.task}.json")
                if os.path.exists(hist_file):
                    uploads.append((hist_file, f"task{args.task}/training_history_task{args.task}.json"))

                # Training plots
                for plot in ["training_curves", "confusion_matrix", "per_class_f1"]:
                    pf = os.path.join(cfg.LOG_DIR, f"{plot}_task{args.task}.png")
                    if os.path.exists(pf):
                        uploads.append((pf, f"task{args.task}/{plot}_task{args.task}.png"))

                # Upload all — skip empty files
                print(f"\nPushing {len(uploads)} files to HF: {cfg.HF_MODEL_REPO}")
                for local_path, repo_path in uploads:
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    if os.path.getsize(local_path) == 0:
                        print(f"  ⚠ SKIP {repo_path} (empty file!)")
                        continue
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=cfg.HF_MODEL_REPO,
                    )
                    print(f"  ✓ {repo_path} ({size_mb:.1f} MB)")

                print(f"✓ All artifacts pushed to HF: {cfg.HF_MODEL_REPO}")

            except Exception as e:
                print(f"⚠ HF push failed: {e}")

    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()
