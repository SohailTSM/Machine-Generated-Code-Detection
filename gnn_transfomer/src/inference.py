"""Inference — test-set prediction and submission CSV generation."""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config as cfg
from src.tree_sitter_utils import detect_language_fast, init_parsers, parse_code
from src.graph_builder import ast_to_graph_v7
from src.codebert_embedder import CodeBERTEmbedder
from src.tfidf_extractor import CodeTFIDFExtractor
from src.vocab import load_vocab, encode_node_types
from src.dataset import _build_single_graph, ChunkedGraphDataset, create_dataloaders


def load_test_data(task="A", test_file=None):
    """Load the test parquet for the given task.

    Args:
        task: Task identifier (A, B, or C)
        test_file: Optional explicit path to test file

    Tries explicit path → final_test_files → local task dir → kaggle input path.
    Returns DataFrame with columns: id, code, (optionally language).
    """
    task_cfg = cfg.TASK_CONFIG[task]
    candidates = []

    if test_file:
        candidates.append(test_file)

    candidates.extend([
        os.path.join(cfg.WORK_DIR, "final_test_files", f"test_task{task}.parquet"),
        os.path.join(cfg.DATA_DIR, "final_test_files", f"test_task{task}.parquet"),
        os.path.join("final_test_files", f"test_task{task}.parquet"),
        os.path.join(cfg.DATA_DIR, f"task_{task.lower()}", f"task_{task.lower()}_test.parquet"),
        os.path.join(cfg.DATA_DIR, "task_a", "task_a_test.parquet"),
    ])
    if cfg.ENV == "kaggle":
        candidates.insert(0, f"/kaggle/input/semeval-2026-task13/task_{task.lower()}/task_{task.lower()}_test.parquet")

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Ensure ID column exists (uppercase)
            if "id" in df.columns and "ID" not in df.columns:
                df = df.rename(columns={"id": "ID"})
            print(f"✓ Loaded test data: {len(df)} samples from {path}")
            return df

    raise FileNotFoundError(f"Test file not found. Searched: {candidates}")


def prepare_test_graphs(df, embedder, tfidf_extractor, vocab, device):
    """Build graph dataset for test data (no labels).

    Returns ChunkedGraphDataset and DataLoader.
    """
    from torch_geometric.data import Data

    parsers = init_parsers()
    graphs = []
    skipped = 0
    total = len(df)

    for idx, row in tqdm(df.iterrows(), total=total, desc="  Building test graphs",
                         miniters=max(1, total // 50)):
        code = row["code"]
        lang = row.get("language", None)
        if lang is None:
            lang = detect_language_fast(code)

        tree = parse_code(code, lang)
        if tree is None:
            skipped += 1
            continue

        g = ast_to_graph_v7(tree, vocab)
        if g is None or g["num_nodes"] == 0:
            skipped += 1
            continue

        # Encode node types
        type_ids = encode_node_types(g["type_ids"], vocab)

        # Semantic embeddings — lookup expects a list of texts
        sem_embs = embedder.lookup(g["node_texts"])

        # TF-IDF
        tfidf_vec = tfidf_extractor.transform([code])
        if hasattr(tfidf_vec, 'toarray'):
            tfidf_vec = tfidf_vec.toarray()

        data = Data(
            x_type=torch.as_tensor(type_ids, dtype=torch.long),
            x_cont=torch.as_tensor(g["cont_features"], dtype=torch.float32),
            x_semantic=torch.as_tensor(sem_embs, dtype=torch.float32),
            edge_index=torch.as_tensor(g["edge_index"], dtype=torch.long),
            edge_type=torch.as_tensor(g["edge_type"], dtype=torch.long),
            tfidf=torch.as_tensor(tfidf_vec, dtype=torch.float32),
            orig_id=row.get("ID", idx),
            num_nodes=g["num_nodes"],
        )
        graphs.append(data)

    if skipped > 0:
        print(f"⚠ Skipped {skipped}/{len(df)} test samples (parse failures)")

    return graphs


@torch.no_grad()
def chunked_predict(model, df, embedder, tfidf_extractor, vocab, device,
                    batch_size=16, chunk_size=25000):
    """Build graphs and run inference in memory-efficient chunks.

    Instead of building all graphs in memory at once, processes `chunk_size`
    samples at a time: build graphs → infer → free → next chunk.

    Returns (all_preds, all_ids).
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import gc

    model.eval()
    all_preds = []
    all_ids = []
    total = len(df)
    n_chunks = (total + chunk_size - 1) // chunk_size

    use_amp = cfg.USE_AMP
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, total)
        chunk_df = df.iloc[start:end]
        print(f"\n  Chunk {ci+1}/{n_chunks}: samples {start:,}–{end-1:,}")

        # Build graphs for this chunk
        graphs = prepare_test_graphs(chunk_df, embedder, tfidf_extractor,
                                     vocab, device)
        if not graphs:
            print(f"    ⚠ 0 graphs built for this chunk, skipping")
            continue

        loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)

        # Run inference on this chunk
        for batch in tqdm(loader, desc=f"    Inference chunk {ci+1}",
                          miniters=max(1, len(loader) // 20)):
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                logits = model(
                    batch.x_type, batch.x_cont, batch.x_semantic,
                    batch.edge_index, batch.edge_type,
                    batch.batch, batch.tfidf,
                )
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            if hasattr(batch, 'orig_id'):
                ids = batch.orig_id
                if isinstance(ids, torch.Tensor):
                    all_ids.extend(ids.cpu().tolist())
                elif isinstance(ids, list):
                    all_ids.extend(ids)
                else:
                    all_ids.extend([ids])

        # Free chunk memory
        del graphs, loader
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"    ✓ {len(all_preds):,} predictions so far")

    return all_preds, all_ids


@torch.no_grad()
def run_inference(model, loader, device, use_amp=None):
    """Run model inference on a DataLoader.

    Returns (preds, ids) — lists of predicted class indices and original IDs.
    """
    if use_amp is None:
        use_amp = cfg.USE_AMP

    model.eval()
    all_preds = []
    all_ids = []

    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    total_batches = len(loader)

    for batch in tqdm(loader, total=total_batches, desc="  Inference",
                      miniters=max(1, total_batches // 50)):
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            logits = model(
                batch.x_type, batch.x_cont, batch.x_semantic,
                batch.edge_index, batch.edge_type,
                batch.batch, batch.tfidf,
            )
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        # Collect IDs stored in each graph's orig_id
        if hasattr(batch, 'orig_id'):
            ids = batch.orig_id
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.cpu().tolist())
            elif isinstance(ids, list):
                all_ids.extend(ids)
            else:
                all_ids.extend([ids])

    return all_preds, all_ids


def save_submission(ids, preds, task="A", filename=None, output_dir=None,
                    test_df=None):
    """Save predictions as a competition submission CSV.

    Columns: ID, label (numeric class indices — matches competition format).
    If test_df is provided, validates that submission IDs exactly match test IDs.
    """
    df = pd.DataFrame({"ID": ids, "label": preds})
    # Ensure IDs are int (not float)
    df["ID"] = df["ID"].astype(int)
    df["label"] = df["label"].astype(int)

    if filename is None:
        filename = f"submission_task{task}.csv"
    out = output_dir or cfg.SUBMISSION_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename)
    df.to_csv(path, index=False)
    print(f"✓ Submission saved: {path} ({len(df)} rows)")

    # Validate format
    assert df.columns.tolist() == ["ID", "label"], \
        f"Bad columns: {df.columns.tolist()}"
    assert not df["ID"].duplicated().any(), "Duplicate IDs in submission"
    assert not df["label"].isna().any(), "NaN predictions in submission"

    # Validate IDs against test file if provided
    if test_df is not None:
        id_col = "ID" if "ID" in test_df.columns else "id"
        if id_col in test_df.columns:
            expected_ids = set(test_df[id_col].astype(int).tolist())
            actual_ids = set(df["ID"].tolist())
            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids
            if missing:
                print(f"  ⚠ {len(missing)} test IDs missing from submission!")
            if extra:
                print(f"  ⚠ {len(extra)} extra IDs in submission not in test file!")
            if not missing and not extra:
                print(f"  ✓ All {len(expected_ids)} test IDs present, no extras")

    # Label distribution
    task_cfg = cfg.TASK_CONFIG[task]
    idx2label = task_cfg["idx2label"]
    named_dist = {idx2label.get(int(k), k): v
                  for k, v in df["label"].value_counts().to_dict().items()}
    print(f"  Label distribution: {named_dist}")

    return df


def validate_submission(submission_path, expected_count=None):
    """Validate a submission CSV file."""
    df = pd.read_csv(submission_path)
    errors = []

    if list(df.columns) != ["ID", "label"]:
        errors.append(f"Wrong columns: {list(df.columns)}")
    if df["ID"].duplicated().any():
        errors.append(f"{df['ID'].duplicated().sum()} duplicate IDs")
    if df["label"].isna().any():
        errors.append(f"{df['label'].isna().sum()} NaN predictions")
    if expected_count and len(df) != expected_count:
        errors.append(f"Expected {expected_count} rows, got {len(df)}")

    if errors:
        raise ValueError(f"Submission validation failed: {'; '.join(errors)}")

    print(f"✓ Submission OK: {len(df)} rows, {df['prediction'].nunique()} unique labels")
    return df
