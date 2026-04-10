"""ChunkedGraphDataset — disk-cached PyG dataset with streaming augmentation."""

import gc
import hashlib
import json
import os
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from src import config as cfg
from src.augmentation import CodeAugmenter
from src.codebert_embedder import CodeBERTEmbedder
from src.graph_builder import ast_to_graph_v7
from src.tree_sitter_utils import parse_code


@lru_cache(maxsize=32)
def _load_chunk(path: str) -> list:
    """LRU-cached chunk loader (maxsize=32 to keep chunks in RAM)."""
    return torch.load(path, weights_only=False)


def clear_chunk_cache():
    """Clear the LRU cache for loaded chunks — call between phases."""
    _load_chunk.cache_clear()


def _build_single_graph(code: str, language: str, label, orig_id: int,
                         vocab: dict, codebert: CodeBERTEmbedder,
                         tfidf_vec, max_nodes: int) -> Data | None:
    """Parse one code snippet into a PyG Data object.

    Stores x_semantic_idx (embedding indices) instead of full 768-d vectors
    to dramatically reduce chunk disk size and memory footprint.
    """
    if not isinstance(code, str) or not code.strip():
        return None

    root = parse_code(code, language)
    if root is None:
        return None

    g = ast_to_graph_v7(root, vocab, max_nodes)
    if g is None or g["num_nodes"] == 0:
        return None

    # CodeBERT: store embedding INDICES instead of full 768-d vectors
    # Reconstruction happens at __getitem__ time via shared mmap'd embeddings
    if codebert is not None and codebert.text_to_idx is not None:
        indices = [codebert.text_to_idx.get(t, 0) for t in g["node_texts"]]
        x_semantic_idx = torch.tensor(indices, dtype=torch.long)
    else:
        x_semantic_idx = torch.zeros(g["num_nodes"], dtype=torch.long)

    # TF-IDF (handle both sparse CSR and dense arrays)
    if tfidf_vec is not None:
        if hasattr(tfidf_vec, 'toarray'):  # scipy sparse row
            tv = tfidf_vec.toarray().flatten()
        else:
            tv = np.asarray(tfidf_vec).flatten()
        tfidf = torch.tensor(tv, dtype=torch.float)
    else:
        tfidf = torch.zeros(cfg.TFIDF_DIM, dtype=torch.float)

    # Safe label handling
    if pd.isna(label):
        safe_label = 0
    else:
        safe_label = max(int(label), 0)

    return Data(
        x_type=g["type_ids"],
        x_cont=g["cont_features"],
        x_semantic_idx=x_semantic_idx,
        edge_index=g["edge_index"],
        edge_type=g["edge_type"],
        tfidf=tfidf,
        num_nodes=g["num_nodes"],
        y=torch.tensor(safe_label, dtype=torch.long),
        orig_id=torch.tensor(int(orig_id), dtype=torch.long),
    )


class ChunkedGraphDataset(torch.utils.data.Dataset):
    """Disk-cached chunked graph dataset with streaming augmentation."""

    def __init__(self, df: pd.DataFrame, vocab: dict,
                 codebert: CodeBERTEmbedder,
                 tfidf_features,
                 cache_dir: str = None,
                 max_nodes: int = None,
                 split_name: str = "data",
                 chunk_size: int = None,
                 augmenter: CodeAugmenter = None,
                 augment_ratio: float = 0.0,
                 embeddings: np.ndarray = None):
        super().__init__()
        self.chunk_size = chunk_size or cfg.CHUNK_SIZE
        cache_dir = cache_dir or cfg.CACHE_DIR
        max_nodes = max_nodes or cfg.MAX_NODES

        # Shared embeddings for x_semantic reconstruction in __getitem__
        # Should be mmap'd numpy array for efficient cross-worker sharing
        self._embeddings = embeddings

        # Cache key includes version + config hash
        v_sample = str(sorted(list(vocab.items()))[:50])
        config_str = (f"{cfg.GRAPH_LOGIC_VERSION}_{max_nodes}_{self.chunk_size}_"
                      f"{len(df)}_{v_sample}")
        c_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.cache_path = os.path.join(cache_dir, f"{split_name}_{c_hash}")
        os.makedirs(self.cache_path, exist_ok=True)

        self.meta_file = os.path.join(self.cache_path, "metadata.json")
        self.chunk_files = []
        self.chunk_lengths = []

        if os.path.exists(self.meta_file):
            print(f"✓ Loading {split_name} from cache [{self.cache_path}]")
            with open(self.meta_file) as f:
                meta = json.load(f)
            self.chunk_files = [os.path.join(self.cache_path, p)
                                for p in meta["chunk_paths"]]
            self.chunk_lengths = meta["chunk_lengths"]
        else:
            self._process(df, vocab, codebert, tfidf_features, max_nodes,
                          augmenter, augment_ratio)

        self.total_samples = sum(self.chunk_lengths)
        self.cum_lens = np.cumsum([0] + self.chunk_lengths)
        print(f"  {split_name}: {self.total_samples:,} graphs in "
              f"{len(self.chunk_files)} chunk(s)")

    def _process(self, df, vocab, codebert, tfidf_features, max_nodes,
                 augmenter, augment_ratio):
        """Build PyG Data objects, with streaming augmentation."""
        total = len(df)
        print(f"Processing {total:,} samples → {self.cache_path}")

        valid_buf = []
        skipped = 0

        for i in tqdm(range(total), desc="  Graphs"):
            row = df.iloc[i]
            tfidf_vec = tfidf_features[i] if tfidf_features is not None else None

            data = _build_single_graph(
                row["code"], row.get("language", ""), row.get("label", 0),
                row["orig_id"], vocab, codebert, tfidf_vec, max_nodes
            )
            if data is not None:
                valid_buf.append(data)

                # Streaming augmentation (M2 from robustness plan)
                if augmenter is not None and random.random() < augment_ratio:
                    aug_code = augmenter.augment(
                        row["code"], row.get("language", "Python"))
                    aug_data = _build_single_graph(
                        aug_code, row.get("language", ""),
                        row.get("label", 0), row["orig_id"],
                        vocab, codebert, tfidf_vec, max_nodes
                    )
                    if aug_data is not None:
                        valid_buf.append(aug_data)

                if len(valid_buf) >= self.chunk_size:
                    self._save_chunk(valid_buf)
                    valid_buf = []
                    gc.collect()
            else:
                skipped += 1

        if valid_buf:
            self._save_chunk(valid_buf)

        meta = {
            "chunk_paths": [os.path.basename(p) for p in self.chunk_files],
            "chunk_lengths": self.chunk_lengths,
        }
        with open(self.meta_file, "w") as f:
            json.dump(meta, f)
        print(f"✓ Saved {sum(self.chunk_lengths):,} graphs | "
              f"Skipped {skipped:,} samples")

    def _save_chunk(self, data_list: list):
        c_id = len(self.chunk_files)
        c_path = os.path.join(self.cache_path, f"chunk_{c_id}.pt")
        torch.save(data_list, c_path)
        self.chunk_files.append(c_path)
        self.chunk_lengths.append(len(data_list))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Data:
        chunk_idx = int(np.searchsorted(self.cum_lens, idx, side="right")) - 1
        item_idx = idx - self.cum_lens[chunk_idx]
        chunk = _load_chunk(self.chunk_files[chunk_idx])
        data = chunk[item_idx]

        # Reconstruct x_semantic from indices + shared mmap'd embeddings
        # Create new Data to avoid mutating cached chunk objects
        if hasattr(data, 'x_semantic_idx') and self._embeddings is not None:
            x_sem = torch.from_numpy(
                self._embeddings[data.x_semantic_idx.numpy()].astype(np.float32))
        elif hasattr(data, 'x_semantic_idx'):
            x_sem = torch.zeros(data.x_semantic_idx.shape[0], 768,
                                dtype=torch.float)
        elif hasattr(data, 'x_semantic'):
            return data  # backward compat with old cache format
        else:
            x_sem = torch.zeros(data.num_nodes, 768, dtype=torch.float)

        return Data(
            x_type=data.x_type,
            x_cont=data.x_cont,
            x_semantic=x_sem,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            tfidf=data.tfidf,
            num_nodes=data.num_nodes,
            y=data.y,
            orig_id=data.orig_id,
        )


def create_dataloaders(train_dataset, val_dataset,
                       batch_size: int = None,
                       num_workers: int = None,
                       pin_memory: bool = None):
    """Create train/val PyG DataLoaders with proper settings."""
    batch_size = batch_size or cfg.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else cfg.NUM_WORKERS
    env_cfg = cfg.ENV_CONFIG[cfg.ENV]
    pin_memory = pin_memory if pin_memory is not None else env_cfg.get("pin_memory", False)
    persistent = env_cfg.get("persistent_workers", False) and num_workers > 0

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    print(f"✓ DataLoaders: train={len(train_loader)} batches, "
          f"val={len(val_loader)} batches (bs={batch_size})")
    return train_loader, val_loader
