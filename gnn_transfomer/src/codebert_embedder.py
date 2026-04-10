"""CodeBERT unique-text dedup embedding engine."""

import gc
import hashlib
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from src import config as cfg


class CodeBERTEmbedder:
    """Precompute per-node CodeBERT embeddings with unique-text deduplication.

    Strategy:
    1. Parse all ASTs to collect unique node texts
    2. Embed only unique texts (~10K-80K vs ~100M total nodes)
    3. Save text→index mapping + embeddings to disk
    4. At graph build time, lookup precomputed embeddings by text
    """

    def __init__(self, model_name="microsoft/codebert-base", device=None,
                 cache_dir=None):
        self.model_name = model_name
        self.device = device or cfg.DEVICE
        self.cache_dir = cache_dir or cfg.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self._model = None
        self._tokenizer = None

        # Loaded after precompute or load_cache
        self.text_to_idx = None   # {text: int}
        self.embeddings = None    # np.ndarray [num_unique, 768]

    def _load_model(self):
        """Lazily load CodeBERT model."""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad = False
        print(f"✓ CodeBERT loaded on {self.device}")

    def _collect_node_texts(self, df) -> list:
        """Parse all code samples and collect unique node texts from ASTs.

        Returns sorted list of unique texts, capped to MAX_UNIQUE_TEXTS
        most frequent to keep CodeBERT time manageable.
        """
        from src.tree_sitter_utils import parse_code, detect_language_fast
        from collections import deque, Counter

        counts = Counter()
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Collecting node texts"):
            lang = row.get("language", "") or detect_language_fast(row["code"])
            root = parse_code(row["code"], lang)
            if root is None:
                continue
            q = deque([root])
            while q:
                nd = q.popleft()
                try:
                    text = nd.text.decode("utf-8", errors="replace") if nd.text else ""
                except Exception:
                    text = ""
                if len(text) > 200:
                    text = text[:200]
                if text:  # skip empty strings
                    counts[text] += 1
                q.extend(nd.children)

        max_texts = getattr(cfg, 'MAX_UNIQUE_TEXTS', 200_000)
        if len(counts) > max_texts:
            print(f"  ⚖ Capping unique texts: {len(counts):,} → {max_texts:,} (top by frequency)")
            top_texts = [t for t, _ in counts.most_common(max_texts)]
        else:
            top_texts = list(counts.keys())

        return sorted(top_texts)

    def precompute(self, df, split_name: str):
        """Parse ASTs → collect unique node texts → embed → save to disk.

        Args:
            df: DataFrame with 'code' and optionally 'language' columns.
            split_name: e.g. "task_a_train" — used for cache filename.
        """
        # Cache key based on dataset size and split
        cache_hash = hashlib.md5(
            f"{self.model_name}_{len(df)}_{split_name}".encode()
        ).hexdigest()[:8]
        emb_path = os.path.join(self.cache_dir, f"codebert_emb_{split_name}_{cache_hash}.npy")
        idx_path = os.path.join(self.cache_dir, f"codebert_idx_{split_name}_{cache_hash}.json")

        if os.path.exists(emb_path) and os.path.exists(idx_path):
            print(f"✓ CodeBERT embeddings loaded from cache: {emb_path}")
            self.embeddings = np.load(emb_path, mmap_mode='r')
            with open(idx_path) as f:
                self.text_to_idx = json.load(f)
            return

        # Collect unique texts from AST nodes
        unique_texts = self._collect_node_texts(df)
        print(f"  CodeBERT: {len(df):,} samples → {len(unique_texts):,} unique node texts")

        if len(unique_texts) == 0:
            print("  ⚠ No node texts found — using zero embeddings")
            self.embeddings = np.zeros((1, 768), dtype=np.float32)
            self.text_to_idx = {"": 0}
            return

        # Embed
        self._load_model()
        batch_size = cfg.ENV_CONFIG[cfg.ENV].get("codebert_batch", 64)
        emb_f32 = self._embed_texts(unique_texts, batch_size)
        # Save as float16 to halve disk usage (~300 MB vs ~600 MB for 200K texts)
        self.embeddings = emb_f32.astype(np.float16)
        self.text_to_idx = {t: i for i, t in enumerate(unique_texts)}

        # Save
        np.save(emb_path, self.embeddings)
        with open(idx_path, "w") as f:
            json.dump(self.text_to_idx, f)
        size_mb = os.path.getsize(emb_path) / (1024 * 1024)
        print(f"✓ CodeBERT embeddings saved: {self.embeddings.shape} float16 "
              f"({size_mb:.0f} MB) → {emb_path}")

        # Reload as memory-mapped for efficient sharing with DataLoader workers
        self.embeddings = np.load(emb_path, mmap_mode='r')

    @torch.no_grad()
    def _embed_texts(self, texts: list, batch_size: int) -> np.ndarray:
        """Batch-encode texts → [N, 768] CLS embeddings."""
        all_embs = []
        amp_dtype = torch.bfloat16 if self.device.type == "mps" else torch.float16

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="  CodeBERT embed",
                      total=total_batches, miniters=max(1, total_batches // 20)):
            batch = [t if t.strip() else "<empty>" for t in texts[i:i + batch_size]]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=amp_dtype,
                                enabled=cfg.USE_AMP):
                outputs = self._model(**inputs)

            cls_emb = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            all_embs.append(cls_emb)

        return np.concatenate(all_embs, axis=0)

    def lookup(self, node_texts: list) -> np.ndarray:
        """Retrieve precomputed embeddings for a list of node texts.

        Returns [len(node_texts), 768] numpy array.
        """
        if self.text_to_idx is None or self.embeddings is None:
            raise RuntimeError("Call precompute() or load cache first")

        # Default embedding for unknown texts (zeros)
        default_idx = None
        indices = []
        for t in node_texts:
            idx = self.text_to_idx.get(t)
            if idx is not None:
                indices.append(idx)
            else:
                if default_idx is None:
                    default_idx = self.text_to_idx.get("", 0)
                indices.append(default_idx)

        return self.embeddings[indices]

    def unload(self):
        """Free CodeBERT model from GPU/RAM."""
        if self._model is not None:
            del self._model, self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ CodeBERT model unloaded")
