"""
Lightweight dataset that loads pre-computed features and embeddings from cache.

Requires running extract_features.py and extract_embeddings.py first.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class CachedDataset(Dataset):
    """
    Loads from cache/:
      {split}_features.pt   - FloatTensor [N, 102]
      {split}_embeddings.pt - FloatTensor [N, 2048]
      {split}_meta.pt       - dict with keys: ids, num_samples, [labels]

    Labels are optional (absent for test split).
    """

    def __init__(self, cache_dir: str, split: str):
        cache = Path(cache_dir)

        feat_path = cache / f"{split}_features.pt"
        emb_path  = cache / f"{split}_embeddings.pt"
        meta_path = cache / f"{split}_meta.pt"

        if not feat_path.exists():
            raise FileNotFoundError(
                f"Features not found: {feat_path}\n"
                f"Run: python extract_features.py --splits {split}"
            )
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {emb_path}\n"
                f"Run: python extract_embeddings.py --splits {split}"
            )

        self.features   = torch.load(feat_path)   # [N, 102]
        self.embeddings = torch.load(emb_path)    # [N, 2048]
        meta            = torch.load(meta_path)

        self.ids        = meta['ids']
        self.has_labels = 'labels' in meta
        if self.has_labels:
            self.labels = meta['labels']          # [N]

        N = len(self.features)
        assert len(self.embeddings) == N, (
            f"Feature/embedding count mismatch for '{split}': "
            f"{N} features vs {len(self.embeddings)} embeddings"
        )
        if self.has_labels:
            assert len(self.labels) == N, (
                f"Label count mismatch for '{split}': {N} vs {len(self.labels)}"
            )

        print(f"[{split}] Loaded {N:,} samples")
        print(f"  features:   {self.features.shape}")
        print(f"  embeddings: {self.embeddings.shape}")
        if self.has_labels:
            counts = self.labels.bincount()
            print(f"  labels:     {counts[0].item():,} Human  |  {counts[1].item():,} AI")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        item = {
            'handcrafted_features': self.features[idx],
            'backbone_embedding':   self.embeddings[idx],
            'id': self.ids[idx],
        }
        if self.has_labels:
            item['labels'] = self.labels[idx]
        return item
