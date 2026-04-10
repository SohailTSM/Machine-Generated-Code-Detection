"""Node vocabulary builder — BFS over ASTs to collect node types."""

import json
import os
from collections import Counter, deque

import pandas as pd
from tqdm import tqdm

from src import config as cfg
from src.tree_sitter_utils import parse_code


def build_global_vocab(dfs_list: list, cache_dir: str = None,
                       vocab_size: int = None,
                       graph_logic_version: str = None) -> dict:
    """Build (or load from cache) a node-type vocabulary from training data.

    Uses deque BFS (O(1) popleft) instead of list.pop(0).
    """
    cache_dir = cache_dir or cfg.CACHE_DIR
    vocab_size = vocab_size or cfg.VOCAB_SIZE
    graph_logic_version = graph_logic_version or cfg.GRAPH_LOGIC_VERSION

    vocab_path = os.path.join(cache_dir, f"global_vocab_{graph_logic_version}.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f"✓ Loaded vocab from cache ({len(vocab)} types): {vocab_path}")
        return vocab

    # Combine DataFrames
    dfs = [d for d in dfs_list if d is not None]
    sampled = pd.concat(dfs, ignore_index=True)

    t_counts = Counter()
    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="  Vocab scan"):
        root = parse_code(row["code"], row.get("language", ""))
        if root is None:
            continue
        q = deque([root])
        while q:
            nd = q.popleft()
            t_counts[nd.type] += 1
            q.extend(nd.children)

    vocab = {"<UNK>": 0}
    for i, (t, _) in enumerate(t_counts.most_common(vocab_size - 1), 1):
        vocab[t] = i

    os.makedirs(cache_dir, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"✓ Vocab built and saved: {len(vocab)} types → {vocab_path}")
    return vocab


def load_vocab(path: str = None) -> dict:
    """Load vocab JSON from cache."""
    path = path or os.path.join(cfg.CACHE_DIR,
                                 f"global_vocab_{cfg.GRAPH_LOGIC_VERSION}.json")
    with open(path) as f:
        return json.load(f)


def encode_node_types(node_types: list, vocab: dict) -> list:
    """Map node type strings to integer IDs."""
    return [vocab.get(t, 0) for t in node_types]
