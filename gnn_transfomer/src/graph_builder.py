"""AST → Graph construction with 4 edge types and structural features."""

import torch

from src import config as cfg


def _subtree_sizes_iterative(root) -> dict:
    """Compute subtree sizes via iterative post-order traversal."""
    cache = {}
    stack = [root]
    order = []
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(n.children)
    for n in reversed(order):
        cache[id(n)] = 1 + sum(cache[id(c)] for c in n.children)
    return cache


def ast_to_graph_v7(root_node, node_vocab: dict, max_nodes: int = None):
    """Convert AST root into a graph dict with 4 edge types.

    Edge types:
        0: parent→child
        1: child→parent
        2: sibling↔sibling (within same parent)
        3: next-token (consecutive leaf nodes by start_byte)

    Returns dict with keys:
        type_ids, cont_features, edge_index, edge_type,
        node_texts, num_nodes
    Or None if the graph is empty.
    """
    max_nodes = max_nodes or cfg.MAX_NODES

    type_ids = []
    cont_fs = []
    node_texts = []
    src, dst = [], []
    edge_types = []
    collected = []
    last_child = {}  # p_idx → most-recently-added child idx
    leaf_nodes = []  # (start_byte, node_idx) for next-token edges

    st_sizes = _subtree_sizes_iterative(root_node)
    total_nodes = st_sizes.get(id(root_node), 1)

    # DFS: (node, depth, parent_idx, child_position)
    stack = [(root_node, 0, -1, 0)]

    while stack:
        if len(collected) >= max_nodes:
            break
        node, depth, p_idx, c_idx = stack.pop()
        cur_idx = len(collected)
        collected.append(node)

        # Node type ID
        type_ids.append(node_vocab.get(node.type, 0))

        # Node text (for CodeBERT embedding lookup)
        try:
            text = node.text.decode("utf-8", errors="replace") if node.text else ""
        except Exception:
            text = ""
        # Truncate long texts for efficiency
        if len(text) > 200:
            text = text[:200]
        node_texts.append(text)

        # Structural features (6 floats, all in [0,1])
        sz = st_sizes.get(id(node), 1)
        parent_sz = st_sizes.get(id(collected[p_idx]), 1) if p_idx >= 0 else total_nodes
        cont_fs.append([
            min(depth, 100) / 100,
            min(c_idx, 50) / 50,
            min(node.child_count, 50) / 50,
            float(node.is_named),
            min(sz, max_nodes) / max_nodes,
            sz / max(parent_sz, 1),
        ])

        # Track leaf nodes for next-token edges
        if node.child_count == 0:
            leaf_nodes.append((node.start_byte, cur_idx))

        # Edges
        if p_idx >= 0:
            # Type 0: parent→child
            src.append(p_idx)
            dst.append(cur_idx)
            edge_types.append(0)
            # Type 1: child→parent
            src.append(cur_idx)
            dst.append(p_idx)
            edge_types.append(1)
            # Type 2: sibling edges
            if p_idx in last_child:
                prev = last_child[p_idx]
                src.extend([prev, cur_idx])
                dst.extend([cur_idx, prev])
                edge_types.extend([2, 2])
            last_child[p_idx] = cur_idx

        # Push children in reverse order (leftmost first)
        for ci in range(node.child_count - 1, -1, -1):
            stack.append((node.children[ci], depth + 1, cur_idx, ci))

    n = len(type_ids)
    if n == 0:
        return None

    # Type 3: next-token edges (consecutive leaf nodes by start_byte)
    leaf_nodes.sort(key=lambda x: x[0])
    for i in range(len(leaf_nodes) - 1):
        _, idx_a = leaf_nodes[i]
        _, idx_b = leaf_nodes[i + 1]
        src.extend([idx_a, idx_b])
        dst.extend([idx_b, idx_a])
        edge_types.extend([3, 3])

    # Build edge index
    if src:
        e_idx = torch.tensor([src, dst], dtype=torch.long)
        e_type = torch.tensor(edge_types, dtype=torch.long)
    elif n == 1:
        # Single-node graph: self-loop
        e_idx = torch.tensor([[0], [0]], dtype=torch.long)
        e_type = torch.tensor([0], dtype=torch.long)
    else:
        # Degenerate multi-node with no edges: chain sequentially
        s = list(range(n - 1)) + list(range(1, n))
        d = list(range(1, n)) + list(range(n - 1))
        e_idx = torch.tensor([s, d], dtype=torch.long)
        e_type = torch.tensor([0] * len(s), dtype=torch.long)

    return {
        "type_ids": torch.tensor(type_ids, dtype=torch.long),
        "cont_features": torch.tensor(cont_fs, dtype=torch.float),
        "edge_index": e_idx,
        "edge_type": e_type,
        "node_texts": node_texts,
        "num_nodes": n,
    }
