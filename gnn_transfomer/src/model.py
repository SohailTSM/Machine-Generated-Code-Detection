"""SemanticGraphTransformer — core GNN architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge, TransformerConv
from torch_geometric.nn.aggr import AttentionalAggregation

from src import config as cfg


class SemanticGraphTransformer(nn.Module):
    """Graph Transformer for code classification.

    Architecture (v11_medium):
        TypeEmbed(384) ⊕ Structural(6) ⊕ CodeBERT→Proj(384) → Linear(774→768)
        → 5×TransformerConv(768, heads=12, edge_dim=96) + Residual + LayerNorm + JK(max)
        → GlobalAttention Pooling → [B, 768]
        → Concat TF-IDF [B, 100] → [B, 868]
        → MLP(868→768→384→NUM_CLASSES)

    ~17M parameters. Sized to use ~8-10 GB on T4 GPU (14.56 GB).
    """

    def __init__(self,
                 vocab_size: int = None,
                 num_classes: int = None,
                 hidden_dim: int = None,
                 emb_dim: int = None,
                 semantic_dim: int = None,
                 cont_dim: int = None,
                 edge_types: int = None,
                 edge_emb_dim: int = None,
                 num_layers: int = None,
                 heads: int = None,
                 tfidf_dim: int = None,
                 dropout: float = None):
        super().__init__()

        vocab_size = vocab_size or cfg.VOCAB_SIZE
        num_classes = num_classes or cfg.NUM_CLASSES
        hidden_dim = hidden_dim or cfg.HIDDEN_DIM
        emb_dim = emb_dim or cfg.EMB_DIM
        semantic_dim = semantic_dim or cfg.SEMANTIC_DIM
        cont_dim = cont_dim or cfg.CONT_DIM
        edge_types = edge_types or cfg.EDGE_TYPES
        edge_emb_dim = edge_emb_dim or cfg.EDGE_EMB_DIM
        num_layers = num_layers or cfg.NUM_GNN_LAYERS
        heads = heads or cfg.TRANSFORMER_HEADS
        tfidf_dim = tfidf_dim or cfg.TFIDF_DIM
        dropout = dropout or cfg.DROPOUT

        # ── Node feature encoding ──
        self.type_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.semantic_proj = nn.Linear(768, semantic_dim)
        input_dim = emb_dim + cont_dim + semantic_dim  # 128+6+128 = 262
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # ── Edge type embedding ──
        self.edge_embed = nn.Embedding(edge_types, edge_emb_dim)

        # ── Transformer Conv blocks ──
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=edge_emb_dim,
                dropout=0.1,
                concat=True,
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # ── JumpingKnowledge ──
        self.jk = JumpingKnowledge(mode="max")

        # ── Global Attention Pooling ──
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pool = AttentionalAggregation(gate_nn)

        # ── Classifier ──
        fusion_dim = hidden_dim + tfidf_dim  # 256 + 100 = 356
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.dropout = nn.Dropout(dropout)
        self._hidden_dim = hidden_dim

    def forward(self, x_type, x_cont, x_semantic, edge_index, edge_type,
                batch, tfidf):
        """Forward pass.

        Args:
            x_type:     [N] long — node type IDs
            x_cont:     [N, 6] float — structural features
            x_semantic: [N, 768] float — CodeBERT embeddings
            edge_index: [2, E] long — edge connectivity
            edge_type:  [E] long — edge type labels {0,1,2,3}
            batch:      [N] long — batch assignment
            tfidf:      [B, 100] float — per-graph TF-IDF features

        Returns:
            logits: [B, NUM_CLASSES]
        """
        # 1. Node encoding
        te = self.type_embed(x_type)            # [N, 128]
        se = self.semantic_proj(x_semantic)      # [N, 128]
        h = torch.cat([te, x_cont, se], dim=1)  # [N, 262]
        h = F.relu(self.input_norm(self.input_proj(h)))  # [N, 256]

        # 2. Edge encoding
        edge_attr = self.edge_embed(edge_type)   # [E, 32]

        # 3. Graph Transformer blocks + JK
        xs = [h]
        for conv, norm in zip(self.convs, self.norms):
            res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.relu(norm(h))
            h = self.dropout(h + res)
            xs.append(h)
        h = self.jk(xs)  # [N, 256]

        # 4. Global attention pooling
        graph_emb = self.pool(h, index=batch)  # [B, 256]

        # 5. TF-IDF fusion — PyG batches 1D tensors as [B*D], reshape to [B, D]
        num_graphs = int(batch.max()) + 1
        tfidf = tfidf.view(num_graphs, -1)
        combined = torch.cat([graph_emb, tfidf], dim=1)  # [B, 356]

        # 6. Classification
        return self.classifier(combined)  # [B, C]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
