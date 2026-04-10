"""
Hybrid classifier: StarCoder2-3b (frozen backbone) + handcrafted features.

Architecture:
  backbone_embedding [2048]  ──► LayerNorm ──► tanh-Linear ──► [2048]
  handcrafted_features [102] ──► MLP (102→256→128) ──► [128]
                                                               │
                                                     cat [2176] ──► fusion MLP ──► [2]

The backbone is always used frozen (weights loaded 8-bit in extract_embeddings.py;
during training only the head — ~7 M params — is updated).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig


class TaskAModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        # ── Backbone (frozen, 8-bit) ──────────────────────────────────────
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

        if torch.cuda.device_count() > 1:
            device_map = {"": 1}
            print(f"Loading backbone on GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            device_map = "auto"

        self.backbone = AutoModel.from_pretrained(
            config['model']['name'],
            quantization_config=quant_cfg,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'gradient_checkpointing_disable'):
            self.backbone.gradient_checkpointing_disable()
        print("Backbone loaded and frozen.")

        hidden_size = self.backbone.config.hidden_size  # 2048

        # ── Head components ───────────────────────────────────────────────
        # Normalise + project backbone output
        self.backbone_norm = nn.LayerNorm(hidden_size)
        self.pooler = nn.Linear(hidden_size, hidden_size)

        # Process handcrafted features
        feat_dim = config['model']['handcrafted_dim']  # 102
        self.feature_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Fusion + classifier
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, config['model']['num_labels']),
        )

        # Move head to the same GPU as the backbone
        if torch.cuda.is_available():
            head_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda")
        else:
            head_device = torch.device("cpu")

        self.backbone_norm = self.backbone_norm.to(head_device)
        self.pooler = self.pooler.to(head_device)
        self.feature_mlp = self.feature_mlp.to(head_device)
        self.fusion = self.fusion.to(head_device)


    # ------------------------------------------------------------------
    def forward(
        self,
        handcrafted_features: torch.Tensor,
        backbone_embedding: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        head_device = self.backbone_norm.weight.device

        # ── Backbone embedding ────────────────────────────────────────────
        if backbone_embedding is not None:
            # Fast path: pre-computed embedding (from extract_embeddings.py)
            mean_pooled = backbone_embedding.to(head_device, dtype=torch.float32)
        else:
            # Slow path: run backbone at inference time (needs input_ids)
            with torch.no_grad():
                out = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
                last_hidden = out.last_hidden_state
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.shape).float()
                mean_pooled = (last_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            mean_pooled = mean_pooled.to(head_device)

        # ── Sanitise backbone output ──────────────────────────────────────
        mean_pooled = self.backbone_norm(mean_pooled)
        mean_pooled = torch.nan_to_num(mean_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
        pooled = torch.tanh(self.pooler(mean_pooled))

        # ── Process handcrafted features ──────────────────────────────────
        feats = handcrafted_features.to(head_device, dtype=torch.float32)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5, 5)
        feat_emb = self.feature_mlp(feats)

        # ── Fuse and classify ─────────────────────────────────────────────
        combined = torch.cat([pooled, feat_emb], dim=-1)  # [B, 2048+128]
        logits = self.fusion(combined)

        loss = None
        if labels is not None:
            labels = labels.to(head_device)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits}
