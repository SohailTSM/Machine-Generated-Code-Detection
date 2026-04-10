"""Loss functions — FocalLoss with label smoothing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with per-task gamma and KL-divergence style label smoothing.

    Args:
        gamma:           Focusing parameter. Higher = more focus on hard examples.
        weight:          Per-class weight tensor [C].
        label_smoothing: Label smoothing epsilon ∈ [0, 1).
        reduction:       'mean' or 'none'.
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0,
                 reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        if self.label_smoothing > 0:
            # KL-divergence style: smooth targets as probability distribution
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(
                    1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            ce_loss = F.nll_loss(
                log_probs, targets, weight=self.weight, reduction='none')

        # Focal modulation
        probs = torch.exp(-ce_loss)
        focal_weight = (1 - probs) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        return loss
