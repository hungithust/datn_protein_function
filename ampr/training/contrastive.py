# ampr/training/contrastive.py
"""Multi-label supervised contrastive loss (Module A)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelSupConLoss(nn.Module):
    """Jaccard-weighted supervised contrastive loss for multi-label targets.

    Generalizes SupCon (Khosla et al., NeurIPS 2020) to soft multi-label
    positives: two proteins are positives in proportion to the Jaccard overlap
    of their GO-term label sets. Pulls together functionally similar proteins
    in representation space to improve low-identity generalization (cf. HEAL,
    Gu et al., Bioinformatics btad410, 2023).
    """

    def __init__(self, temp: float = 0.1, jaccard_thr: float = 0.0,
                 eps: float = 1e-8):
        super().__init__()
        self.temp = temp
        self.jaccard_thr = jaccard_thr
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, d) projected representations (un-normalized OK).
            labels:   (B, C) binary multi-label targets.
        Returns:
            scalar loss (0.0 if no Jaccard-positive pairs in the batch).
        """
        B = features.size(0)
        if B < 2:
            return features.sum() * 0.0

        z = F.normalize(features, dim=1)
        sim = (z @ z.t()) / self.temp
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # stability

        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)

        lab = labels.float()
        inter = lab @ lab.t()
        card = lab.sum(dim=1, keepdim=True)
        union = card + card.t() - inter
        jacc = inter / (union + self.eps)
        jacc = jacc.masked_fill(self_mask, 0.0)
        if self.jaccard_thr > 0:
            jacc = torch.where(jacc >= self.jaccard_thr, jacc,
                               torch.zeros_like(jacc))

        w_sum = jacc.sum(dim=1)
        valid = w_sum > self.eps
        if valid.sum() == 0:
            return features.sum() * 0.0

        per_row = -(jacc * log_prob).sum(dim=1) / (w_sum + self.eps)
        return per_row[valid].mean()
