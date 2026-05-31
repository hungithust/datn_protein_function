# ampr/training/loss.py — full new content
"""Custom loss functions for AMPR."""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021).

    Down-weight easy negatives more aggressively than positives, with
    probability shifting (clip) to drop very-easy negatives entirely.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x_sig = torch.sigmoid(logits)
        xs_pos = x_sig
        xs_neg = 1.0 - x_sig
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        loss_pos = labels * log_pos
        loss_neg = (1.0 - labels) * log_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * labels
            pt1 = xs_neg * (1.0 - labels)
            pt = pt0 + pt1
            gamma = self.gamma_pos * labels + self.gamma_neg * (1.0 - labels)
            w = torch.pow(1.0 - pt, gamma)
            loss = -(loss_pos + loss_neg) * w
        else:
            loss = -(loss_pos + loss_neg)

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class AMPRLoss(nn.Module):
    """
    AMPR total loss: classification + λ·DAG.

    loss_type:
        'bce' — BCEWithLogitsLoss (default backward-compat)
        'asl' — AsymmetricLoss
    """

    def __init__(self, dag_matrix, lambda_dag: float = 0.5,
                 loss_type: str = 'bce',
                 asl_gamma_neg: float = 4.0, asl_gamma_pos: float = 0.0,
                 asl_clip: float = 0.05,
                 pos_weight=None):
        super().__init__()
        self.register_buffer('dag_matrix', dag_matrix)
        self.lambda_dag = lambda_dag
        self.loss_type = loss_type
        if loss_type == 'bce':
            # pos_weight (per-class) chống mất cân bằng lớp: gradient positive sống
            # ngay cả khi logit rất âm (BCEWithLogitsLoss ổn định số học).
            # BCEWithLogitsLoss tự register pos_weight làm buffer -> .to(device) sẽ chuyển.
            if pos_weight is not None and not torch.is_tensor(pos_weight):
                pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_type == 'asl':
            self.cls_loss = AsymmetricLoss(asl_gamma_neg, asl_gamma_pos, asl_clip)
        else:
            raise ValueError(f"unknown loss_type: {loss_type}")
        self._n_edges = float(dag_matrix.sum().item())

    def forward(self, logits, labels):
        cls = self.cls_loss(logits, labels)

        if self._n_edges == 0:
            dag_penalty = torch.tensor(0.0, device=logits.device)
        else:
            probs = torch.sigmoid(logits)
            probs_c = probs.unsqueeze(2)
            probs_p = probs.unsqueeze(1)
            mask = self.dag_matrix.unsqueeze(0)
            violation = torch.relu(probs_c - probs_p) * mask
            dag_penalty = (violation ** 2).sum() / (self._n_edges * logits.size(0))

        loss = cls + self.lambda_dag * dag_penalty
        return loss, {'cls': cls.item(), 'dag': dag_penalty.item()}
