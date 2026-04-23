"""Custom loss functions for AMPR."""

import torch
import torch.nn as nn


class AMPRLoss(nn.Module):
    """
    AMPR total loss: BCE + λ·DAG_loss

    DAG_loss enforces GO True Path Rule: penalizes max(0, P_child - P_parent)²
    Vectorized over all child-parent pairs — avoids Python loop over n_terms.
    """

    def __init__(self, dag_matrix, lambda_dag=0.5):
        """
        Args:
            dag_matrix: (C, C) float tensor, A[i,j]=1 if j is parent of i
            lambda_dag: weight of DAG penalty term
        """
        super().__init__()
        self.register_buffer('dag_matrix', dag_matrix)
        self.lambda_dag = lambda_dag
        self.bce_loss = nn.BCEWithLogitsLoss()
        self._n_edges = float(dag_matrix.sum().item())

    def forward(self, logits, labels):
        """
        Args:
            logits: (batch, C)
            labels: (batch, C)

        Returns:
            loss: scalar tensor
            loss_dict: {'bce': float, 'dag': float} for logging
        """
        bce = self.bce_loss(logits, labels)

        if self._n_edges == 0:
            dag_penalty = torch.tensor(0.0, device=logits.device)
        else:
            probs = torch.sigmoid(logits)

            # probs_child[b, i, 1]  — expand for broadcasting
            # probs_parent[b, 1, j] — expand for broadcasting
            # dag_matrix[i, j] = 1 means j is parent of i
            # violation[b,i,j] = relu(P_child_i - P_parent_j) * A[i,j]
            probs_c = probs.unsqueeze(2)                     # (B, C, 1)
            probs_p = probs.unsqueeze(1)                     # (B, 1, C)
            mask = self.dag_matrix.unsqueeze(0)              # (1, C, C)

            violation = torch.relu(probs_c - probs_p) * mask # (B, C, C)
            dag_penalty = (violation ** 2).sum() / (self._n_edges * logits.size(0))

        loss = bce + self.lambda_dag * dag_penalty

        return loss, {'bce': bce.item(), 'dag': dag_penalty.item()}
