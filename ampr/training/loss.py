"""Custom loss functions for AMPR."""

import torch
import torch.nn as nn


class AMPRLoss(nn.Module):
    """
    AMPR total loss: BCE + λ·DAG_loss

    DAG_loss enforces True Path Rule: if protein annotates child GO term,
    must also annotate parent GO term. Penalizes: max(0, P_child - P_parent)²
    """

    def __init__(self, dag_matrix, lambda_dag=0.5):
        """
        Args:
            dag_matrix: (C, C) adjacency matrix, A[i,j]=1 if j is parent of i
            lambda_dag: weight of DAG loss term
        """
        super().__init__()
        self.register_buffer('dag_matrix', dag_matrix)
        self.lambda_dag = lambda_dag
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits: (batch, n_terms)
            labels: (batch, n_terms)

        Returns:
            loss: scalar
            loss_dict: {bce, dag} for logging
        """
        batch_size = logits.size(0)

        bce = self.bce_loss(logits, labels)

        probs = torch.sigmoid(logits)

        dag_penalty = 0.0
        for i in range(self.dag_matrix.size(0)):
            parents = (self.dag_matrix[i] > 0).nonzero(as_tuple=True)[0]
            if len(parents) > 0:
                p_child = probs[:, i]
                p_parents = probs[:, parents]
                violation = torch.relu(p_child.unsqueeze(1) - p_parents)
                dag_penalty += (violation ** 2).mean()

        if self.dag_matrix.sum() > 0:
            dag_penalty = dag_penalty / self.dag_matrix.sum()
        else:
            dag_penalty = torch.tensor(0.0, device=logits.device)

        loss = bce + self.lambda_dag * dag_penalty

        return loss, {'bce': bce.item(), 'dag': dag_penalty.item()}
