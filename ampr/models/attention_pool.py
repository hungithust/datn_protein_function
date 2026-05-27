# ampr/models/attention_pool.py
"""Deep attention pooling for variable-length residue embeddings."""

import torch
import torch.nn as nn


class DeepAttentionPool(nn.Module):
    """
    Two-layer learned attention over residues with masking.

    Args:
        input_dim: D — chiều residue embedding
        dropout: dropout giữa 2 layer
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, input_dim)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            mask: (B, L) bool — True = real residue, False = padding
        Returns:
            (B, D) pooled
        """
        h = self.act(self.hidden(x))
        h = self.drop(h)
        scores = self.score(h)               # (B, L, 1)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(scores, dim=1)  # (B, L, 1)
        return (weights * x).sum(dim=1)
