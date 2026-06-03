"""Label-attention classification head.

Scores the fused protein vector z against the GO label-embedding matrix with
multi-head bilinear attention: logit[b,c] = sum_h <Wq(z)_h, Wk(go_c)_h> / sqrt(d_head) + bias_c.
A richer alternative to the flat `z · proj(go)^T` dot-product (TALE/ATGO-style).
"""
import torch
import torch.nn as nn


class LabelAttentionHead(nn.Module):
    def __init__(self, d_hidden: int, go_emb_dim: int, n_terms: int,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.scale = self.d_head ** -0.5
        self.q = nn.Linear(d_hidden, d_hidden)
        self.k = nn.Linear(go_emb_dim, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(n_terms))

    def forward(self, z: torch.Tensor, go_emb: torch.Tensor) -> torch.Tensor:
        # z: (B, D); go_emb: (C, go_emb_dim)
        B = z.shape[0]
        C = go_emb.shape[0]
        q = self.q(self.drop(z)).view(B, self.n_heads, self.d_head)
        k = self.k(go_emb).view(C, self.n_heads, self.d_head)
        logits = torch.einsum('bhd,chd->bc', q, k) * self.scale
        return logits + self.bias
