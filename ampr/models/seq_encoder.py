# ampr/models/seq_encoder.py
"""Transformer encoder over per-residue embeddings (e.g., ESM-2 output)."""

import torch
import torch.nn as nn


class SeqTransformer(nn.Module):
    """
    PreNorm Transformer encoder, masked self-attention over L residues.

    Args:
        d_model: residue embedding dim (e.g., 1280 for ESM-2)
        n_heads: number of attention heads
        n_layers: stack depth
        dropout: dropout
        ffn_mult: hidden multiplier for FFN
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 2,
                 dropout: float = 0.1, ffn_mult: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            mask: (B, L) bool — True = real residue
        Returns:
            (B, L, D)
        """
        # nn.Transformer expects key_padding_mask: True = ignore
        key_padding = ~mask
        return self.encoder(x, src_key_padding_mask=key_padding)
