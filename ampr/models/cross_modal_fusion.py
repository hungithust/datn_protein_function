# ampr/models/cross_modal_fusion.py
"""Fuse 3 modality vectors via Transformer self-attention over modality tokens."""

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    Stack 3 modality vectors as tokens (B,3,D), apply Transformer encoder,
    return seq_token concatenated with mean of remaining tokens projected to D.

    Args:
        d_model: 512
        n_heads: 8
        n_layers: 2
        dropout: 0.1
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, h_seq: torch.Tensor, h_gnn: torch.Tensor,
                h_ppi: torch.Tensor, ppi_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_seq, h_gnn, h_ppi: (B, D)
            ppi_mask: (B,) bool — True = protein has PPI embedding
        Returns:
            z: (B, D)
        """
        B, D = h_seq.shape
        tokens = torch.stack([h_seq, h_gnn, h_ppi], dim=1)  # (B, 3, D)
        # key_padding_mask: True = ignore. seq & gnn never masked; ppi by ppi_mask
        kp = torch.zeros(B, 3, dtype=torch.bool, device=h_seq.device)
        kp[:, 2] = ~ppi_mask
        x = self.encoder(tokens, src_key_padding_mask=kp)  # (B, 3, D)
        seq_tok = x[:, 0]
        # Mean của gnn + ppi-nếu-có; tránh chia 0 bằng cách luôn count gnn
        other = x[:, 1:]                                    # (B, 2, D)
        other_mask = torch.ones(B, 2, device=h_seq.device)
        other_mask[:, 1] = ppi_mask.float()
        mean_other = (other * other_mask.unsqueeze(-1)).sum(dim=1) / other_mask.sum(dim=1, keepdim=True)
        return self.out(torch.cat([seq_tok, mean_other], dim=-1))
