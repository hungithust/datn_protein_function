"""GCN encoder over residue contact map."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single-layer mean-aggregation GCN with residual + LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, A_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, L, D_in); A_norm: (B, L, L) row-normalized adjacency; mask: (B, L) bool
        agg = torch.bmm(A_norm, h)                 # (B, L, D_in)
        out = self.lin(agg) + self.skip(h)
        out = F.relu(out)
        out = self.norm(out)
        out = out * mask.unsqueeze(-1).float()
        return out


class AttentionPool(nn.Module):
    """Mask-aware attention pool over residue dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.hidden = nn.Linear(dim, dim)
        self.attn = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.attn(torch.tanh(self.hidden(h)))  # (B, L, 1)
        score = score.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(score, dim=1)
        return (w * h).sum(dim=1)                       # (B, D)


class GNNEncoder(nn.Module):
    """3-layer GCN over contact map -> attention pool -> (B, output_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 cmap_threshold: float = 10.0, n_layers: int = 3):
        super().__init__()
        self.cmap_threshold = cmap_threshold
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.pool = AttentionPool(output_dim)

    def _normalize(self, cmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Binary adjacency
        A = (cmap < self.cmap_threshold).float()
        # Zero out padding rows/cols
        m2 = mask.unsqueeze(-1).float() * mask.unsqueeze(1).float()
        A = A * m2
        # Add self-loops
        eye = torch.eye(A.shape[1], device=A.device).unsqueeze(0)
        A = A + eye * mask.unsqueeze(-1).float()
        # Row-normalize
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return A / deg

    def forward(self, seq_feat: torch.Tensor, cmap: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        A_norm = self._normalize(cmap, mask)
        h = seq_feat
        for layer in self.layers:
            h = layer(h, A_norm, mask)
        return self.pool(h, mask)
