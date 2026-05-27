import torch
from ampr.models.ampr import AMPRModelV3


def test_v3_forward_shape_and_grad():
    torch.manual_seed(42)
    m = AMPRModelV3(
        n_terms=12,
        seq_dim=64, seq_n_heads=4, seq_n_layers=1,
        gnn_node_dim=32, gnn_n_layers=2,
        ppi_dim=16,
        d_hidden=64, fusion_n_heads=4, fusion_n_layers=1,
        classifier='both', go_emb_dim=24,
        cmap_threshold=10.0, dropout=0.0,
    )
    B, L = 2, 8
    batch = {
        'x_seq_residue': torch.randn(B, L, 64, requires_grad=True),
        'seq_mask': torch.ones(B, L, dtype=torch.bool),
        'cmap': torch.rand(B, L, L) * 20,
        'x_ppi': torch.randn(B, 16),
        'ppi_mask': torch.tensor([True, False]),
    }
    go_emb = torch.randn(12, 24)
    logits = m(batch, go_emb=go_emb)
    assert logits.shape == (B, 12)
    logits.sum().backward()
    assert batch['x_seq_residue'].grad is not None
