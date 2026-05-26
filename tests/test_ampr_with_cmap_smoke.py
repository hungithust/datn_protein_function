"""End-to-end forward pass: AMPRModel in GNN mode produces valid logits."""

import torch


def test_ampr_gnn_forward():
    from ampr.models.ampr import AMPRModel
    model = AMPRModel(d_hidden=128, n_terms=10, structure_modality='gnn',
                      ppi_dim=256, gnn_input_dim=26, gnn_hidden_dim=64,
                      classifier='linear')
    B, L = 2, 15
    x_seq = torch.randn(B, 1024)
    x_3di = torch.zeros(B, 1024)  # ignored in gnn mode
    x_ppi = torch.randn(B, 256)
    seq_1hot = torch.zeros(B, L, 26)
    seq_1hot[..., 0] = 1.0
    cmap = torch.rand(B, L, L) * 15
    mask = torch.ones(B, L, dtype=torch.bool)
    logits = model(x_seq, x_3di, x_ppi, cmap=cmap, cmap_mask=mask, seq_1hot=seq_1hot)
    assert logits.shape == (B, 10)


def test_ampr_prostt5_mode_unchanged():
    """Backward-compat: default mode must work without cmap."""
    from ampr.models.ampr import AMPRModel
    model = AMPRModel(d_hidden=128, n_terms=10, ppi_dim=256, classifier='linear')
    x_seq = torch.randn(2, 1024)
    x_3di = torch.randn(2, 1024)
    x_ppi = torch.randn(2, 256)
    logits = model(x_seq, x_3di, x_ppi)
    assert logits.shape == (2, 10)
