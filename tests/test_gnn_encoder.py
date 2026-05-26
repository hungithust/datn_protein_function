"""Tests for ampr/models/gnn_encoder.py."""

import torch


def test_forward_shape():
    from ampr.models.gnn_encoder import GNNEncoder
    enc = GNNEncoder(input_dim=64, hidden_dim=128, output_dim=512,
                     cmap_threshold=10.0)
    L = 20
    seq_feat = torch.randn(1, L, 64)
    cmap = torch.rand(1, L, L) * 15
    mask = torch.ones(1, L, dtype=torch.bool)
    out = enc(seq_feat, cmap, mask)
    assert out.shape == (1, 512)


def test_handles_variable_length_batch():
    from ampr.models.gnn_encoder import GNNEncoder
    enc = GNNEncoder(input_dim=64, hidden_dim=128, output_dim=512,
                     cmap_threshold=10.0)
    B, L = 4, 30
    seq_feat = torch.randn(B, L, 64)
    cmap = torch.rand(B, L, L) * 15
    # Pretend proteins have lengths 10, 20, 30, 25
    lengths = [10, 20, 30, 25]
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, lx in enumerate(lengths):
        mask[i, :lx] = True
    out = enc(seq_feat, cmap, mask)
    assert out.shape == (B, 512)


def test_gradient_flows():
    from ampr.models.gnn_encoder import GNNEncoder
    enc = GNNEncoder(input_dim=64, hidden_dim=128, output_dim=512,
                     cmap_threshold=10.0)
    seq_feat = torch.randn(2, 10, 64, requires_grad=True)
    cmap = torch.rand(2, 10, 10) * 15
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = enc(seq_feat, cmap, mask)
    loss = out.sum()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
