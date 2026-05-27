# tests/test_seq_encoder.py
import torch
from ampr.models.seq_encoder import SeqTransformer


def test_seq_transformer_shape_and_mask():
    torch.manual_seed(42)
    enc = SeqTransformer(d_model=64, n_heads=4, n_layers=2, dropout=0.0, ffn_mult=2)
    x = torch.randn(2, 20, 64, requires_grad=True)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 10:] = False
    out = enc(x, mask)
    assert out.shape == (2, 20, 64)
    out.sum().backward()
    assert x.grad is not None


def test_seq_transformer_padding_invariance():
    torch.manual_seed(42)
    enc = SeqTransformer(d_model=32, n_heads=2, n_layers=2, dropout=0.0)
    enc.eval()
    x = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    out_short = enc(x, mask)
    x_pad = torch.cat([x, torch.randn(1, 3, 32)], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
    out_pad = enc(x_pad, mask_pad)
    assert torch.allclose(out_short, out_pad[:, :5], atol=1e-4)
