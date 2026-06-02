import torch
from ampr.models.label_attention import LabelAttentionHead


def test_label_attn_shape_and_grad():
    torch.manual_seed(0)
    head = LabelAttentionHead(d_hidden=64, go_emb_dim=24, n_terms=12,
                              n_heads=4, dropout=0.0)
    z = torch.randn(3, 64, requires_grad=True)
    go_emb = torch.randn(12, 24, requires_grad=True)
    logits = head(z, go_emb)
    assert logits.shape == (3, 12)
    logits.sum().backward()
    assert z.grad is not None and go_emb.grad is not None
    assert head.bias.grad is not None


def test_label_attn_requires_divisible_heads():
    import pytest
    with pytest.raises(AssertionError):
        LabelAttentionHead(d_hidden=10, go_emb_dim=8, n_terms=4, n_heads=3)
