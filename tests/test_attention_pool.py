# tests/test_attention_pool.py
import torch
from ampr.models.attention_pool import DeepAttentionPool


def test_deep_attention_pool_shape_and_grad():
    torch.manual_seed(42)
    pool = DeepAttentionPool(input_dim=128, dropout=0.0)
    x = torch.randn(4, 50, 128, requires_grad=True)
    mask = torch.ones(4, 50, dtype=torch.bool)
    mask[0, 30:] = False  # protein 0 chỉ dài 30 residues
    out = pool(x, mask)
    assert out.shape == (4, 128)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0


def test_deep_attention_pool_masks_padding():
    torch.manual_seed(42)
    pool = DeepAttentionPool(input_dim=16, dropout=0.0)
    x = torch.randn(1, 10, 16)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :5] = True
    # Đặt residue padding (5..10) thành giá trị rất lớn — nếu mask đúng,
    # output không bị ảnh hưởng
    x_padded = x.clone()
    x_padded[0, 5:] = 1e6
    out_clean = pool(x, mask)
    out_padded = pool(x_padded, mask)
    assert torch.allclose(out_clean, out_padded, atol=1e-4)
