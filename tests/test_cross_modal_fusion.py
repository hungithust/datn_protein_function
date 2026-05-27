# tests/test_cross_modal_fusion.py
import torch
from ampr.models.cross_modal_fusion import CrossModalFusion


def test_fusion_shape_and_grad():
    torch.manual_seed(42)
    f = CrossModalFusion(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    h_seq = torch.randn(3, 32, requires_grad=True)
    h_gnn = torch.randn(3, 32, requires_grad=True)
    h_ppi = torch.randn(3, 32, requires_grad=True)
    ppi_mask = torch.tensor([True, False, True])  # protein 1 không có PPI
    z = f(h_seq, h_gnn, h_ppi, ppi_mask)
    assert z.shape == (3, 32)
    z.sum().backward()
    assert h_seq.grad is not None and h_gnn.grad is not None


def test_fusion_ppi_mask_ignored_value():
    """Protein không có PPI → giá trị h_ppi không ảnh hưởng output."""
    torch.manual_seed(42)
    f = CrossModalFusion(d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    f.eval()
    h_seq = torch.randn(1, 16)
    h_gnn = torch.randn(1, 16)
    h_ppi_a = torch.zeros(1, 16)
    h_ppi_b = torch.full((1, 16), 1000.0)
    ppi_mask = torch.tensor([False])
    z_a = f(h_seq, h_gnn, h_ppi_a, ppi_mask)
    z_b = f(h_seq, h_gnn, h_ppi_b, ppi_mask)
    assert torch.allclose(z_a, z_b, atol=1e-4)
