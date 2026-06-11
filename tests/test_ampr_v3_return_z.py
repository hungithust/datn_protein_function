# tests/test_ampr_v3_return_z.py
import torch
from ampr.models.ampr import AMPRModelV3
from ampr.data.dataset import collate_variable_length


def _tiny_batch():
    torch.manual_seed(0)
    items = []
    for i, L in enumerate((6, 5)):
        items.append({
            'prot_id': f'P{i}',
            'x_seq_residue': torch.rand(L, 16),
            'seq_len': L,
            'cmap': (torch.rand(L, L) * 20),
            'x_ppi': torch.rand(8),
            'ppi_mask': torch.tensor(True),
            'labels': torch.randint(0, 2, (2,)).float(),
        })
    return collate_variable_length(items)


def _model():
    return AMPRModelV3(n_terms=2, seq_dim=16, seq_n_heads=2, seq_n_layers=1,
                       gnn_node_dim=16, gnn_n_layers=1, ppi_dim=8,
                       d_hidden=16, fusion_n_heads=2, fusion_n_layers=1,
                       go_emb_dim=8, dropout=0.0, contrastive_proj_dim=4)


def test_return_z_shapes():
    model = _model().eval()
    batch = _tiny_batch()
    go_emb = torch.rand(2, 8)
    logits, z = model(batch, go_emb=go_emb, return_z=True)
    assert logits.shape == (2, 2)
    assert z.shape == (2, 16)  # (B, d_hidden)


def test_logits_only_backward_compat():
    model = _model().eval()
    out = model(_tiny_batch(), go_emb=torch.rand(2, 8))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 2)


def test_ablate_zeroes_branch_changes_logits():
    model = _model().eval()
    batch = _tiny_batch()
    go_emb = torch.rand(2, 8)
    full = model(batch, go_emb=go_emb)
    ablated = model(batch, go_emb=go_emb, ablate=('gnn',))
    assert not torch.allclose(full, ablated)


def test_project_contrastive_shape():
    model = _model().eval()
    _, z = model(_tiny_batch(), go_emb=torch.rand(2, 8), return_z=True)
    feats = model.project_contrastive(z)
    assert feats.shape == (2, 4)
