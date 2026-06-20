# tests/test_contrastive.py
import torch
from ampr.training.contrastive import MultiLabelSupConLoss


def test_supcon_shape_and_grad():
    torch.manual_seed(0)
    loss_fn = MultiLabelSupConLoss(temp=0.1)
    feats = torch.randn(6, 16, requires_grad=True)
    labels = torch.randint(0, 2, (6, 4)).float()
    # ensure at least one shared-label pair exists
    labels[0] = labels[1]
    l = loss_fn(feats, labels)
    assert l.dim() == 0
    l.backward()
    assert feats.grad is not None


def test_supcon_pulls_shared_labels():
    """Features aligned with shared-label structure give lower loss."""
    loss_fn = MultiLabelSupConLoss(temp=0.1)
    labels = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0],
                           [0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float)
    aligned = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    misaligned = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    assert loss_fn(aligned, labels).item() < loss_fn(misaligned, labels).item()


def test_supcon_no_positive_returns_zero():
    """Disjoint single-label samples → no Jaccard positives → zero loss."""
    loss_fn = MultiLabelSupConLoss(temp=0.1)
    labels = torch.eye(3)
    feats = torch.randn(3, 8)
    assert loss_fn(feats, labels).item() == 0.0


def test_supcon_batch_lt_2_is_zero():
    loss_fn = MultiLabelSupConLoss()
    assert loss_fn(torch.randn(1, 8), torch.ones(1, 4)).item() == 0.0
