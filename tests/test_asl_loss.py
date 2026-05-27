# tests/test_asl_loss.py
import torch
from ampr.training.loss import AsymmetricLoss, AMPRLoss


def test_asl_basic_shape():
    torch.manual_seed(42)
    asl = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    logits = torch.randn(4, 10, requires_grad=True)
    labels = torch.randint(0, 2, (4, 10)).float()
    loss = asl(logits, labels)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None


def test_asl_downweights_easy_negatives():
    """Logit rất âm cho label 0 → loss gần 0 hơn so với BCE."""
    asl = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    logits = torch.full((1, 1), -10.0)  # very confident negative
    labels = torch.zeros(1, 1)
    assert asl(logits, labels).item() < bce(logits, labels).item() * 0.5


def test_amprloss_asl_mode():
    torch.manual_seed(42)
    dag = torch.zeros(5, 5)
    dag[1, 0] = 1.0
    loss_fn = AMPRLoss(dag, lambda_dag=0.5, loss_type='asl',
                       asl_gamma_neg=4, asl_gamma_pos=0, asl_clip=0.05)
    logits = torch.randn(3, 5, requires_grad=True)
    labels = torch.tensor([[1, 1, 0, 0, 1],
                           [0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 1]], dtype=torch.float)
    loss, parts = loss_fn(logits, labels)
    assert 'cls' in parts and 'dag' in parts
    loss.backward()
    assert logits.grad is not None
