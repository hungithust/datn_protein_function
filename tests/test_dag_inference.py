import numpy as np
from ampr.evaluation.dag_inference import propagate_scores_upward


def test_propagation_enforces_true_path():
    # 3 terms: 0=root, 1=child of 0, 2=child of 1
    # dag[i,j]=1 nếu j là parent của i
    dag = np.zeros((3, 3))
    dag[1, 0] = 1.0
    dag[2, 1] = 1.0
    probs = np.array([[0.1, 0.2, 0.9]])
    out = propagate_scores_upward(probs, dag)
    # term 0 phải >= max(child 1) >= max(child 2)
    assert out[0, 1] >= 0.9
    assert out[0, 0] >= 0.9


def test_no_edges_returns_copy():
    dag = np.zeros((4, 4))
    probs = np.random.rand(2, 4)
    out = propagate_scores_upward(probs, dag)
    assert np.allclose(out, probs)
    assert out is not probs
