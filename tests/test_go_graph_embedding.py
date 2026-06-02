import numpy as np
from scripts.precompute_go_graph import graph_embedding


def test_graph_embedding_shape_and_determinism():
    rng = np.random.default_rng(0)
    n = 20
    dag = (rng.random((n, n)) < 0.2).astype(np.float32)
    np.fill_diagonal(dag, 0)
    e1 = graph_embedding(dag, dim=8)
    e2 = graph_embedding(dag, dim=8)
    assert e1.shape == (n, 8)
    assert np.allclose(e1, e2)            # deterministic


def test_graph_embedding_dim_caps_at_rank():
    dag = np.zeros((5, 5), dtype=np.float32)
    dag[1, 0] = dag[2, 0] = 1.0          # tiny graph
    e = graph_embedding(dag, dim=16)     # dim > nodes
    assert e.shape == (5, 16)            # zero-padded, no crash
