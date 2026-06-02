import numpy as np
from scripts.build_go_combined import l2norm_concat


def test_l2norm_concat_shape_and_unit_blocks():
    text = np.random.randn(4, 6).astype(np.float32)
    graph = np.random.randn(4, 3).astype(np.float32)
    out = l2norm_concat(text, graph)
    assert out.shape == (4, 9)
    # each block L2-normalized per row
    assert np.allclose(np.linalg.norm(out[:, :6], axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(out[:, 6:], axis=1), 1.0, atol=1e-5)


def test_l2norm_concat_row_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        l2norm_concat(np.zeros((4, 6), np.float32), np.zeros((3, 3), np.float32))
