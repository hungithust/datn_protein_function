"""Tests for baselines/diamond/score.py — synthetic search results."""

import numpy as np


def test_compute_diamond_scores_picks_max_homolog():
    from baselines.diamond.score import compute_diamond_scores
    # Test: q1 has 2 hits — h1 (60% pident, labels [1,0]), h2 (80% pident, labels [0,1])
    # Expected: score[q1] = [0.6, 0.8]
    rows = [("q1", "h1", 60.0), ("q1", "h2", 80.0)]
    train_labels = np.array([[1, 0], [0, 1]], dtype=np.float32)
    train_idx = {"h1": 0, "h2": 1}
    test_ids = ["q1"]
    scores = compute_diamond_scores(rows, train_labels, train_idx, test_ids, n_terms=2)
    assert scores.shape == (1, 2)
    np.testing.assert_allclose(scores[0], [0.6, 0.8], atol=1e-5)


def test_compute_diamond_scores_zero_when_no_hits():
    from baselines.diamond.score import compute_diamond_scores
    scores = compute_diamond_scores([], np.zeros((0, 5)), {}, ["q1"], n_terms=5)
    assert scores.shape == (1, 5)
    assert scores.sum() == 0.0
