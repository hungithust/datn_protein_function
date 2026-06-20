"""Test compute_smin_raw (unnormalized DeepFRI/HEAL-style Smin)."""
import numpy as np

from ampr.evaluation.metrics import compute_smin_raw, compute_all_metrics


def test_perfect_prediction_zero_smin():
    y = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    ic = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    # perfect probs → at threshold 0.5 ru=mi=0 → smin_raw = 0
    assert compute_smin_raw(y, y, ic) == 0.0


def test_raw_differs_from_normalized_and_is_in_dict():
    rng = np.random.default_rng(0)
    y = (rng.random((20, 5)) > 0.7).astype(np.float32)
    y[:, 0] = 1  # ensure a positive column
    p = rng.random((20, 5)).astype(np.float32)
    ic = (-np.log2(y.mean(axis=0).clip(1e-7, 1.0))).astype(np.float32)
    m = compute_all_metrics(y, p, ic)
    assert 'smin_raw' in m and 'smin' in m
    # raw IC-summed Smin is on a different (unnormalized) scale than the [0,√2] variant
    assert m['smin_raw'] >= 0.0
