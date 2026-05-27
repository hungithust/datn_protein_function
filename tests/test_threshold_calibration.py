import json
import numpy as np
from pathlib import Path
from ampr.evaluation.threshold_calibration import find_optimal_threshold, calibrate_and_save


def test_find_optimal_threshold_finds_perfect_split():
    y = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    p = np.array([[0.8, 0.1], [0.7, 0.2], [0.2, 0.9]], dtype=np.float32)
    t, fmax = find_optimal_threshold(y, p, thresholds=np.array([0.3, 0.5, 0.75]))
    assert 0.3 <= t <= 0.75
    assert fmax > 0.9


def test_calibrate_and_save_writes_json(tmp_path):
    y = np.array([[1, 0], [0, 1]], dtype=np.float32)
    p = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    out = tmp_path / 'thr.json'
    res = calibrate_and_save(p, y, branch='MF', output_path=str(out))
    assert out.exists()
    d = json.loads(out.read_text())
    assert d['branch'] == 'MF' and 0 < d['threshold'] < 1 and d['val_fmax'] > 0.9
    assert res['threshold'] == d['threshold']
