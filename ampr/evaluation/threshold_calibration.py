"""Per-branch threshold calibration on validation set."""

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from ampr.evaluation.metrics import compute_fmax


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray,
                           thresholds: np.ndarray | None = None) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        # compute F at specific t by hand to avoid double sweep
        preds = (y_probs >= t).astype(np.float32)
        tp = (preds * y_true).sum(axis=1)
        fp = (preds * (1 - y_true)).sum(axis=1)
        fn = ((1 - preds) * y_true).sum(axis=1)
        precision = np.where(tp + fp > 0, tp / (tp + fp + 1e-12), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn + 1e-12), 0.0)
        denom = precision + recall
        f = np.where(denom > 0, 2 * precision * recall / (denom + 1e-12), 0.0).mean()
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t, best_f


def calibrate_and_save(val_probs: np.ndarray, val_labels: np.ndarray,
                       branch: str, output_path: str) -> dict:
    t, f = find_optimal_threshold(val_labels, val_probs)
    payload = {
        'branch': branch,
        'threshold': t,
        'val_fmax': f,
        'calibration_date': _dt.datetime.utcnow().isoformat() + 'Z',
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2))
    return payload
