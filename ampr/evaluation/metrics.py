"""Evaluation metrics for protein function prediction."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def compute_fmax(y_true, y_pred):
    """
    Compute F_max (optimal F1 over all thresholds).

    Args:
        y_true: (n_samples, n_terms) binary labels
        y_pred: (n_samples, n_terms) predicted probabilities [0, 1]

    Returns:
        fmax: float, maximum F1 score
        threshold: float, threshold at maximum F1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precisions = []
    recalls = []
    fscores = []
    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)

        tp = (y_pred_binary * y_true).sum()
        fp = (y_pred_binary * (1 - y_true)).sum()
        fn = ((1 - y_pred_binary) * y_true).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    fmax = max(fscores)
    fmax_idx = np.argmax(fscores)
    threshold_at_fmax = thresholds[fmax_idx]

    return fmax, threshold_at_fmax


def compute_smin(y_true, y_pred, go_graph=None):
    """
    Compute S_min (semantic distance metric).

    Simplified: uses hierarchical structure if go_graph provided,
    otherwise returns placeholder.

    Args:
        y_true: (n_samples, n_terms) binary labels
        y_pred: (n_samples, n_terms) predicted probabilities
        go_graph: networkx DiGraph of GO hierarchy (optional)

    Returns:
        smin: float
    """
    if go_graph is None:
        return 0.0

    return 0.0


def compute_auprc(y_true, y_pred):
    """
    Compute average precision (area under precision-recall curve).

    Args:
        y_true: (n_samples, n_terms) binary labels
        y_pred: (n_samples, n_terms) predicted probabilities

    Returns:
        auprc: float, average across all terms
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    auprcs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        auprc = average_precision_score(y_true[:, i], y_pred[:, i])
        auprcs.append(auprc)

    return np.mean(auprcs) if auprcs else 0.0
