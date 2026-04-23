"""Evaluation metrics for protein function prediction (CAFA standard)."""

import numpy as np
from sklearn.metrics import average_precision_score


def compute_fmax(y_true, y_pred):
    """
    Compute F_max using the CAFA per-protein protocol.

    For each threshold t:
        - Per protein: pr_i(t) = TP_i / (TP_i + FP_i), rc_i(t) = TP_i / (TP_i + FN_i)
        - Average precision/recall over all proteins that have ≥1 positive label
        - F(t) = 2 * avg_pr * avg_rc / (avg_pr + avg_rc)
    F_max = max over all t.

    Reference: CAFA3 evaluation (Zhou et al., 2019, Genome Biology).

    Args:
        y_true: (n_samples, n_terms) binary int/float array
        y_pred: (n_samples, n_terms) float array in [0, 1]

    Returns:
        fmax:      float
        threshold: float at which F_max is achieved
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # Only evaluate proteins that have at least one positive annotation
    has_annot = y_true.sum(axis=1) > 0
    y_true = y_true[has_annot]
    y_pred = y_pred[has_annot]

    if len(y_true) == 0:
        return 0.0, 0.0

    thresholds = np.linspace(0.0, 1.0, 101)
    best_fmax = 0.0
    best_t    = 0.0

    for t in thresholds:
        pred_bin = (y_pred >= t).astype(np.float32)

        tp = (pred_bin * y_true).sum(axis=1)          # (n_proteins,)
        fp = (pred_bin * (1 - y_true)).sum(axis=1)
        fn = ((1 - pred_bin) * y_true).sum(axis=1)

        # Per-protein precision (undefined when no predictions → set to 0)
        denom_pr = tp + fp
        pr = np.where(denom_pr > 0, tp / denom_pr, 0.0)

        # Per-protein recall (always defined since has_annot filtered empties)
        denom_rc = tp + fn
        rc = np.where(denom_rc > 0, tp / denom_rc, 0.0)

        avg_pr = pr.mean()
        avg_rc = rc.mean()

        denom_f = avg_pr + avg_rc
        f = (2 * avg_pr * avg_rc / denom_f) if denom_f > 0 else 0.0

        if f > best_fmax:
            best_fmax = f
            best_t    = t

    return float(best_fmax), float(best_t)


def compute_auprc(y_true, y_pred):
    """
    Mean AUPRC across GO terms that have at least one positive sample.

    Args:
        y_true: (n_samples, n_terms) binary
        y_pred: (n_samples, n_terms) probabilities

    Returns:
        auprc: float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        scores.append(average_precision_score(y_true[:, i], y_pred[:, i]))

    return float(np.mean(scores)) if scores else 0.0


def compute_smin(_y_true, _y_pred, _go_graph=None):
    """
    S_min (semantic distance, lower is better).

    Full implementation requires GO graph with information content per term.
    Returns placeholder 0.0 until go_graph integration is added.

    Args:
        y_true: (n_samples, n_terms)
        y_pred: (n_samples, n_terms)
        go_graph: networkx DiGraph (optional)

    Returns:
        smin: float
    """
    return 0.0
