"""Evaluation metrics for protein function prediction (CAFA standard)."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_fmax(y_true, y_pred):
    """
    F_max using CAFA per-protein protocol.

    For each threshold t:
        per-protein precision & recall → macro-average → F(t)
    F_max = max over all t.

    Returns:
        fmax:      float
        threshold: float at which F_max is achieved
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    has_annot = y_true.sum(axis=1) > 0
    y_true = y_true[has_annot]
    y_pred = y_pred[has_annot]

    if len(y_true) == 0:
        return 0.0, 0.0

    thresholds = np.linspace(0.0, 1.0, 101)
    best_fmax = 0.0
    best_t = 0.0

    for t in thresholds:
        pred_bin = (y_pred >= t).astype(np.float32)
        tp = (pred_bin * y_true).sum(axis=1)
        fp = (pred_bin * (1 - y_true)).sum(axis=1)
        fn = ((1 - pred_bin) * y_true).sum(axis=1)

        denom_pr = tp + fp
        pr = np.where(denom_pr > 0, tp / denom_pr, 0.0)
        rc = tp / (tp + fn)  # always defined (has_annot ensures fn+tp > 0)

        avg_pr = pr.mean()
        avg_rc = rc.mean()
        denom_f = avg_pr + avg_rc
        f = (2 * avg_pr * avg_rc / denom_f) if denom_f > 0 else 0.0

        if f > best_fmax:
            best_fmax = f
            best_t = t

    return float(best_fmax), float(best_t)


def compute_auprc(y_true, y_pred):
    """
    Mean AUPRC across GO terms that have ≥1 positive sample.
    Equivalent to macro-averaged AUPR.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = [
        average_precision_score(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
        if y_true[:, i].sum() > 0
    ]
    return float(np.mean(scores)) if scores else 0.0


def compute_smin(y_true, y_pred, term_ic):
    """
    S_min (semantic distance, lower is better).

    Uses annotation-based information content:
        IC(t) = -log2(freq(t))  where freq(t) = fraction of proteins annotated with t.

    For each threshold t:
        misinformation (mi) = mean per-protein sum of IC for false-positive terms
        remaining uncertainty (ru) = mean per-protein sum of IC for false-negative terms
        S(t) = sqrt(mi² + ru²)
    S_min = min over all t.

    Args:
        y_true:  (n_samples, n_terms) binary
        y_pred:  (n_samples, n_terms) probabilities
        term_ic: (n_terms,) information content per term
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    term_ic = np.asarray(term_ic, dtype=np.float32)

    has_annot = y_true.sum(axis=1) > 0
    y_true = y_true[has_annot]
    y_pred = y_pred[has_annot]

    if len(y_true) == 0:
        return 0.0

    thresholds = np.linspace(0.0, 1.0, 51)
    best_smin = np.inf

    for t in thresholds:
        pred_bin = (y_pred >= t).astype(np.float32)
        fp = pred_bin * (1 - y_true)
        fn = (1 - pred_bin) * y_true

        mi = (fp * term_ic).sum(axis=1).mean()
        ru = (fn * term_ic).sum(axis=1).mean()
        s = np.sqrt(mi ** 2 + ru ** 2)

        if s < best_smin:
            best_smin = s

    return float(best_smin)


def compute_auroc(y_true, y_pred):
    """
    Micro and macro AUROC across GO terms with both positive and negative samples.

    Returns:
        (micro_auroc, macro_auroc)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = [
        i for i in range(y_true.shape[1])
        if 0 < y_true[:, i].sum() < len(y_true)
    ]
    if not valid:
        return 0.0, 0.0

    y_t = y_true[:, valid]
    y_p = y_pred[:, valid]

    try:
        micro = float(roc_auc_score(y_t, y_p, average='micro'))
        macro = float(roc_auc_score(y_t, y_p, average='macro'))
    except ValueError:
        micro = macro = 0.0

    return micro, macro


def compute_coverage(y_true, y_pred, threshold):
    """Fraction of proteins that have ≥1 predicted term at the given threshold."""
    pred_bin = np.asarray(y_pred) >= threshold
    return float(pred_bin.any(axis=1).mean())


def compute_all_metrics(y_true, y_pred, term_ic):
    """
    Full CAFA-style evaluation suite.

    Returns dict with:
        fmax, fmax_threshold, auprc, smin, micro_auroc, macro_auroc, coverage
    """
    fmax, threshold = compute_fmax(y_true, y_pred)
    auprc = compute_auprc(y_true, y_pred)
    smin = compute_smin(y_true, y_pred, term_ic)
    micro_auroc, macro_auroc = compute_auroc(y_true, y_pred)
    coverage = compute_coverage(y_true, y_pred, threshold)

    return {
        'fmax': fmax,
        'fmax_threshold': threshold,
        'auprc': auprc,
        'smin': smin,
        'micro_auroc': micro_auroc,
        'macro_auroc': macro_auroc,
        'coverage': coverage,
    }
