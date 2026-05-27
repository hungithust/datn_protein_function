"""DiamondScore ensemble (DeepGOPlus-style sequence homology)."""

from pathlib import Path

import numpy as np

from ampr.evaluation.dag_inference import propagate_scores_upward
from ampr.evaluation.threshold_calibration import find_optimal_threshold


def compute_diamond_scores(diamond_results_path: str, train_labels: np.ndarray,
                           train_protein_order: dict, test_protein_ids: list,
                           n_terms: int) -> np.ndarray:
    n_test = len(test_protein_ids)
    scores = np.zeros((n_test, n_terms), dtype=np.float32)
    test_idx = {p: i for i, p in enumerate(test_protein_ids)}
    with open(diamond_results_path) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            q, s, pident = parts[0], parts[1], float(parts[2])
            if q not in test_idx or s not in train_protein_order:
                continue
            sim = pident / 100.0
            contrib = sim * train_labels[train_protein_order[s]]
            i = test_idx[q]
            np.maximum(scores[i], contrib, out=scores[i])
    return scores


def ensemble_scores(model_probs: np.ndarray, diamond_probs: np.ndarray,
                    alpha: float = 0.6) -> np.ndarray:
    has_hom = diamond_probs.sum(axis=1) > 0
    out = alpha * model_probs + (1.0 - alpha) * diamond_probs
    out[~has_hom] = model_probs[~has_hom]
    return out.astype(model_probs.dtype)


def tune_alpha(val_model_probs: np.ndarray, val_diamond_probs: np.ndarray,
               val_labels: np.ndarray, dag_matrix: np.ndarray,
               alpha_range: np.ndarray | None = None) -> float:
    if alpha_range is None:
        alpha_range = np.arange(0.3, 0.91, 0.05)
    best_a, best_f = 0.6, -1.0
    for a in alpha_range:
        ens = ensemble_scores(val_model_probs, val_diamond_probs, a)
        if dag_matrix.sum() > 0:
            ens = propagate_scores_upward(ens, dag_matrix)
        _, f = find_optimal_threshold(val_labels, ens)
        if f > best_f:
            best_f, best_a = f, float(a)
    return best_a
