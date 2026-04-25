"""Visualization utilities for AMPR training and evaluation results."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc as sk_auc

from ampr.evaluation.metrics import compute_fmax


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[PLOT] → {path}')


def plot_training_curves(history, save_dir, branch):
    """Loss components and val Fmax over training epochs."""
    epochs   = [h['epoch']     for h in history]
    bce      = [h['bce']       for h in history]
    dag      = [h['dag']       for h in history]
    total    = [h['train_loss'] for h in history]
    val_fmax = [h['val_fmax']  for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'Training — {branch.upper()}', fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, total, color='#2c3e50', linewidth=2, label='Total')
    ax.plot(epochs, bce,   color='#e74c3c', linestyle='--', label='BCE')
    ax.plot(epochs, dag,   color='#e67e22', linestyle=':', label='DAG')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    best_idx = int(np.argmax(val_fmax))
    ax.plot(epochs, val_fmax, color='#27ae60', linewidth=2)
    ax.axvline(epochs[best_idx], color='#c0392b', linestyle=':', linewidth=1.5,
               label=f'Best epoch {epochs[best_idx]}')
    ax.scatter([epochs[best_idx]], [val_fmax[best_idx]],
               color='#c0392b', zorder=5, s=60)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Fmax')
    ax.set_title(f'Validation Fmax  (best = {max(val_fmax):.4f})')
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, f'{save_dir}/training_curves_{branch}.png')


def plot_alpha_evolution(history, save_dir, branch):
    """Stacked area chart of gating weights α_seq, α_3di, α_ppi over epochs."""
    epochs = [h['epoch'] for h in history]
    alphas = [h.get('alphas', [1/3, 1/3, 1/3]) for h in history]

    a_seq = [a[0] for a in alphas]
    a_3di = [a[1] for a in alphas]
    a_ppi = [a[2] for a in alphas]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        epochs, a_seq, a_3di, a_ppi,
        labels=['α_seq (ProteinBERT)', 'α_3di (ProstT5)', 'α_ppi (DeepGO)'],
        colors=['#3498db', '#e74c3c', '#2ecc71'],
        alpha=0.85,
    )
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gating weight (α)')
    ax.set_title(f'Modality Gating Weights — {branch.upper()}')
    ax.legend(loc='upper right'); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    _save(fig, f'{save_dir}/alpha_evolution_{branch}.png')


def plot_pr_curve(y_true, y_pred, save_dir, branch):
    """Micro-averaged Precision-Recall curve with Fmax point marked."""
    # Use only terms that have positive samples
    valid = y_true.sum(axis=0) > 0
    y_t = y_true[:, valid].ravel()
    y_p = y_pred[:, valid].ravel()

    precision, recall, thresholds = precision_recall_curve(y_t, y_p)
    pr_auc = sk_auc(recall, precision)

    fmax, fmax_t = compute_fmax(y_true, y_pred)
    # Locate the PR curve point closest to fmax_threshold
    if len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - fmax_t)))
        fmax_pr, fmax_rc = float(precision[idx]), float(recall[idx])
    else:
        fmax_pr = fmax_rc = 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#2980b9', linewidth=2,
            label=f'Micro-avg PR  (AUPRC = {pr_auc:.4f})')
    ax.scatter([fmax_rc], [fmax_pr], color='#e74c3c', s=100, zorder=5,
               label=f'Fmax = {fmax:.4f}  @ t = {fmax_t:.2f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {branch.upper()}')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    _save(fig, f'{save_dir}/pr_curve_{branch}.png')


def plot_roc_curve(y_true, y_pred, save_dir, branch):
    """Micro-averaged ROC curve."""
    valid = [i for i in range(y_true.shape[1])
             if 0 < y_true[:, i].sum() < len(y_true)]
    if not valid:
        return

    y_t = y_true[:, valid].ravel()
    y_p = y_pred[:, valid].ravel()

    fpr, tpr, _ = roc_curve(y_t, y_p)
    roc_auc = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='#8e44ad', linewidth=2,
            label=f'Micro ROC  (AUROC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {branch.upper()}')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    _save(fig, f'{save_dir}/roc_curve_{branch}.png')


def plot_threshold_sweep(y_true, y_pred, save_dir, branch):
    """Precision, Recall, F1 vs classification threshold (CAFA macro-avg)."""
    has_annot = y_true.sum(axis=1) > 0
    yt = y_true[has_annot].astype(np.float32)
    yp = y_pred[has_annot]

    thresholds = np.linspace(0.0, 1.0, 101)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        pred = (yp >= t).astype(np.float32)
        tp = (pred * yt).sum(axis=1)
        fp = (pred * (1 - yt)).sum(axis=1)
        fn = ((1 - pred) * yt).sum(axis=1)

        pr = np.where(tp + fp > 0, tp / (tp + fp), 0.0).mean()
        rc = (tp / (tp + fn)).mean()
        f  = (2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0

        precisions.append(pr); recalls.append(rc); f1s.append(f)

    best_idx = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, color='#3498db', label='Precision')
    ax.plot(thresholds, recalls,    color='#e74c3c', label='Recall')
    ax.plot(thresholds, f1s,        color='#27ae60', linewidth=2, label='F1 (→ Fmax)')
    ax.axvline(thresholds[best_idx], color='gray', linestyle='--', linewidth=1.2,
               label=f'Fmax = {f1s[best_idx]:.4f} @ t = {thresholds[best_idx]:.2f}')
    ax.set_xlabel('Classification Threshold'); ax.set_ylabel('Score')
    ax.set_title(f'Threshold Sweep — {branch.upper()}')
    ax.legend(); ax.grid(alpha=0.3)

    _save(fig, f'{save_dir}/threshold_sweep_{branch}.png')


def generate_all_plots(history, y_true, y_pred, save_dir, branch):
    """Generate all plots for one branch. Call after training completes."""
    plot_training_curves(history, save_dir, branch)
    plot_alpha_evolution(history, save_dir, branch)
    plot_pr_curve(y_true, y_pred, save_dir, branch)
    plot_roc_curve(y_true, y_pred, save_dir, branch)
    plot_threshold_sweep(y_true, y_pred, save_dir, branch)