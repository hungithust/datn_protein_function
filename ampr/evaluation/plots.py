"""Visualization utilities for AMPR training and evaluation results."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc as sk_auc

from ampr.evaluation.metrics import compute_fmax


# ─── helpers ─────────────────────────────────────────────────────────────────

def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[PLOT] → {path}')


# ─── training-time plots (require history) ───────────────────────────────────

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


# ─── evaluation plots (only need predictions) ────────────────────────────────

def plot_pr_curve(y_true, y_pred, save_dir, branch, suffix=''):
    """Micro-averaged PR curve with Fmax point marked."""
    valid = y_true.sum(axis=0) > 0
    y_t = y_true[:, valid].ravel()
    y_p = y_pred[:, valid].ravel()

    precision, recall, thresholds = precision_recall_curve(y_t, y_p)
    pr_auc = sk_auc(recall, precision)

    fmax, fmax_t = compute_fmax(y_true, y_pred)
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
    title_suffix = f'  ({suffix})' if suffix else ''
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {branch.upper()}{title_suffix}')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fname = f'pr_curve_{branch}{("_" + suffix) if suffix else ""}.png'
    _save(fig, f'{save_dir}/{fname}')


def plot_roc_curve(y_true, y_pred, save_dir, branch, suffix=''):
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
    title_suffix = f'  ({suffix})' if suffix else ''
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {branch.upper()}{title_suffix}')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fname = f'roc_curve_{branch}{("_" + suffix) if suffix else ""}.png'
    _save(fig, f'{save_dir}/{fname}')


def plot_threshold_sweep(y_true, y_pred, save_dir, branch, suffix=''):
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
    title_suffix = f'  ({suffix})' if suffix else ''
    ax.set_xlabel('Classification Threshold'); ax.set_ylabel('Score')
    ax.set_title(f'Threshold Sweep — {branch.upper()}{title_suffix}')
    ax.legend(); ax.grid(alpha=0.3)

    fname = f'threshold_sweep_{branch}{("_" + suffix) if suffix else ""}.png'
    _save(fig, f'{save_dir}/{fname}')


# ─── DeepFRI comparison plots ────────────────────────────────────────────────

def plot_identity_stratified(metrics_per_split, save_dir, branch, deepfri_baseline=None):
    """
    Line chart: Fmax + AUPR_micro vs sequence-identity threshold.

    metrics_per_split: dict { 'LT_30': {fmax, auprc_micro, smin, ...},
                              'LT_40': {...}, ... }
    deepfri_baseline:  optional dict same structure for overlay
    """
    thresholds = ['LT_30', 'LT_40', 'LT_50', 'LT_70', 'LT_95']
    x_labels = ['<30%', '<40%', '<50%', '<70%', '<95%']

    def gather(metric_key, source):
        return [source.get(t, {}).get(metric_key, np.nan) for t in thresholds]

    ampr_fmax  = gather('fmax',         metrics_per_split)
    ampr_aupr  = gather('auprc_micro',  metrics_per_split)
    ampr_smin  = gather('smin',         metrics_per_split)

    n_panels = 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    fig.suptitle(f'Identity-Stratified Performance — {branch.upper()}',
                 fontweight='bold')

    panels = [
        (axes[0], 'Fmax ↑',     ampr_fmax, 'fmax',        '#27ae60'),
        (axes[1], 'AUPR_micro ↑', ampr_aupr, 'auprc_micro', '#2980b9'),
        (axes[2], 'Smin ↓',     ampr_smin, 'smin',        '#c0392b'),
    ]
    for ax, title, ampr_vals, key, color in panels:
        ax.plot(x_labels, ampr_vals, marker='o', linewidth=2,
                color=color, label='AMPR (ours)')
        if deepfri_baseline is not None:
            df_vals = gather(key, deepfri_baseline.get(branch.upper(), {}))
            if any(not np.isnan(v) for v in df_vals):
                ax.plot(x_labels, df_vals, marker='s', linewidth=2,
                        linestyle='--', color='gray', label='DeepFRI')
        ax.set_xlabel('Sequence identity to training set')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    _save(fig, f'{save_dir}/identity_stratified_{branch}.png')


def plot_deepfri_comparison(all_metrics, deepfri_baseline, save_dir):
    """
    Grouped bar chart: AMPR vs DeepFRI for each (branch, threshold).
    Shows Fmax and AUPR_micro side-by-side.

    all_metrics: { 'mf': {'LT_30': {...}, ...}, 'bp': {...}, 'cc': {...} }
    """
    thresholds = ['LT_30', 'LT_40', 'LT_50', 'LT_70', 'LT_95']
    branches = ['mf', 'bp', 'cc']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('AMPR vs DeepFRI — Identity-Stratified Test Performance',
                 fontweight='bold', fontsize=13)

    for col, branch in enumerate(branches):
        for row, (key, ylabel) in enumerate([('fmax', 'Fmax'),
                                             ('auprc_micro', 'AUPR (micro)')]):
            ax = axes[row, col]
            x = np.arange(len(thresholds))
            width = 0.4

            ampr_vals = [all_metrics.get(branch, {}).get(t, {}).get(key, 0)
                         for t in thresholds]
            df_vals   = [deepfri_baseline.get(branch.upper(), {}).get(t, {}).get(key, 0)
                         for t in thresholds]

            ax.bar(x - width/2, ampr_vals, width,
                   label='AMPR (ours)', color='#3498db')
            ax.bar(x + width/2, df_vals,   width,
                   label='DeepFRI',      color='#95a5a6')

            ax.set_xticks(x)
            ax.set_xticklabels(['<30', '<40', '<50', '<70', '<95'])
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f'{branch.upper()}  ({ylabel})', fontweight='bold')
            else:
                ax.set_title(f'{branch.upper()}  ({ylabel})')
            ax.grid(alpha=0.3, axis='y')
            if col == 0 and row == 0:
                ax.legend(loc='upper left')

    fig.tight_layout()
    _save(fig, f'{save_dir}/deepfri_comparison.png')


# ─── orchestration ───────────────────────────────────────────────────────────

def generate_all_plots(history, y_true, y_pred, save_dir, branch, suffix=''):
    """Generate evaluation plots. Skip training/alpha if history is None/empty."""
    if history:
        plot_training_curves(history, save_dir, branch)
        plot_alpha_evolution(history, save_dir, branch)
    plot_pr_curve(y_true, y_pred, save_dir, branch, suffix=suffix)
    plot_roc_curve(y_true, y_pred, save_dir, branch, suffix=suffix)
    plot_threshold_sweep(y_true, y_pred, save_dir, branch, suffix=suffix)
