#!/usr/bin/env python
"""
Run sequence-identity-stratified evaluation for AMPR — DeepFRI-style.

Loads the best checkpoint of each branch and evaluates on multiple test
subsets (test_LT_30, test_LT_40, test_LT_50, test_LT_70, test_LT_95) in
splits_stratified.json. Saves per-split metrics + predictions and produces
identity-stratified comparison plots vs DeepFRI baseline.

Usage:
    python scripts/evaluate_stratified.py \
        --branches mf,bp,cc \
        --splits test_LT_30,test_LT_40,test_LT_50,test_LT_70,test_LT_95,test \
        --configs-dir configs \
        --baseline data/deepfri_baseline.json \
        --kaggle-suffix _kaggle
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def setup_logger():
    log = logging.getLogger('eval_strat')
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    log.addHandler(h)
    return log


def run_one_eval(config_path, checkpoint, split, log):
    """Invoke main.py --eval-only for one (config, checkpoint, split) triple."""
    cmd = [
        sys.executable, 'main.py',
        '--config', str(config_path),
        '--eval-only',
        '--checkpoint', str(checkpoint),
        '--test-split', split,
    ]
    log.info(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def load_metrics(results_dir, branch, split):
    """Load test_metrics_{branch}{_LT_xx}.json produced by trainer."""
    suffix = '' if split in ('test', 'test_'  ) else split.replace('test_', '_')
    if split == 'test':
        path = Path(results_dir) / f'test_metrics_{branch}.json'
    else:
        s = split.replace('test_', '')
        path = Path(results_dir) / f'test_metrics_{branch}_{s}.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description='Stratified DeepFRI-style evaluation')
    ap.add_argument('--branches', default='mf,bp,cc',
                    help='Comma-separated branches')
    ap.add_argument('--splits',
                    default='test_LT_30,test_LT_40,test_LT_50,test_LT_70,test_LT_95,test',
                    help='Comma-separated split names from splits_stratified.json')
    ap.add_argument('--configs-dir', default='configs')
    ap.add_argument('--kaggle-suffix', default='_kaggle',
                    help='Suffix for Kaggle-flavored configs (e.g. mf_kaggle.yaml)')
    ap.add_argument('--baseline', default='data/deepfri_baseline.json',
                    help='Path to DeepFRI baseline metrics (for comparison plots)')
    ap.add_argument('--results-dir', default=None,
                    help='Override results dir (default: read from config)')
    ap.add_argument('--skip-eval', action='store_true',
                    help='Skip running eval; just regenerate plots from existing JSONs')
    args = ap.parse_args()

    log = setup_logger()
    branches = [b.strip().lower() for b in args.branches.split(',')]
    splits = [s.strip() for s in args.splits.split(',')]

    # Resolve config paths and per-branch checkpoint/results dirs
    branch_info = {}
    for branch in branches:
        cfg_path = Path(args.configs_dir) / f'{branch}{args.kaggle_suffix}.yaml'
        if not cfg_path.exists():
            cfg_path = Path(args.configs_dir) / f'{branch}.yaml'
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        ckpt = Path(cfg['output']['checkpoint_dir']) / 'best.pt'
        results = Path(args.results_dir) if args.results_dir else \
                  Path(cfg['output']['results_file']).parent

        branch_info[branch] = {
            'config': cfg_path,
            'checkpoint': ckpt,
            'results_dir': results,
        }
        log.info(f"[{branch.upper()}] config={cfg_path} ckpt={ckpt} results={results}")

    # ── Run evaluations ───────────────────────────────────────────────────
    if not args.skip_eval:
        for branch in branches:
            info = branch_info[branch]
            if not info['checkpoint'].exists():
                log.warning(f"[{branch.upper()}] checkpoint missing — skipping all splits")
                continue
            for split in splits:
                ok = run_one_eval(info['config'], info['checkpoint'], split, log)
                if not ok:
                    log.warning(f"[{branch.upper()}] eval failed for split={split}")

    # ── Aggregate metrics across (branch, split) ──────────────────────────
    all_metrics = {}  # {branch: {LT_xx: {fmax, ...}}}
    for branch in branches:
        results_dir = branch_info[branch]['results_dir']
        per_split = {}
        for split in splits:
            m = load_metrics(results_dir, branch, split)
            if m is None:
                continue
            key = 'test' if split == 'test' else split.replace('test_', '')
            per_split[key] = m
        all_metrics[branch] = per_split

    # ── Print aggregated table ────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("STRATIFIED EVALUATION SUMMARY")
    log.info("=" * 80)
    cols = ['fmax', 'auprc_micro', 'auprc_macro', 'smin', 'coverage']
    head = ['Fmax', 'AUPRm', 'AUPRM', 'Smin↓', 'Cov']
    log.info(f"{'Branch':>6} | {'Split':>8} | {'N':>5} | " +
             " | ".join(f"{h:>7}" for h in head))
    log.info("-" * 80)
    for branch in branches:
        for split_key, m in all_metrics[branch].items():
            n = m.get('n_proteins', 0)
            vals = " | ".join(f"{m.get(c, 0.0):>7.4f}" for c in cols)
            log.info(f"{branch.upper():>6} | {split_key:>8} | {n:>5} | {vals}")
        log.info("-" * 80)

    # ── Save aggregated JSON ──────────────────────────────────────────────
    agg_path = branch_info[branches[0]]['results_dir'] / 'stratified_summary.json'
    with open(agg_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"[SUMMARY] → {agg_path}")

    # ── Identity-stratified plots + DeepFRI comparison ───────────────────
    deepfri = None
    if Path(args.baseline).exists():
        with open(args.baseline) as f:
            deepfri = json.load(f)
        log.info(f"[BASELINE] DeepFRI loaded from {args.baseline}")

    try:
        from ampr.evaluation.plots import (
            plot_identity_stratified, plot_deepfri_comparison
        )
        plots_dir = branch_info[branches[0]]['results_dir'] / 'plots'
        for branch in branches:
            if all_metrics[branch]:
                plot_identity_stratified(
                    all_metrics[branch], str(plots_dir), branch,
                    deepfri_baseline=deepfri,
                )
        if deepfri is not None:
            plot_deepfri_comparison(all_metrics, deepfri, str(plots_dir))
    except Exception as exc:
        log.warning(f"[PLOT] Failed: {exc}")

    log.info("[DONE] Stratified evaluation complete.")


if __name__ == '__main__':
    main()
