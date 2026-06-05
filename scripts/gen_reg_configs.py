#!/usr/bin/env python
"""Generate regularization-sweep configs (weight_decay x dropout) for a branch.

Probes whether stronger regularization shrinks the val->test gap. Checkpoints and
logs go to dedicated `*_reg` paths so they can be archived without re-downloading
the original `checkpoints/` tree.

Grid (4 cells): weight_decay {1e-4, 1e-2} x dropout {0.2, 0.3}.
Baseline (wd=0, dropout=0.1) is the existing main run — compare against it.

Usage:
  python scripts/gen_reg_configs.py --base configs/mf_v3_esm3b.yaml --branch mf
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_sweep_configs import expand_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--branch', required=True)
    ap.add_argument('--out_dir', default='configs/sweep_reg')
    args = ap.parse_args()
    base = yaml.safe_load(open(args.base))
    b = args.branch
    grid = {
        'training.weight_decay': [('wd1e4', 1.0e-4), ('wd1e2', 1.0e-2)],
        'model.seq.dropout':     [('do2', 0.2), ('do3', 0.3)],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in expand_grid(base, grid):
        tag = f"{b}_reg_{name}"
        cfg['output']['checkpoint_dir'] = f"checkpoints_reg/{tag}/"
        cfg['output']['log_file'] = f"logs/{tag}.log"
        cfg['output']['results_file'] = f"results/{tag}_predictions.tsv"
        cfg['output']['threshold_path'] = f"checkpoints_reg/{tag}/threshold.json"
        path = out_dir / f"{tag}.yaml"
        yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)
        print(f"[REG] wrote {path}")


if __name__ == '__main__':
    main()
