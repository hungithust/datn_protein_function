#!/usr/bin/env python
"""Generate one v3 config per sweep cell from a base config + grid.

Sweep axes (8 cells = 8 GPUs):
  classifier : {both, label_attn}
  d_hidden   : {512, 1024}
  go_emb     : {text+graph combined, text-only}

Usage:
  python scripts/gen_sweep_configs.py --base configs/mf_v3_esm3b.yaml \
    --branch mf --out_dir configs/sweep
"""
import argparse
import copy
import itertools
from pathlib import Path
import yaml


def set_dotted(cfg: dict, dotted: str, value):
    keys = dotted.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def expand_grid(base: dict, grid: dict):
    """Cartesian product over grid -> list of (name, config). base is not mutated."""
    keys = list(grid)
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = copy.deepcopy(base)
        tags = []
        for k, (tag, val) in zip(keys, combo):
            set_dotted(cfg, k, val)
            tags.append(tag)
        out.append(('_'.join(tags), cfg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--branch', required=True)
    ap.add_argument('--out_dir', default='configs/sweep')
    args = ap.parse_args()
    base = yaml.safe_load(open(args.base))
    b = args.branch
    grid = {
        'model.classifier': [('both', 'both'), ('la', 'label_attn')],
        'model.d_hidden':   [('h512', 512), ('h1024', 1024)],
        'data.go_emb':      [('comb', f'data/embeddings/go_emb_{b}_v2.npy'),
                             ('text', f'data/embeddings/go_text_{b}.npy')],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in expand_grid(base, grid):
        # 8 cells run concurrently on one node. Multiprocessing DataLoader workers
        # exhaust shm file descriptors (Bus error) even with file_system sharing.
        # Use 0 workers for the sweep: synchronous loading from precomputed h5 —
        # slower per cell but rock-solid under 8-way concurrency.
        set_dotted(cfg, 'training.num_workers', 0)
        tag = f"{b}_{name}"
        cfg['output']['checkpoint_dir'] = f"checkpoints/sweep_{tag}/"
        cfg['output']['log_file'] = f"logs/sweep_{tag}.log"
        cfg['output']['results_file'] = f"results/sweep_{tag}.tsv"
        path = out_dir / f"{tag}.yaml"
        yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)
        print(f"[SWEEP] wrote {path}")


if __name__ == '__main__':
    main()
