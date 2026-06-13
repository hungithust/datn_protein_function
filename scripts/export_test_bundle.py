#!/usr/bin/env python
"""Export a self-contained TEST-ONLY bundle for later re-evaluation.

Subsets the two large HDF5 stores (ESM-2 residue embeddings, contact maps) to only
the proteins appearing in any `test_*` split, and copies the small data files
(labels, DAG, GO embeddings, PPI, DIAMOND, splits, protein order) verbatim — keeping
the SAME relative paths so the existing configs run unchanged from the bundle root
(they will only be able to eval test splits, which is the point).

Checkpoints are copied separately (see the tar command printed at the end).

Usage:
  python scripts/export_test_bundle.py --config configs/mf_v3_esm3b.yaml --out-dir /raid/team/test_bundle
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def subset_h5(src_path, dst_path, pids):
    """Copy datasets keyed by pid from src h5 into a new dst h5."""
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    kept, missing = 0, 0
    with h5py.File(src_path, 'r') as fs, h5py.File(dst_path, 'w') as fd:
        for pid in pids:
            if pid in fs:
                fs.copy(pid, fd)
                kept += 1
            else:
                missing += 1
    print(f"[BUNDLE] {Path(src_path).name}: copied {kept} proteins "
          f"({missing} not found) -> {dst_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mf_v3_esm3b.yaml')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data = cfg['data']
    out = Path(args.out_dir)

    # 1. Union of all test_* protein ids (LT_95 is the superset; be safe and union all).
    with open(data['splits']) as f:
        splits = json.load(f)
    test_ids = sorted({pid for k, ids in splits.items()
                       if k.startswith('test') for pid in ids})
    print(f"[BUNDLE] {len(test_ids)} unique test proteins across "
          f"{[k for k in splits if k.startswith('test')]}")

    # 2. Subset the two big h5 stores (shared across MF/BP/CC), keep same rel paths.
    subset_h5(data['esm2_h5'], out / data['esm2_h5'], test_ids)
    subset_h5(data['cmap_h5'], out / data['cmap_h5'], test_ids)

    # 3. Copy small files verbatim (all three branches) at their original rel paths.
    small = [
        'data/pdbch/protein_order.json', 'data/pdbch/splits.json', 'data/pdbch/go-basic.obo',
        'data/embeddings/ppi_deepgo.npy', 'data/embeddings/ppi_deepgo_mask.npy',
    ]
    for b in ('mf', 'bp', 'cc'):
        small += [f'data/pdbch/labels_{b}.npy', f'data/pdbch/dag_matrix_{b}.npy',
                  f'data/pdbch/go_terms_{b}.json', f'data/embeddings/go_emb_{b}_v2.npy',
                  f'data/diamond/diamond_results_{b}.tsv']
    for rel in small:
        src = Path(rel)
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[BUNDLE] copied {rel}")
        else:
            print(f"[BUNDLE] SKIP missing {rel}")

    print("\n[BUNDLE] data done. Now add checkpoints + code and tar, e.g.:")
    print(f"  cp -r --parents checkpoints/*/best.pt {out}/")
    print(f"  cp -r ampr configs scripts main.py {out}/   # or rely on git history")
    print(f"  tar -czf {out}.tar.gz -C {out.parent} {out.name}")


if __name__ == '__main__':
    main()
