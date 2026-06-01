#!/usr/bin/env python
"""Verify AMPR v3 inputs are mutually aligned before training.

Checks (per branch config):
  - protein_order length == labels rows == ppi rows == ppi_mask len
  - labels cols == dag cols == go_emb rows == n_terms
  - config dims: seq.d_model == ESM-2 residue dim; ppi.in_dim == ppi dim
  - every split protein has an ESM-2 and a cmap entry
  - DAG orientation matches loss.py (child->parent has fewer TPR violations)

Usage:
  python scripts/verify_inputs.py --config configs/mf_v3.yaml
Exit code 0 = all pass, 1 = any fail.
"""
import argparse, json, sys
import numpy as np, yaml, h5py


def load_order(path):
    o = json.loads(open(path).read())
    if isinstance(o, dict):
        o = [k for k, _ in sorted(o.items(), key=lambda kv: kv[1])]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg['data']
    n_terms = cfg['n_terms']
    seq_dim_cfg = cfg['model']['seq']['d_model']
    ppi_dim_cfg = cfg['model']['ppi']['in_dim']

    order = load_order(d['protein_order'])
    labels = np.load(d['labels'])
    ppi = np.load(d['ppi_emb'])
    ppi_mask = np.load(d['ppi_mask'])
    dag = np.load(d['dag_matrix'])
    go_emb = np.load(d['go_emb'])
    splits = json.loads(open(d['splits']).read())

    fails = []
    def check(name, cond, detail=""):
        tag = 'PASS' if cond else 'FAIL'
        print(f"[{tag}] {name}  {detail}")
        if not cond:
            fails.append(name)

    N = len(order)
    check("protein_order==labels rows", labels.shape[0] == N, f"{labels.shape[0]} vs {N}")
    check("ppi rows==N", ppi.shape[0] == N, f"{ppi.shape[0]} vs {N}")
    check("ppi_mask len==N", ppi_mask.shape[0] == N, f"{ppi_mask.shape[0]} vs {N}")
    check("labels cols==n_terms", labels.shape[1] == n_terms, f"{labels.shape[1]} vs {n_terms}")
    check("dag is n_terms x n_terms", dag.shape == (n_terms, n_terms), str(dag.shape))
    check("go_emb rows==n_terms", go_emb.shape[0] == n_terms, f"{go_emb.shape[0]} vs {n_terms}")
    check("ppi dim==cfg ppi.in_dim", ppi.shape[1] == ppi_dim_cfg, f"{ppi.shape[1]} vs {ppi_dim_cfg}")

    prot2idx = {p: i for i, p in enumerate(order)}
    for k in ('train', 'valid'):
        miss = [p for p in splits.get(k, []) if p not in prot2idx]
        check(f"split '{k}' subset of protein_order", not miss, f"missing={len(miss)}")

    with h5py.File(d['esm2_h5'], 'r') as fe, h5py.File(d['cmap_h5'], 'r') as fc:
        ek, ck = set(fe.keys()), set(fc.keys())
        any_pid = next(iter(splits['train']))
        seq_dim = fe[any_pid].shape[1]
        check("ESM-2 residue dim==cfg seq.d_model", seq_dim == seq_dim_cfg, f"{seq_dim} vs {seq_dim_cfg}")
        for k in ('train', 'valid'):
            ids = splits.get(k, [])
            check(f"'{k}' all have ESM-2", all(p in ek for p in ids),
                  f"missing={sum(p not in ek for p in ids)}")
            check(f"'{k}' all have cmap", all(p in ck for p in ids),
                  f"missing={sum(p not in ck for p in ids)}")

    L = labels.astype(np.float32)
    viol_a = float(((L @ dag) * (1 - L)).sum())     # dag[child,parent] (matches loss.py)
    viol_b = float(((L @ dag.T) * (1 - L)).sum())
    check("DAG orientation matches loss.py (A<B)", viol_a < viol_b, f"A={viol_a:.0f} B={viol_b:.0f}")

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
