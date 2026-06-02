#!/usr/bin/env python
"""GO-ontology graph embedding via truncated SVD of the normalized DAG adjacency.

Deterministic and dependency-light (numpy/scipy). Concatenated later with the
SapBERT text embedding to give each GO term both semantic and topological signal.

Usage:
  python scripts/precompute_go_graph.py \
    --dag data/pdbch/dag_matrix_mf.npy \
    --out data/embeddings/go_graph_mf.npy --dim 128
"""
import argparse
from pathlib import Path
import numpy as np


def graph_embedding(dag: np.ndarray, dim: int = 128) -> np.ndarray:
    """Symmetrize -> add self-loops -> sym-normalize -> truncated SVD -> (N, dim)."""
    n = dag.shape[0]
    A = ((dag + dag.T) > 0).astype(np.float64)
    np.fill_diagonal(A, 1.0)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    A_norm = (A * dinv[:, None]) * dinv[None, :]
    # Deterministic SVD (sign-fixed by largest-magnitude component per column)
    U, S, _ = np.linalg.svd(A_norm)
    k = min(dim, n)
    emb = U[:, :k] * S[:k]
    for j in range(k):
        if emb[np.argmax(np.abs(emb[:, j])), j] < 0:
            emb[:, j] = -emb[:, j]
    if k < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - k)))
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dim', type=int, default=128)
    args = ap.parse_args()
    dag = np.load(args.dag)
    emb = graph_embedding(dag, args.dim)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, emb)
    print(f"[GO-GRAPH] saved {args.out} shape={emb.shape}")


if __name__ == '__main__':
    main()
