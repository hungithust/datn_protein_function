#!/usr/bin/env python
"""Concatenate L2-normalized GO text + graph embeddings -> final GO label matrix.

Usage:
  python scripts/build_go_combined.py \
    --text data/embeddings/go_text_mf.npy \
    --graph data/embeddings/go_graph_mf.npy \
    --out data/embeddings/go_emb_mf_v2.npy
"""
import argparse
from pathlib import Path
import numpy as np


def l2norm_concat(text: np.ndarray, graph: np.ndarray) -> np.ndarray:
    if text.shape[0] != graph.shape[0]:
        raise ValueError(f"row mismatch: text {text.shape[0]} vs graph {graph.shape[0]}")
    def norm(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    return np.concatenate([norm(text), norm(graph)], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True)
    ap.add_argument('--graph', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = l2norm_concat(np.load(args.text), np.load(args.graph))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"[GO-COMBINED] saved {args.out} shape={out.shape}")


if __name__ == '__main__':
    main()
