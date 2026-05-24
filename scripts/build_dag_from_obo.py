#!/usr/bin/env python
"""Build per-ontology DAG adjacency matrices from go-basic.obo.

Reads <data-dir>/go_terms_{mf,bp,cc}.json (column order from build_labels_from_annot.py)
and <data-dir>/go-basic.obo. Writes:
    dag_matrix_mf.npy   (n_mf, n_mf)   float32
    dag_matrix_bp.npy   (n_bp, n_bp)   float32
    dag_matrix_cc.npy   (n_cc, n_cc)   float32

Convention (matches ampr/training/loss.py:AMPRLoss):
    A[i, j] = 1.0  iff  go_terms[j] is a DIRECT parent of go_terms[i] in the GO graph
                        AND both terms belong to this ontology.

Only direct parents are stored; transitive closure is enforced at training/inference
time by the relu-of-difference DAG penalty (True Path Rule).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import obonet

logging.basicConfig(level=logging.INFO, format="[build_dag] %(message)s")
log = logging.getLogger(__name__)

ONT_SHORT = ["mf", "bp", "cc"]


def build_dag_matrix(go_terms: list[str], graph) -> np.ndarray:
    """Square (C, C) matrix; A[i,j]=1 iff term[j] is a direct parent of term[i]."""
    C = len(go_terms)
    term_set = set(go_terms)
    col = {t: i for i, t in enumerate(go_terms)}
    A = np.zeros((C, C), dtype=np.float32)
    for i, child in enumerate(go_terms):
        if child not in graph:
            continue
        # In obonet, graph.successors(child) yields parents for is_a edges.
        for parent in graph.successors(child):
            if parent in term_set and parent != child:
                A[i, col[parent]] = 1.0
    return A


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with go-basic.obo and go_terms_{mf,bp,cc}.json")
    args = parser.parse_args()

    data_dir: Path = args.data_dir

    log.info("loading %s", data_dir / "go-basic.obo")
    graph = obonet.read_obo(str(data_dir / "go-basic.obo"))
    log.info("OBO graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    for ont in ONT_SHORT:
        with open(data_dir / f"go_terms_{ont}.json", "r", encoding="utf-8") as f:
            terms = json.load(f)
        A = build_dag_matrix(terms, graph)
        out_path = data_dir / f"dag_matrix_{ont}.npy"
        np.save(out_path, A)
        log.info("%s: %d terms, %d parent edges -> %s",
                 ont.upper(), len(terms), int(A.sum()), out_path.name)


if __name__ == "__main__":
    main()
