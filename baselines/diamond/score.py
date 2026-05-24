#!/usr/bin/env python
"""DeepGOPlus-style scoring from Diamond search results.

score(q, t) = max over homologs h of: pident(q,h)/100 * label(h, t)
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="[diamond] %(message)s")
log = logging.getLogger(__name__)


def compute_diamond_scores(rows, train_labels, train_idx, test_ids, n_terms):
    """rows: iterable of (qid, sid, pident_percent)."""
    test_idx = {t: i for i, t in enumerate(test_ids)}
    scores = np.zeros((len(test_ids), n_terms), dtype=np.float32)
    hits = defaultdict(list)
    for q, s, p in rows:
        hits[q].append((s, p))
    for q, hs in hits.items():
        if q not in test_idx:
            continue
        i = test_idx[q]
        for s, p in hs:
            if s not in train_idx:
                continue
            sim = p / 100.0
            row = train_labels[train_idx[s]] * sim
            scores[i] = np.maximum(scores[i], row)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-results", type=Path,
                        default=Path("results/baselines/diamond/search_results.tsv"))
    parser.add_argument("--ontology", choices=["mf", "bp", "cc"], required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/pdbch/splits.json"))
    parser.add_argument("--protein-order", type=Path, default=Path("data/pdbch/protein_order.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.splits) as f:
        splits = json.load(f)
    with open(args.protein_order) as f:
        order = json.load(f)
    row_of = {p: i for i, p in enumerate(order)}

    labels_full = np.load(args.labels)
    train_idx = {p: row_of[p] for p in splits["train"] if p in row_of}
    train_labels = labels_full  # use full matrix; train_idx subselects rows

    test_ids = splits["test"]

    rows = []
    with open(args.search_results) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1], float(parts[2])))
    log.info("loaded %d search rows", len(rows))

    n_terms = labels_full.shape[1]
    y_pred = compute_diamond_scores(rows, train_labels, train_idx, test_ids, n_terms)
    y_true = np.zeros_like(y_pred)
    for i, p in enumerate(test_ids):
        if p in row_of:
            y_true[i] = labels_full[row_of[p]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, y_true=y_true, y_pred=y_pred,
                        prot_ids=np.asarray(test_ids))
    log.info("saved %s", args.out)


if __name__ == "__main__":
    main()
