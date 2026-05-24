#!/usr/bin/env python
"""Build per-ontology label matrices and protein_order.json from annot.tsv.

annot.tsv structure (DeepFRI/HEAL convention):
    Lines 1-12:  6 blocks of `### {GO-terms,GO-names} ({ontology})` headers + tab-separated lists
    Line 13:     `### PDB-chain<TAB>GO-terms (molecular_function)<TAB>...(biological_process)<TAB>...(cellular_component)`
    Lines 14+:   <prot_id>\\t<comma-sep MF ids>\\t<comma-sep BP ids>\\t<comma-sep CC ids>

Output files in --data-dir:
    protein_order.json   list[str]                            — row order of label matrices
    go_terms_mf.json     list[str]                            — column order of labels_mf.npy
    go_terms_bp.json     list[str]
    go_terms_cc.json     list[str]
    labels_mf.npy        np.ndarray shape (N, n_mf) float32   — binary
    labels_bp.npy        np.ndarray shape (N, n_bp) float32
    labels_cc.npy        np.ndarray shape (N, n_cc) float32
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[build_labels] %(message)s")
log = logging.getLogger(__name__)

ONT_NAMES = ["molecular_function", "biological_process", "cellular_component"]
ONT_SHORT = ["mf", "bp", "cc"]


def parse_annot(annot_path: Path):
    """Return (go_terms_per_ont, rows).

    go_terms_per_ont: dict mapping ont_short -> list[str]
    rows: list of (prot_id, [mf_set, bp_set, cc_set])
    """
    with open(annot_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [ln.rstrip("\n") for ln in lines if ln.strip()]

    # Lines 0,1,2,3 = MF terms header, MF terms, MF names header, MF names
    # Lines 4,5,6,7 = BP terms header, BP terms, BP names header, BP names
    # Lines 8,9,10,11 = CC terms header, CC terms, CC names header, CC names
    go_terms = {}
    for i, ont in enumerate(ONT_NAMES):
        terms_header_idx = i * 4
        terms_content_idx = i * 4 + 1
        header = lines[terms_header_idx]
        content = lines[terms_content_idx]
        assert header == f"### GO-terms ({ont})", \
            f"line {terms_header_idx}: expected '### GO-terms ({ont})', got {header!r}"
        go_terms[ONT_SHORT[i]] = content.split("\t")

    # Column-header line
    col_header_idx = 12
    assert lines[col_header_idx].startswith("### PDB-chain"), \
        f"line {col_header_idx}: expected column header starting with '### PDB-chain'"

    # Data rows
    rows = []
    for line in lines[col_header_idx + 1:]:
        parts = line.split("\t")
        while len(parts) < 4:
            parts.append("")
        prot_id = parts[0].strip()
        per_ont_ids = []
        for i in range(3):
            s = parts[i + 1].strip()
            per_ont_ids.append(set(s.split(",")) - {""} if s else set())
        rows.append((prot_id, per_ont_ids))

    return go_terms, rows


def build_label_matrix(rows, go_terms_for_ont: list, ont_idx: int) -> np.ndarray:
    col = {t: i for i, t in enumerate(go_terms_for_ont)}
    N = len(rows)
    C = len(go_terms_for_ont)
    M = np.zeros((N, C), dtype=np.float32)
    for r, (_, per_ont_ids) in enumerate(rows):
        for go_id in per_ont_ids[ont_idx]:
            if go_id in col:
                M[r, col[go_id]] = 1.0
    return M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing annot.tsv")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    go_terms, rows = parse_annot(data_dir / "annot.tsv")

    protein_order = [pid for pid, _ in rows]
    with open(data_dir / "protein_order.json", "w", encoding="utf-8") as f:
        json.dump(protein_order, f)
    log.info("protein_order: %d proteins", len(protein_order))

    for ont_idx, ont in enumerate(ONT_SHORT):
        terms = go_terms[ont]
        with open(data_dir / f"go_terms_{ont}.json", "w", encoding="utf-8") as f:
            json.dump(terms, f)
        labels = build_label_matrix(rows, terms, ont_idx)
        np.save(data_dir / f"labels_{ont}.npy", labels)
        n_annotated = int((labels.sum(axis=1) > 0).sum())
        log.info("%s: %d terms, labels shape %s, %d proteins with >=1 label",
                 ont.upper(), len(terms), labels.shape, n_annotated)


if __name__ == "__main__":
    main()
