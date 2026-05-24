"""Tests for scripts/build_labels_from_annot.py"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_build_labels(data_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "build_labels_from_annot.py"
    subprocess.run(
        [sys.executable, str(script), "--data-dir", str(data_dir)],
        check=True,
    )


def test_protein_order_lists_all_proteins(tiny_pdbch):
    run_build_labels(tiny_pdbch)
    with open(tiny_pdbch / "protein_order.json", "r", encoding="utf-8") as f:
        order = json.load(f)
    assert sorted(order) == ["1AAA-A", "1BBB-A", "1CCC-A"]


def test_labels_shape_matches_term_counts(tiny_pdbch):
    run_build_labels(tiny_pdbch)
    mf = np.load(tiny_pdbch / "labels_mf.npy")
    bp = np.load(tiny_pdbch / "labels_bp.npy")
    cc = np.load(tiny_pdbch / "labels_cc.npy")
    assert mf.shape == (3, 4)
    assert bp.shape == (3, 3)
    assert cc.shape == (3, 2)
    assert mf.dtype == np.float32


def test_labels_values_match_annot(tiny_pdbch):
    run_build_labels(tiny_pdbch)

    with open(tiny_pdbch / "protein_order.json", "r", encoding="utf-8") as f:
        order = json.load(f)
    idx = {p: i for i, p in enumerate(order)}

    with open(tiny_pdbch / "go_terms_mf.json", "r", encoding="utf-8") as f:
        mf_terms = json.load(f)
    col = {t: i for i, t in enumerate(mf_terms)}

    mf = np.load(tiny_pdbch / "labels_mf.npy")
    # 1AAA-A → GO:0001, GO:0003
    assert mf[idx["1AAA-A"], col["GO:0001"]] == 1.0
    assert mf[idx["1AAA-A"], col["GO:0003"]] == 1.0
    assert mf[idx["1AAA-A"], col["GO:0002"]] == 0.0
    # 1BBB-A → GO:0002
    assert mf[idx["1BBB-A"], col["GO:0002"]] == 1.0
    assert mf[idx["1BBB-A"], col["GO:0001"]] == 0.0


def test_protein_with_empty_ontology_gets_zero_row(tiny_pdbch):
    run_build_labels(tiny_pdbch)
    with open(tiny_pdbch / "protein_order.json", "r", encoding="utf-8") as f:
        order = json.load(f)
    idx = {p: i for i, p in enumerate(order)}

    bp = np.load(tiny_pdbch / "labels_bp.npy")
    # 1CCC-A has empty BP column
    assert bp[idx["1CCC-A"]].sum() == 0.0


def test_go_terms_json_preserves_annot_order(tiny_pdbch):
    """Column order in labels_*.npy MUST match the GO term order written
    in go_terms_*.json — otherwise Phase 1/2 baselines that index columns
    by GO term ID will silently corrupt every prediction."""
    run_build_labels(tiny_pdbch)
    with open(tiny_pdbch / "go_terms_mf.json", "r", encoding="utf-8") as f:
        mf_terms = json.load(f)
    assert mf_terms == ["GO:0001", "GO:0002", "GO:0003", "GO:0004"]
