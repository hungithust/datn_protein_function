"""Tests for scripts/build_dag_from_obo.py"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_build_dag(data_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "build_dag_from_obo.py"
    subprocess.run(
        [sys.executable, str(script), "--data-dir", str(data_dir)],
        check=True,
    )


def _seed_go_terms_json(data_dir: Path):
    """Write the go_terms_*.json files that build_dag depends on (normally
    created by build_labels_from_annot.py — we stub them here to keep the
    DAG test independent of labels test ordering)."""
    (data_dir / "go_terms_mf.json").write_text(
        json.dumps(["GO:0001", "GO:0002", "GO:0003", "GO:0004"]), encoding="utf-8")
    (data_dir / "go_terms_bp.json").write_text(
        json.dumps(["GO:0010", "GO:0011", "GO:0012"]), encoding="utf-8")
    (data_dir / "go_terms_cc.json").write_text(
        json.dumps(["GO:0020", "GO:0021"]), encoding="utf-8")


def test_dag_shape_matches_term_counts(tiny_pdbch):
    _seed_go_terms_json(tiny_pdbch)
    run_build_dag(tiny_pdbch)
    mf = np.load(tiny_pdbch / "dag_matrix_mf.npy")
    bp = np.load(tiny_pdbch / "dag_matrix_bp.npy")
    cc = np.load(tiny_pdbch / "dag_matrix_cc.npy")
    assert mf.shape == (4, 4)
    assert bp.shape == (3, 3)
    assert cc.shape == (2, 2)
    assert mf.dtype == np.float32


def test_parent_relationships_recorded(tiny_pdbch):
    """In the tiny OBO: GO:0001 is_a GO:0003; GO:0002 is_a GO:0003; GO:0003 is_a GO:0004.
    Convention: A[i, j] = 1 iff term j is parent of term i."""
    _seed_go_terms_json(tiny_pdbch)
    run_build_dag(tiny_pdbch)
    mf = np.load(tiny_pdbch / "dag_matrix_mf.npy")
    cols = {"GO:0001": 0, "GO:0002": 1, "GO:0003": 2, "GO:0004": 3}
    # GO:0001 (row 0) has parent GO:0003 (col 2)
    assert mf[cols["GO:0001"], cols["GO:0003"]] == 1.0
    # GO:0002 (row 1) has parent GO:0003 (col 2)
    assert mf[cols["GO:0002"], cols["GO:0003"]] == 1.0
    # GO:0003 (row 2) has parent GO:0004 (col 3)
    assert mf[cols["GO:0003"], cols["GO:0004"]] == 1.0
    # No transitive parents stored (GO:0001 is NOT directly recorded as having GO:0004 as parent)
    assert mf[cols["GO:0001"], cols["GO:0004"]] == 0.0
    # Root has no parents
    assert mf[cols["GO:0004"]].sum() == 0.0


def test_no_self_loops(tiny_pdbch):
    _seed_go_terms_json(tiny_pdbch)
    run_build_dag(tiny_pdbch)
    for ont in ["mf", "bp", "cc"]:
        m = np.load(tiny_pdbch / f"dag_matrix_{ont}.npy")
        assert np.diag(m).sum() == 0.0, f"{ont} DAG has self-loops"


def test_cross_ontology_edges_dropped(tiny_pdbch):
    """A BP term should never be parent of an MF term in any ontology's DAG."""
    _seed_go_terms_json(tiny_pdbch)
    run_build_dag(tiny_pdbch)
    bp = np.load(tiny_pdbch / "dag_matrix_bp.npy")
    # BP terms in our tiny set have no parent relationships → all zeros
    assert bp.sum() == 0.0
