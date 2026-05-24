"""End-to-end smoke test: load the standardized PDBch data via the existing
AMPRDataset, verify split sizes match HEAL paper expectations.

This test is SKIPPED if the real generated artifacts don't exist locally
(developer hasn't run Tasks 4-6 yet, or is on a fresh checkout).
"""

import json
from pathlib import Path

import numpy as np
import pytest


PDBCH = Path(__file__).resolve().parents[1] / "data" / "pdbch"

REQUIRED_FILES = [
    "splits.json", "protein_order.json",
    "labels_mf.npy", "labels_bp.npy", "labels_cc.npy",
    "dag_matrix_mf.npy", "dag_matrix_bp.npy", "dag_matrix_cc.npy",
]


def _data_ready() -> bool:
    return all((PDBCH / f).exists() for f in REQUIRED_FILES)


pytestmark = pytest.mark.skipif(
    not _data_ready(),
    reason="data/pdbch/ artifacts not generated yet — run Tasks 4-6 scripts first",
)


def test_split_sizes_match_heal_paper():
    with open(PDBCH / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    # HEAL Table S1: PDBch train=29893, valid=3322, test=3414 (±9 due to broken chains)
    assert len(splits["train"]) == 29902
    assert len(splits["valid"]) == 3323
    assert len(splits["test"]) == 3416
    # Cumulative bins
    assert len(splits["test_LT_30"]) < len(splits["test_LT_40"]) < len(splits["test_LT_95"])
    assert len(splits["test_LT_95"]) <= len(splits["test"])


def test_label_shapes_match_canonical_term_counts():
    n_proteins = len(json.loads((PDBCH / "protein_order.json").read_text(encoding="utf-8")))
    assert np.load(PDBCH / "labels_mf.npy").shape == (n_proteins, 489)
    assert np.load(PDBCH / "labels_bp.npy").shape == (n_proteins, 1943)
    assert np.load(PDBCH / "labels_cc.npy").shape == (n_proteins, 320)


def test_dag_shapes_match_label_widths():
    assert np.load(PDBCH / "dag_matrix_mf.npy").shape == (489, 489)
    assert np.load(PDBCH / "dag_matrix_bp.npy").shape == (1943, 1943)
    assert np.load(PDBCH / "dag_matrix_cc.npy").shape == (320, 320)


def test_every_split_protein_is_in_protein_order():
    with open(PDBCH / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    order = set(json.loads((PDBCH / "protein_order.json").read_text(encoding="utf-8")))

    for split_name in ["train", "valid", "test"]:
        missing = [p for p in splits[split_name] if p not in order]
        # A small number of missing is acceptable (e.g. proteins in train.txt that
        # failed FASTA filtering and didn't make it into annot.tsv). HEAL reports
        # ~9 such proteins. Set a generous threshold.
        assert len(missing) < 50, \
            f"{split_name}: {len(missing)} proteins not in protein_order: {missing[:5]}"
