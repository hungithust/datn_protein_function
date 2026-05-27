import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_ppi_from_deepgo import build_ppi_matrix


def test_build_ppi_matrix_with_partial_coverage(tmp_path):
    # Simulate DeepGO pkl: DataFrame with 'accessions' and 'embeddings'
    df = pd.DataFrame({
        'accessions': ['P0A6Y8', 'P12345'],
        'embeddings': [np.ones(256, dtype=np.float32),
                       np.full(256, 2.0, dtype=np.float32)],
    })
    pkl = tmp_path / 'graph.pkl'
    df.to_pickle(pkl)

    # SIFTS mapping: PDB-chain → UniProt
    sifts = {'1A0R-A': 'P0A6Y8', '2B3C-B': 'P12345', '3XYZ-A': 'Q99999'}
    protein_order = ['1A0R-A', '2B3C-B', '3XYZ-A', '4PDB-C']  # last has no SIFTS

    emb, mask = build_ppi_matrix(str(pkl), sifts, protein_order, dim=256)
    assert emb.shape == (4, 256)
    assert mask.shape == (4,) and mask.dtype == bool
    assert mask.tolist() == [True, True, False, False]
    assert np.allclose(emb[0], 1.0)
    assert np.allclose(emb[1], 2.0)
    assert np.all(emb[2] == 0.0) and np.all(emb[3] == 0.0)
