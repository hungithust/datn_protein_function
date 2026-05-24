"""Tests for baselines/pdb_to_cmap.py — synthetic structure, no PDB download."""

import gzip
from pathlib import Path

import h5py
import numpy as np
import pytest


SYNTH_CIF = """\
data_FAKE
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 ? 0.000 0.000 0.000 1.00 10.00 1 ALA A CA 1
ATOM 2 C CA . GLY A 1 2 ? 3.800 0.000 0.000 1.00 10.00 2 GLY A CA 1
ATOM 3 C CA . SER A 1 3 ? 7.600 0.000 0.000 1.00 10.00 3 SER A CA 1
"""


@pytest.fixture
def synth_cif(tmp_path: Path) -> Path:
    p = tmp_path / "FAKE.cif.gz"
    with gzip.open(p, "wt") as f:
        f.write(SYNTH_CIF)
    return p


def test_compute_cmap_returns_symmetric_matrix(synth_cif):
    from baselines.pdb_to_cmap import compute_cmap
    cmap, seq = compute_cmap(synth_cif, chain_id="A")
    assert cmap.shape == (3, 3)
    assert np.allclose(cmap, cmap.T)
    assert np.diag(cmap).sum() == 0.0


def test_compute_cmap_distances_match_euclidean(synth_cif):
    from baselines.pdb_to_cmap import compute_cmap
    cmap, _ = compute_cmap(synth_cif, chain_id="A")
    assert abs(cmap[0, 1] - 3.8) < 0.01
    assert abs(cmap[0, 2] - 7.6) < 0.01
    assert abs(cmap[1, 2] - 3.8) < 0.01


def test_save_to_hdf5_roundtrip(tmp_path, synth_cif):
    from baselines.pdb_to_cmap import compute_cmap, save_cmap_h5
    cmap, seq = compute_cmap(synth_cif, chain_id="A")
    h5_path = tmp_path / "out.h5"
    save_cmap_h5(h5_path, "FAKE-A", cmap, seq)
    with h5py.File(h5_path, "r") as f:
        assert "FAKE-A" in f
        assert np.allclose(f["FAKE-A"][...], cmap)
        assert f["FAKE-A"].attrs["sequence"] == seq
