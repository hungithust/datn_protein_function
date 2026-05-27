import h5py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.precompute_esm2_residue import write_residue_h5


def test_write_residue_h5_resume(tmp_path):
    h5_path = tmp_path / 'esm2.h5'
    # First pass: write 2 proteins
    iter1 = [('P1', np.random.rand(10, 8).astype(np.float32)),
             ('P2', np.random.rand(5, 8).astype(np.float32))]
    write_residue_h5(iter1, str(h5_path))
    with h5py.File(h5_path, 'r') as f:
        assert set(f.keys()) == {'P1', 'P2'}
        assert f['P1'].shape == (10, 8)

    # Second pass: same keys should be skipped, new key added
    iter2 = [('P1', np.random.rand(99, 8).astype(np.float32)),
             ('P3', np.random.rand(7, 8).astype(np.float32))]
    write_residue_h5(iter2, str(h5_path))
    with h5py.File(h5_path, 'r') as f:
        assert set(f.keys()) == {'P1', 'P2', 'P3'}
        assert f['P1'].shape == (10, 8)  # NOT overwritten
        assert f['P3'].shape == (7, 8)
