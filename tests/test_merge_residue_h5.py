import h5py
import numpy as np

from scripts.merge_residue_h5 import merge_h5


def test_merge_disjoint_and_skip_existing(tmp_path):
    s0 = tmp_path / 's0.h5'
    s1 = tmp_path / 's1.h5'
    with h5py.File(s0, 'w') as f:
        f.create_dataset('A', data=np.zeros((3, 2), np.float32))
        f.create_dataset('B', data=np.zeros((4, 2), np.float32))
    with h5py.File(s1, 'w') as f:
        f.create_dataset('B', data=np.ones((9, 2), np.float32))  # dup key
        f.create_dataset('C', data=np.zeros((5, 2), np.float32))

    out = tmp_path / 'merged.h5'
    merge_h5([str(s0), str(s1)], str(out))
    with h5py.File(out, 'r') as f:
        assert set(f.keys()) == {'A', 'B', 'C'}
        assert f['B'].shape == (4, 2)  # first writer wins, not overwritten
