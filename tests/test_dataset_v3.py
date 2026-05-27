import json
import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from ampr.data.dataset import AMPRDatasetV3, collate_variable_length


def _fixture(tmp_path):
    order = ['P1', 'P2', 'P3']
    Path(tmp_path / 'order.json').write_text(json.dumps(order))
    splits = {'train': order}
    Path(tmp_path / 'splits.json').write_text(json.dumps(splits))
    labels = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    np.save(tmp_path / 'labels.npy', labels)
    dag = np.zeros((3, 3), dtype=np.float32)
    np.save(tmp_path / 'dag.npy', dag)
    go_emb = np.random.rand(3, 8).astype(np.float32)
    np.save(tmp_path / 'go.npy', go_emb)
    # ESM2 HDF5 — ragged
    with h5py.File(tmp_path / 'esm2.h5', 'w') as f:
        f.create_dataset('P1', data=np.random.rand(10, 16).astype(np.float32))
        f.create_dataset('P2', data=np.random.rand(7, 16).astype(np.float32))
        f.create_dataset('P3', data=np.random.rand(5, 16).astype(np.float32))
    # PPI: P2 missing
    ppi = np.zeros((3, 4), dtype=np.float32)
    ppi[0] = 1.0; ppi[2] = 2.0
    mask = np.array([True, False, True])
    np.save(tmp_path / 'ppi.npy', ppi)
    np.save(tmp_path / 'ppi_mask.npy', mask)
    # cmap HDF5
    with h5py.File(tmp_path / 'cmap.h5', 'w') as f:
        f.create_dataset('P1', data=(np.random.rand(10, 10) * 20).astype(np.float32))
        f.create_dataset('P2', data=(np.random.rand(7, 7) * 20).astype(np.float32))
        f.create_dataset('P3', data=(np.random.rand(5, 5) * 20).astype(np.float32))
    return tmp_path


def test_dataset_v3_item_shapes(tmp_path):
    d = _fixture(tmp_path)
    ds = AMPRDatasetV3(
        esm2_h5=str(d / 'esm2.h5'),
        ppi_emb=str(d / 'ppi.npy'),
        ppi_mask=str(d / 'ppi_mask.npy'),
        cmap_h5=str(d / 'cmap.h5'),
        labels=str(d / 'labels.npy'),
        dag_matrix=str(d / 'dag.npy'),
        go_emb=str(d / 'go.npy'),
        splits=str(d / 'splits.json'),
        protein_order=str(d / 'order.json'),
        branch='MF', split='train', max_len=20,
    )
    assert len(ds) == 3
    item = ds[0]
    assert item['x_seq_residue'].shape == (10, 16)
    assert item['cmap'].shape == (10, 10)
    assert item['x_ppi'].shape == (4,)
    assert item['ppi_mask'].item() is True
    assert item['labels'].shape == (3,)


def test_collate_pads_and_masks(tmp_path):
    d = _fixture(tmp_path)
    ds = AMPRDatasetV3(
        esm2_h5=str(d / 'esm2.h5'),
        ppi_emb=str(d / 'ppi.npy'),
        ppi_mask=str(d / 'ppi_mask.npy'),
        cmap_h5=str(d / 'cmap.h5'),
        labels=str(d / 'labels.npy'),
        dag_matrix=str(d / 'dag.npy'),
        go_emb=str(d / 'go.npy'),
        splits=str(d / 'splits.json'),
        protein_order=str(d / 'order.json'),
        branch='MF', split='train', max_len=20,
    )
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_variable_length)
    batch = next(iter(loader))
    assert batch['x_seq_residue'].shape == (3, 10, 16)  # max L = 10
    assert batch['cmap'].shape == (3, 10, 10)
    assert batch['seq_mask'].shape == (3, 10)
    # Protein 1 (P2) length 7, expect 3 padding positions
    assert batch['seq_mask'][1, :7].all() and not batch['seq_mask'][1, 7:].any()
    assert batch['x_ppi'].shape == (3, 4)
    assert batch['ppi_mask'].tolist() == [True, False, True]
    assert batch['labels'].shape == (3, 3)
