import json
import h5py
import numpy as np
import torch
from pathlib import Path

from ampr.training.trainer import train_one_epoch_v3
from ampr.models.ampr import AMPRModelV3
from ampr.training.loss import AMPRLoss
from ampr.data.dataset import AMPRDatasetV3, collate_variable_length
from torch.utils.data import DataLoader


def test_trainer_v3_one_epoch_drops_loss(tmp_path):
    torch.manual_seed(0)
    order = ['P1', 'P2']
    Path(tmp_path / 'order.json').write_text(json.dumps(order))
    Path(tmp_path / 'splits.json').write_text(json.dumps({'train': order}))
    np.save(tmp_path / 'labels.npy', np.array([[1, 0], [0, 1]], dtype=np.float32))
    np.save(tmp_path / 'dag.npy', np.zeros((2, 2), dtype=np.float32))
    np.save(tmp_path / 'go.npy', np.random.rand(2, 8).astype(np.float32))
    with h5py.File(tmp_path / 'esm2.h5', 'w') as f:
        f.create_dataset('P1', data=np.random.rand(6, 16).astype(np.float32))
        f.create_dataset('P2', data=np.random.rand(5, 16).astype(np.float32))
    np.save(tmp_path / 'ppi.npy', np.random.rand(2, 8).astype(np.float32))
    np.save(tmp_path / 'mask.npy', np.array([True, True]))
    with h5py.File(tmp_path / 'cmap.h5', 'w') as f:
        f.create_dataset('P1', data=(np.random.rand(6, 6) * 20).astype(np.float32))
        f.create_dataset('P2', data=(np.random.rand(5, 5) * 20).astype(np.float32))

    ds = AMPRDatasetV3(
        esm2_h5=str(tmp_path / 'esm2.h5'),
        ppi_emb=str(tmp_path / 'ppi.npy'),
        ppi_mask=str(tmp_path / 'mask.npy'),
        cmap_h5=str(tmp_path / 'cmap.h5'),
        labels=str(tmp_path / 'labels.npy'),
        dag_matrix=str(tmp_path / 'dag.npy'),
        go_emb=str(tmp_path / 'go.npy'),
        splits=str(tmp_path / 'splits.json'),
        protein_order=str(tmp_path / 'order.json'),
        branch='MF', split='train', max_len=20,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_variable_length)
    model = AMPRModelV3(n_terms=2, seq_dim=16, seq_n_heads=2, seq_n_layers=1,
                       gnn_node_dim=16, gnn_n_layers=1, ppi_dim=8,
                       d_hidden=16, fusion_n_heads=2, fusion_n_layers=1,
                       go_emb_dim=8, dropout=0.0)
    loss_fn = AMPRLoss(ds.dag_matrix, lambda_dag=0.0, loss_type='asl')
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(5):
        losses.append(train_one_epoch_v3(model, loader, loss_fn, opt,
                                         go_emb=ds.go_emb, device='cpu'))
    assert losses[-1] < losses[0]
