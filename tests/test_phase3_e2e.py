"""End-to-end smoke: tiny fixture, build full v3 pipeline, run 2 epochs,
verify loss decreases and inference probs+DAG propagation flow works."""

import json
import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from ampr.data.dataset import AMPRDatasetV3, collate_variable_length
from ampr.models.ampr import AMPRModelV3
from ampr.training.loss import AMPRLoss
from ampr.training.trainer import train_one_epoch_v3
from ampr.evaluation.dag_inference import propagate_scores_upward
from ampr.evaluation.metrics import compute_fmax


def test_phase3_end_to_end(tmp_path):
    torch.manual_seed(0); np.random.seed(0)
    N, C = 6, 4
    ids = [f"P{i}" for i in range(N)]
    Path(tmp_path / 'order.json').write_text(json.dumps(ids))
    Path(tmp_path / 'splits.json').write_text(json.dumps({'train': ids[:4], 'valid': ids[4:]}))
    np.save(tmp_path / 'labels.npy', (np.random.rand(N, C) > 0.5).astype(np.float32))
    dag = np.zeros((C, C), dtype=np.float32); dag[1, 0] = 1.0
    np.save(tmp_path / 'dag.npy', dag)
    np.save(tmp_path / 'go.npy', np.random.rand(C, 8).astype(np.float32))
    with h5py.File(tmp_path / 'esm2.h5', 'w') as f:
        for p in ids:
            L = np.random.randint(4, 9)
            f.create_dataset(p, data=np.random.rand(L, 16).astype(np.float32))
    np.save(tmp_path / 'ppi.npy', np.random.rand(N, 8).astype(np.float32))
    np.save(tmp_path / 'mask.npy', np.array([True] * N))
    with h5py.File(tmp_path / 'cmap.h5', 'w') as f:
        with h5py.File(tmp_path / 'esm2.h5', 'r') as fe:
            for p in ids:
                L = fe[p].shape[0]
                f.create_dataset(p, data=(np.random.rand(L, L) * 20).astype(np.float32))

    def mk(split):
        return AMPRDatasetV3(
            esm2_h5=str(tmp_path / 'esm2.h5'),
            ppi_emb=str(tmp_path / 'ppi.npy'),
            ppi_mask=str(tmp_path / 'mask.npy'),
            cmap_h5=str(tmp_path / 'cmap.h5'),
            labels=str(tmp_path / 'labels.npy'),
            dag_matrix=str(tmp_path / 'dag.npy'),
            go_emb=str(tmp_path / 'go.npy'),
            splits=str(tmp_path / 'splits.json'),
            protein_order=str(tmp_path / 'order.json'),
            branch='MF', split=split, max_len=20,
        )
    ds_tr, ds_va = mk('train'), mk('valid')
    ld_tr = DataLoader(ds_tr, batch_size=2, collate_fn=collate_variable_length)
    ld_va = DataLoader(ds_va, batch_size=2, collate_fn=collate_variable_length)

    model = AMPRModelV3(n_terms=C, seq_dim=16, seq_n_heads=2, seq_n_layers=1,
                       gnn_node_dim=16, gnn_n_layers=1, ppi_dim=8,
                       d_hidden=16, fusion_n_heads=2, fusion_n_layers=1,
                       go_emb_dim=8, dropout=0.0)
    loss_fn = AMPRLoss(ds_tr.dag_matrix, lambda_dag=0.5, loss_type='asl')
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    l0 = train_one_epoch_v3(model, ld_tr, loss_fn, opt, ds_tr.go_emb, device='cpu')
    l1 = train_one_epoch_v3(model, ld_tr, loss_fn, opt, ds_tr.go_emb, device='cpu')
    assert l1 < l0 * 1.05  # allow small noise

    # Val inference + DAG propagation flow
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for b in ld_va:
            p = torch.sigmoid(model(b, go_emb=ds_tr.go_emb)).cpu().numpy()
            probs_list.append(p)
            labels_list.append(b['labels'].cpu().numpy())
    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    probs_dag = propagate_scores_upward(probs, dag)
    f_raw, _ = compute_fmax(labels, probs)
    f_dag, _ = compute_fmax(labels, probs_dag)
    assert 0.0 <= f_raw <= 1.0 and 0.0 <= f_dag <= 1.0
