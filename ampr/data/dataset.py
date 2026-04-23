"""AMPR PyTorch Dataset class."""

import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger('ampr')


class AMPRDataset(Dataset):
    """
    Load precomputed embeddings + GO labels for AMPR training.

    Inputs (.npy files must all share the same protein ordering — the order
    in which proteins were saved during precomputation):
        seq_emb_path      (N, 1024) ProteinBERT
        struct_emb_path   (N, 1024) ProstT5
        ppi_emb_path      (N, 128)  Node2Vec (zero rows for missing PPI)
        labels_path       (N, C)    binary GO annotations
        dag_matrix_path   (C, C)    GO hierarchy adjacency
        go_emb_path       (C, 768)  BioBERT GO term embeddings
        splits_path       JSON: {"train": [...prot_ids...], "valid": [...], "test": [...]}
        protein_order_path  JSON: [prot_id_0, prot_id_1, ...] — row order of .npy files

    The protein_order_path is required to correctly map split protein IDs to
    .npy row indices. Without it, a protein-ID mismatch would silently corrupt
    every training example.
    """

    def __init__(self, seq_emb_path, struct_emb_path, ppi_emb_path, labels_path,
                 dag_matrix_path, go_emb_path, splits_path, protein_order_path,
                 branch='MF', split='train'):

        self.seq_emb = np.load(seq_emb_path).astype(np.float32)
        self.struct_emb = np.load(struct_emb_path).astype(np.float32)
        self.ppi_emb = np.load(ppi_emb_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.float32)
        self._dag_matrix = np.load(dag_matrix_path).astype(np.float32)
        self._go_emb = np.load(go_emb_path).astype(np.float32)

        # Row index mapping: prot_id → position in .npy files
        with open(protein_order_path, 'r') as f:
            protein_order = json.load(f)
        self._prot2idx = {pid: i for i, pid in enumerate(protein_order)}

        with open(splits_path, 'r') as f:
            splits = json.load(f)
        all_split_ids = splits.get(split, [])

        # Only keep proteins that have a row in the .npy files
        self.protein_ids = [p for p in all_split_ids if p in self._prot2idx]
        missing = len(all_split_ids) - len(self.protein_ids)
        if missing > 0:
            logger.warning(f"[DATASET] {missing} proteins from {split} split not found in protein_order — skipped")

        logger.info(f"[DATASET] Branch={branch} split={split}: {len(self.protein_ids)} proteins")
        logger.info(f"[DATASET] seq_emb  shape : {self.seq_emb.shape}")
        logger.info(f"[DATASET] labels   shape : {self.labels.shape}")
        logger.info(f"[DATASET] DAG      shape : {self._dag_matrix.shape}")
        logger.info(f"[DATASET] GO emb   shape : {self._go_emb.shape}")
        logger.info(f"[DATASET] PPI zero rows  : {(self.ppi_emb.sum(axis=1) == 0).sum()}/{len(self.ppi_emb)}")

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        row = self._prot2idx[pid]
        return {
            'x_seq':   torch.from_numpy(self.seq_emb[row]),
            'x_3di':   torch.from_numpy(self.struct_emb[row]),
            'x_ppi':   torch.from_numpy(self.ppi_emb[row]),
            'labels':  torch.from_numpy(self.labels[row]),
            'prot_id': pid,
        }

    @property
    def dag_matrix_torch(self):
        return torch.from_numpy(self._dag_matrix)

    @property
    def go_emb_torch(self):
        return torch.from_numpy(self._go_emb)


def get_dataloaders(data_config, batch_size, num_workers=0):
    """
    Build train/valid/test DataLoaders from config['data'] dict.

    data_config keys (matching configs/mf.yaml):
        seq_emb, struct_emb, ppi_emb, labels, dag_matrix,
        go_emb, splits, protein_order, branch
    """
    branch = data_config.get('branch', 'MF')

    def make_dataset(split):
        return AMPRDataset(
            seq_emb_path=data_config['seq_emb'],
            struct_emb_path=data_config['struct_emb'],
            ppi_emb_path=data_config['ppi_emb'],
            labels_path=data_config['labels'],
            dag_matrix_path=data_config['dag_matrix'],
            go_emb_path=data_config['go_emb'],
            splits_path=data_config['splits'],
            protein_order_path=data_config['protein_order'],
            branch=branch,
            split=split,
        )

    train_ds = make_dataset('train')
    valid_ds = make_dataset('valid')
    test_ds  = make_dataset('test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, train_ds
