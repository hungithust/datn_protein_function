"""AMPR PyTorch Dataset class."""

import logging
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('ampr')


class AMPRDataset(Dataset):
    """
    Load precomputed embeddings + GO labels for AMPR training.

    Input:
        seq_embeddings.npy    (N, 1024) ProteinBERT
        struct_embeddings.npy (N, 1024) ProstT5
        ppi_embeddings.npy    (N, 128)  Node2Vec
        labels_{branch}.npy   (N, C)    binary GO annotations
        dag_matrix_{branch}.npy (C, C)  GO hierarchy
        go_emb_{branch}.npy   (C, 768)  BioBERT GO term embeddings
        splits.json           {train: [...], valid: [...], test: [...]}

    Yields:
        (x_seq, x_3di, x_ppi, labels, prot_id)
    """

    def __init__(self, seq_emb_path, struct_emb_path, ppi_emb_path, labels_path,
                 dag_matrix_path, go_emb_path, splits_path, branch='MF', split='train'):

        self.seq_emb = np.load(seq_emb_path).astype(np.float32)
        self.struct_emb = np.load(struct_emb_path).astype(np.float32)
        self.ppi_emb = np.load(ppi_emb_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.float32)
        self.dag_matrix = np.load(dag_matrix_path).astype(np.float32)
        self.go_emb = np.load(go_emb_path).astype(np.float32)

        with open(splits_path, 'r') as f:
            splits = json.load(f)
        self.protein_ids = splits.get(split, [])

        logger.info(f"[DATASET] Loaded {branch} {split} set: {len(self.protein_ids)} proteins")
        logger.info(f"[DATASET] seq_emb shape: {self.seq_emb.shape}")
        logger.info(f"[DATASET] labels shape: {self.labels.shape}")
        logger.info(f"[DATASET] DAG matrix shape: {self.dag_matrix.shape}")

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        return {
            'x_seq': torch.from_numpy(self.seq_emb[idx]),
            'x_3di': torch.from_numpy(self.struct_emb[idx]),
            'x_ppi': torch.from_numpy(self.ppi_emb[idx]),
            'labels': torch.from_numpy(self.labels[idx]),
            'prot_id': pid,
        }

    @property
    def dag_matrix_torch(self):
        """Return DAG matrix as torch tensor."""
        return torch.from_numpy(self.dag_matrix)

    @property
    def go_emb_torch(self):
        """Return GO embeddings as torch tensor."""
        return torch.from_numpy(self.go_emb)


def get_dataloaders(config, batch_size, num_workers=0):
    """Create train/valid/test DataLoaders."""
    train_ds = AMPRDataset(**config['data'], split='train')
    valid_ds = AMPRDataset(**config['data'], split='valid')
    test_ds = AMPRDataset(**config['data'], split='test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, train_ds
