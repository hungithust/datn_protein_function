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
                 branch='MF', split='train',
                 cmap_h5_paths: dict | None = None,
                 use_cmap: bool = False,
                 max_len: int = 1000):

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

        self.use_cmap = use_cmap
        self.max_len = max_len
        self._cmap_store = None
        if use_cmap:
            from ampr.data.contact_map_h5 import ContactMapStore
            assert cmap_h5_paths is not None, "use_cmap=True requires cmap_h5_paths"
            self._cmap_store = ContactMapStore(cmap_h5_paths)
            # Filter to proteins with available cmap
            before = len(self.protein_ids)
            self.protein_ids = [p for p in self.protein_ids if p in self._cmap_store]
            logger.info(f"[DATASET] cmap filter: {before} → {len(self.protein_ids)}")

    def __len__(self):
        return len(self.protein_ids)

    _ALPHABET = "ACDEFGHIKLMNPQRSTVWYBOUXZ-."
    _CHAR2IDX = {a: i for i, a in enumerate(_ALPHABET)}

    def _encode_seq(self, seq: str) -> torch.Tensor:
        L = len(seq)
        M = torch.zeros(L, 26, dtype=torch.float32)
        for i, a in enumerate(seq):
            M[i, self._CHAR2IDX.get(a, self._CHAR2IDX.get("X", 22))] = 1.0
        return M

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        row = self._prot2idx[pid]
        item = {
            'x_seq':   torch.from_numpy(self.seq_emb[row]),
            'x_3di':   torch.from_numpy(self.struct_emb[row]),
            'x_ppi':   torch.from_numpy(self.ppi_emb[row]),
            'labels':  torch.from_numpy(self.labels[row]),
            'prot_id': pid,
        }
        if self.use_cmap:
            cmap = self._cmap_store[pid]               # (L, L) float32
            L = min(cmap.shape[0], self.max_len)
            cmap = cmap[:L, :L]
            seq = self._cmap_store.get_sequence(pid)[:L]
            item['cmap'] = torch.from_numpy(cmap)
            item['seq_1hot'] = self._encode_seq(seq)
            item['length'] = L
        return item

    @property
    def dag_matrix_torch(self):
        return torch.from_numpy(self._dag_matrix)

    @property
    def go_emb_torch(self):
        return torch.from_numpy(self._go_emb)


def collate_with_cmap(batch: list[dict]) -> dict:
    """Pad cmap and seq_1hot to longest length in batch; build boolean mask."""
    if 'cmap' not in batch[0]:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

    B = len(batch)
    L_max = max(item['length'] for item in batch)
    cmap_padded = torch.zeros(B, L_max, L_max)
    seq_1hot_padded = torch.zeros(B, L_max, 26)
    cmap_mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item['length']
        cmap_padded[i, :L, :L] = item['cmap']
        seq_1hot_padded[i, :L] = item['seq_1hot']
        cmap_mask[i, :L] = True

    return {
        'x_seq':     torch.stack([b['x_seq']  for b in batch]),
        'x_3di':     torch.stack([b['x_3di']  for b in batch]),
        'x_ppi':     torch.stack([b['x_ppi']  for b in batch]),
        'labels':    torch.stack([b['labels'] for b in batch]),
        'cmap':      cmap_padded,
        'seq_1hot':  seq_1hot_padded,
        'cmap_mask': cmap_mask,
        'prot_id':   [b['prot_id'] for b in batch],
    }


def get_dataloaders(data_config, batch_size, num_workers=0):
    """
    Build train/valid/test DataLoaders from config['data'] dict.

    data_config keys (matching configs/mf.yaml):
        seq_emb, struct_emb, ppi_emb, labels, dag_matrix,
        go_emb, splits, protein_order, branch
    """
    branch = data_config.get('branch', 'MF')
    use_cmap = data_config.get('use_cmap', False)
    cmap_h5_paths = data_config.get('cmap_h5')

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
            use_cmap=use_cmap,
            cmap_h5_paths=cmap_h5_paths,
        )

    collate_fn = collate_with_cmap if use_cmap else None

    train_ds = make_dataset('train')
    valid_ds = make_dataset('valid')
    test_ds  = make_dataset('test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader, train_ds
