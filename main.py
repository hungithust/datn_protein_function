#!/usr/bin/env python
"""
AMPR: Adaptive Multimodal Protein Representation — Main training entry point.

Usage:
    python main.py --config configs/mf.yaml
    python main.py --config configs/bp.yaml
    python main.py --config configs/cc.yaml
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from ampr.data.dataset import AMPRDataset
from ampr.models.ampr import AMPRModel
from ampr.training.trainer import Trainer


def setup_logging(log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('[%(name)s] %(message)s')
    log = logging.getLogger('ampr')
    log.setLevel(logging.INFO)
    for handler in [logging.FileHandler(log_file), logging.StreamHandler()]:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='AMPR training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed',   type=int,  default=42)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    log = setup_logging(config['output']['log_file'])
    log.info(f"Config: {args.config}")
    log.info(f"Branch: {config['branch']} | n_terms: {config['n_terms']}")
    log.info(f"Device: {config['training']['device']}")

    seed_everything(args.seed)
    log.info(f"Seed: {args.seed}")

    # Load train-split dataset (for dag_matrix + go_emb properties used by Trainer)
    data_cfg = config['data']
    dataset = AMPRDataset(
        seq_emb_path=data_cfg['seq_emb'],
        struct_emb_path=data_cfg['struct_emb'],
        ppi_emb_path=data_cfg['ppi_emb'],
        labels_path=data_cfg['labels'],
        dag_matrix_path=data_cfg['dag_matrix'],
        go_emb_path=data_cfg['go_emb'],
        splits_path=data_cfg['splits'],
        protein_order_path=data_cfg['protein_order'],
        branch=config['branch'],
        split='train',
    )
    log.info(f"Train set: {len(dataset)} proteins")

    model = AMPRModel(
        d_hidden=config['model']['d_hidden'],
        n_terms=config['n_terms'],
        dropout_3di=config['model']['dropout_3di'],
        dropout_ppi=config['model']['dropout_ppi'],
        classifier=config['model']['classifier'],
        go_emb_dim=config['model']['go_emb_dim'],
        ppi_dim=config['model'].get('ppi_dim', 128),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model trainable params: {n_params:,}")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        logger_obj=log,
    )
    trainer.train()


if __name__ == '__main__':
    main()
