#!/usr/bin/env python
"""
AMPR: Adaptive Multimodal Protein Representation — Main training entry point.

Usage:
    python main.py --config configs/mf.yaml
    python main.py --config configs/bp.yaml
    python main.py --config configs/cc.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

from ampr.data.dataset import AMPRDataset
from ampr.models.ampr import AMPRModel
from ampr.training.trainer import Trainer


def setup_logging(log_file):
    """Configure logging to file + console."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter('[%(name)s] %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('ampr')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description='AMPR: Adaptive Multimodal Protein Representation')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config['output']['log_file'])
    logger.info(f"Loading config from {args.config}")
    logger.info(f"Branch: {config['branch']}")
    logger.info(f"Device: {config['training']['device']}")

    torch.manual_seed(args.seed)

    dataset = AMPRDataset(
        seq_emb_path=config['data']['seq_emb'],
        struct_emb_path=config['data']['struct_emb'],
        ppi_emb_path=config['data']['ppi_emb'],
        labels_path=config['data']['labels'],
        dag_matrix_path=config['data']['dag_matrix'],
        go_emb_path=config['data']['go_emb'],
        splits_path=config['data']['splits'],
        branch=config['branch'],
    )

    logger.info(f"Dataset loaded: {len(dataset)} proteins")

    model = AMPRModel(
        d_hidden=config['model']['d_hidden'],
        n_terms=config['n_terms'],
        dropout_3di=config['model']['dropout_3di'],
        dropout_ppi=config['model']['dropout_ppi'],
        classifier=config['model']['classifier'],
    )

    logger.info(f"Model created: {model}")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        logger=logger,
    )

    trainer.train()

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
