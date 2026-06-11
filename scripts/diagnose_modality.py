#!/usr/bin/env python
"""Eval-only modality ablation for a trained v3 checkpoint.

For each ablation set (none, -gnn, -ppi, seq-only) evaluate Fmax on a split by
zeroing the named branch(es) before fusion. No retraining — this is a lower
bound on each modality's contribution (see spec §2 caveat).

Usage:
  python scripts/diagnose_modality.py --config configs/mf_v3_esm3b.yaml \
      --checkpoint checkpoints/mf_v3_esm3b/best.pt --split test_LT_95
"""
import argparse
import json
import numpy as np
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from ampr.data.dataset import AMPRDatasetV3, collate_variable_length
from ampr.models.ampr import AMPRModelV3
from ampr.evaluation.dag_inference import propagate_scores_upward
from ampr.evaluation.metrics import compute_fmax

ABLATIONS = {'full': (), '-gnn': ('gnn',), '-ppi': ('ppi',),
             'seq_only': ('gnn', 'ppi')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='test_LT_95')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg, model_cfg, train_cfg = cfg['data'], cfg['model'], cfg['training']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = AMPRDatasetV3(
        esm2_h5=data_cfg['esm2_h5'], ppi_emb=data_cfg['ppi_emb'],
        ppi_mask=data_cfg['ppi_mask'], cmap_h5=data_cfg['cmap_h5'],
        labels=data_cfg['labels'], dag_matrix=data_cfg['dag_matrix'],
        go_emb=data_cfg['go_emb'], splits=data_cfg['splits'],
        protein_order=data_cfg['protein_order'], branch=cfg['branch'],
        split=args.split, max_len=train_cfg.get('max_seq_len', 1000))
    loader = DataLoader(ds, batch_size=train_cfg['batch_size'], shuffle=False,
                        collate_fn=collate_variable_length, num_workers=0)
    go_emb = ds.go_emb.to(device)
    dag_np = ds.dag_matrix.numpy()

    seq_cfg = model_cfg.get('seq', {}); gnn_cfg = model_cfg.get('gnn', {})
    ppi_cfg = model_cfg.get('ppi', {}); fusion_cfg = model_cfg.get('fusion', {})
    model = AMPRModelV3(
        n_terms=cfg['n_terms'], seq_dim=seq_cfg.get('d_model', 1280),
        seq_n_heads=seq_cfg.get('n_heads', 8),
        seq_n_layers=seq_cfg.get('n_transformer_layers', 2),
        gnn_node_dim=gnn_cfg.get('node_dim', 256),
        gnn_n_layers=gnn_cfg.get('n_layers', 3), ppi_dim=ppi_cfg.get('in_dim', 256),
        d_hidden=model_cfg.get('d_hidden', 512),
        fusion_n_heads=fusion_cfg.get('n_heads', 8),
        fusion_n_layers=fusion_cfg.get('n_layers', 2),
        classifier=model_cfg.get('classifier', 'both'), go_emb_dim=go_emb.shape[1],
        cmap_threshold=gnn_cfg.get('cmap_threshold', 10.0),
        dropout=seq_cfg.get('dropout', 0.1)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
    model.eval()

    results = {}
    for name, ab in ABLATIONS.items():
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                bd = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in batch.items()}
                logits = model(bd, go_emb=go_emb, ablate=ab)
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
                labels_list.append(batch['labels'].numpy())
        probs = propagate_scores_upward(np.concatenate(probs_list), dag_np)
        fmax, _ = compute_fmax(np.concatenate(labels_list), probs)
        results[name] = round(float(fmax), 4)
        print(f"[ABLATE] {name:9s} Fmax={fmax:.4f}")

    out = Path(cfg['output']['results_file']).with_suffix(f'.ablate_{args.split}.json')
    out.write_text(json.dumps({'split': args.split, 'fmax': results}, indent=2))
    print(f"[ABLATE] wrote {out}")


if __name__ == '__main__':
    main()
