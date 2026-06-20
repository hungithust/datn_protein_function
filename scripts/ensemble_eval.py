#!/usr/bin/env python
"""Multi-seed ensemble eval for v3 checkpoints.

Averages sigmoid probabilities across several trained checkpoints (e.g. different
seeds of the same config), then DAG-propagates and optionally blends with the
DIAMOND homology ensemble — identical metric pipeline to main.py `_eval_v3`.

Usage:
  python scripts/ensemble_eval.py --config configs/mf_v3_esm3b.yaml \
      --checkpoints checkpoints/mf_v3_esm3b/best.pt checkpoints/mf_s123/best.pt \
                    checkpoints/mf_s2024/best.pt \
      --split test_LT_95
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ampr.data.dataset import AMPRDatasetV3, collate_variable_length
from ampr.models.ampr import AMPRModelV3
from ampr.evaluation.dag_inference import propagate_scores_upward
from ampr.evaluation.metrics import compute_all_metrics


def build_model(cfg, go_emb_dim, device):
    m = cfg['model']
    seq, gnn = m.get('seq', {}), m.get('gnn', {})
    ppi, fus = m.get('ppi', {}), m.get('fusion', {})
    contr = (cfg['training'].get('contrastive', {}) or {})
    return AMPRModelV3(
        n_terms=cfg['n_terms'], seq_dim=seq.get('d_model', 1280),
        seq_n_heads=seq.get('n_heads', 8), seq_n_layers=seq.get('n_transformer_layers', 2),
        gnn_node_dim=gnn.get('node_dim', 256), gnn_n_layers=gnn.get('n_layers', 3),
        ppi_dim=ppi.get('in_dim', 256), d_hidden=m.get('d_hidden', 512),
        fusion_n_heads=fus.get('n_heads', 8), fusion_n_layers=fus.get('n_layers', 2),
        classifier=m.get('classifier', 'both'), go_emb_dim=go_emb_dim,
        cmap_threshold=gnn.get('cmap_threshold', 10.0), dropout=seq.get('dropout', 0.1),
        contrastive_proj_dim=contr.get('proj_dim', 0)).to(device)


@torch.no_grad()
def infer(model, loader, go_emb, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        bd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(bd, go_emb=go_emb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch['labels'].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--split', default='test_LT_95')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg, train_cfg = cfg['data'], cfg['training']
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

    # Average sigmoid probs across all checkpoints.
    acc, labels = None, None
    for ckpt_path in args.checkpoints:
        model = build_model(cfg, go_emb.shape[1], device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
        p, labels = infer(model, loader, go_emb, device)
        acc = p if acc is None else acc + p
        print(f"[ENS] loaded {ckpt_path}")
    probs = acc / len(args.checkpoints)

    labels_all = np.load(data_cfg['labels'])
    term_ic = (-np.log2(labels_all.mean(axis=0).clip(1e-7, 1.0))).astype('float32')

    # raw (no DAG propagation) — for the DAG-propagation ablation
    m_raw = compute_all_metrics(labels, probs, term_ic)
    print(f"[ENS][raw] n_seeds={len(args.checkpoints)} Fmax={m_raw['fmax']:.4f} "
          f"Smin={m_raw['smin']:.4f} AUPRC_macro={m_raw['auprc_macro']:.4f}")

    probs_dag = propagate_scores_upward(probs, dag_np)
    m_dag = compute_all_metrics(labels, probs_dag, term_ic)
    print(f"[ENS][dag] n_seeds={len(args.checkpoints)} Fmax={m_dag['fmax']:.4f} "
          f"Smin={m_dag['smin']:.4f} AUPRC_macro={m_dag['auprc_macro']:.4f}")
    out = {'split': args.split, 'n_seeds': len(args.checkpoints),
           'checkpoints': args.checkpoints, 'raw': m_raw, 'dag': m_dag}

    inf_cfg = cfg.get('inference', {})
    diamond_tsv = data_cfg.get('diamond_tsv')
    if inf_cfg.get('use_diamond_ensemble', False) and diamond_tsv and Path(diamond_tsv).exists():
        from ampr.evaluation.diamond_ensemble import compute_diamond_scores, ensemble_scores
        with open(data_cfg['splits']) as f:
            train_ids = json.load(f).get('train', [])
        train_order = {p: ds._prot2idx[p] for p in train_ids if p in ds._prot2idx}
        alpha = float(inf_cfg.get('diamond_alpha', 0.6))
        diamond_probs = compute_diamond_scores(
            diamond_tsv, ds.labels, train_order, ds.protein_ids, cfg['n_terms'])
        n_hom = int((diamond_probs.sum(axis=1) > 0).sum())
        ens_dag = propagate_scores_upward(ensemble_scores(probs, diamond_probs, alpha), dag_np)
        m_ens = compute_all_metrics(labels, ens_dag, term_ic)
        print(f"[ENS][ens] alpha={alpha} hom_hits={n_hom}/{labels.shape[0]} "
              f"Fmax={m_ens['fmax']:.4f} Smin={m_ens['smin']:.4f} "
              f"AUPRC_macro={m_ens['auprc_macro']:.4f}")
        out['ensemble'] = m_ens

    res = Path(cfg['output']['results_file']).with_suffix(f'.ensemble_{args.split}.json')
    res.write_text(json.dumps(out, indent=2))
    print(f"[ENS] wrote {res}")


if __name__ == '__main__':
    main()
