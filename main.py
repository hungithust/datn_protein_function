#!/usr/bin/env python
"""
AMPR: Adaptive Multimodal Protein Representation — entry point.

Usage (training, full pipeline):
    python main.py --config configs/mf.yaml

Usage (v3 config):
    python main.py --config configs/mf_v3.yaml [--dry_run]

Usage (eval only, on a specific test split):
    python main.py --config configs/mf.yaml \
        --eval-only \
        --checkpoint checkpoints/mf/best.pt \
        --test-split test_LT_30
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


def _resolve_device(cfg_device: str) -> str:
    if cfg_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg_device


def _run_v3(config: dict, args, log):
    """V3 dispatch: ESM-2 residue + GNN cmap + DeepGO PPI + CrossModalFusion."""
    from torch.utils.data import DataLoader

    from ampr.data.dataset import AMPRDatasetV3, collate_variable_length
    from ampr.models.ampr import AMPRModelV3
    from ampr.training.loss import AMPRLoss
    from ampr.training.trainer import train_one_epoch_v3
    from ampr.evaluation.dag_inference import propagate_scores_upward
    from ampr.evaluation.metrics import compute_fmax

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    out_cfg = config['output']

    device = _resolve_device(train_cfg.get('device', 'auto'))
    log.info(f"[V3] device={device}")

    def mk_dataset(split):
        return AMPRDatasetV3(
            esm2_h5=data_cfg['esm2_h5'],
            ppi_emb=data_cfg['ppi_emb'],
            ppi_mask=data_cfg['ppi_mask'],
            cmap_h5=data_cfg['cmap_h5'],
            labels=data_cfg['labels'],
            dag_matrix=data_cfg['dag_matrix'],
            go_emb=data_cfg['go_emb'],
            splits=data_cfg['splits'],
            protein_order=data_cfg['protein_order'],
            branch=config['branch'],
            split=split,
            max_len=train_cfg.get('max_seq_len', 1000),
        )

    ds_train = mk_dataset('train')
    ds_val = mk_dataset('valid')

    if args.dry_run:
        ds_train.protein_ids = ds_train.protein_ids[:50]
        ds_val.protein_ids = ds_val.protein_ids[:20]
        log.info(f"[V3] --dry_run: train={len(ds_train)}, val={len(ds_val)}")

    num_workers = train_cfg.get('num_workers', 0)
    ld_train = DataLoader(ds_train, batch_size=train_cfg['batch_size'],
                          shuffle=True, collate_fn=collate_variable_length,
                          num_workers=num_workers, pin_memory=(device == 'cuda'))
    ld_val = DataLoader(ds_val, batch_size=train_cfg['batch_size'],
                        shuffle=False, collate_fn=collate_variable_length,
                        num_workers=num_workers, pin_memory=(device == 'cuda'))

    # Load go_emb as tensor
    go_emb = ds_train.go_emb.to(device)
    go_emb_dim = go_emb.shape[1]
    # [DIAG] go_emb chưa chuẩn hoá -> logit bio = proj(z) @ go_emb.t() có thể bão hoà
    log.info(f"[DIAG] go_emb shape={tuple(go_emb.shape)} "
             f"row_norm_mean={go_emb.norm(dim=1).mean().item():.3f} "
             f"abs_max={go_emb.abs().max().item():.3f}")

    seq_cfg = model_cfg.get('seq', {})
    gnn_cfg = model_cfg.get('gnn', {})
    ppi_cfg = model_cfg.get('ppi', {})
    fusion_cfg = model_cfg.get('fusion', {})

    model = AMPRModelV3(
        n_terms=config['n_terms'],
        seq_dim=seq_cfg.get('d_model', 1280),
        seq_n_heads=seq_cfg.get('n_heads', 8),
        seq_n_layers=seq_cfg.get('n_transformer_layers', 2),
        gnn_node_dim=gnn_cfg.get('node_dim', 256),
        gnn_n_layers=gnn_cfg.get('n_layers', 3),
        ppi_dim=ppi_cfg.get('in_dim', 256),
        d_hidden=model_cfg.get('d_hidden', 512),
        fusion_n_heads=fusion_cfg.get('n_heads', 8),
        fusion_n_layers=fusion_cfg.get('n_layers', 2),
        classifier=model_cfg.get('classifier', 'both'),
        go_emb_dim=go_emb_dim,
        cmap_threshold=gnn_cfg.get('cmap_threshold', 10.0),
        dropout=seq_cfg.get('dropout', 0.1),
    )

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[V3] Trainable params: {n_params:,}")

    dag_np = ds_train.dag_matrix.numpy()

    # pos_weight per-class từ nhãn TRAIN (chỉ dùng cho loss bce). Cap để tránh
    # các term hiếm tạo trọng số quá lớn gây bất ổn.
    pos_weight = None
    if train_cfg.get('loss_type', 'asl') == 'bce':
        train_rows = [ds_train._prot2idx[p] for p in ds_train.protein_ids]
        y_train = ds_train.labels[train_rows]                      # (n_train, C)
        pos = y_train.sum(axis=0)                                  # số dương / term
        neg = y_train.shape[0] - pos
        pw_cap = float(train_cfg.get('pos_weight_cap', 50.0))
        pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, pw_cap).astype('float32')
        log.info(f"[V3] pos_weight per-class: mean={pos_weight.mean():.2f} "
                 f"min={pos_weight.min():.2f} max={pos_weight.max():.2f} cap={pw_cap}")

    loss_fn = AMPRLoss(
        ds_train.dag_matrix,
        lambda_dag=train_cfg.get('lambda_dag', 0.5),
        loss_type=train_cfg.get('loss_type', 'asl'),
        asl_gamma_neg=train_cfg.get('asl_gamma_neg', 4.0),
        asl_gamma_pos=train_cfg.get('asl_gamma_pos', 0.0),
        asl_clip=train_cfg.get('asl_clip', 0.05),
        pos_weight=pos_weight,
    )
    loss_fn = loss_fn.to(device)  # move dag_matrix + pos_weight buffers to GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get('lr', 1e-3))
    grad_clip = float(train_cfg.get('grad_clip', 0.0))  # 0 = off

    ckpt_dir = Path(out_cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_fmax_dag = -1.0
    epochs = 1 if args.dry_run else train_cfg.get('epochs', 50)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_v3(model, ld_train, loss_fn, optimizer,
                                        go_emb=go_emb, device=device,
                                        grad_clip=grad_clip)

        # Val inference
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in ld_val:
                batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                logits = model(batch_dev, go_emb=go_emb)
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
                labels_list.append(batch['labels'].numpy())

        if probs_list:
            probs = np.concatenate(probs_list)
            labels = np.concatenate(labels_list)
            # [DIAG] collapse check: std giữa các protein cho mỗi class ~0 => model
            # xuất gần như cùng 1 vector cho mọi protein (chỉ học class-prior).
            log.info(
                f"[DIAG] probs mean={probs.mean():.4f} global_std={probs.std():.4f} "
                f"cross_protein_std={probs.std(axis=0).mean():.6f} "
                f"per_protein_std={probs.std(axis=1).mean():.6f} "
                f"frac>0.5={(probs > 0.5).mean():.4f} | label_pos_rate={labels.mean():.4f}"
            )
            probs_dag = propagate_scores_upward(probs, dag_np)
            fmax_raw, _ = compute_fmax(labels, probs)
            fmax_dag, _ = compute_fmax(labels, probs_dag)
        else:
            fmax_raw = fmax_dag = 0.0

        log.info(f"[V3] Epoch {epoch}/{epochs}: loss={train_loss:.4f} "
                 f"val_Fmax_raw={fmax_raw:.4f} val_Fmax_dag={fmax_dag:.4f}")

        if fmax_dag > best_fmax_dag:
            best_fmax_dag = fmax_dag
            ckpt_path = ckpt_dir / 'best.pt'
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'fmax_dag': fmax_dag, 'fmax_raw': fmax_raw},
                       str(ckpt_path))
            log.info(f"[V3] Saved best checkpoint (fmax_dag={fmax_dag:.4f})")

    log.info(f"[V3] Training complete. Best val Fmax (DAG): {best_fmax_dag:.4f}")


def main():
    parser = argparse.ArgumentParser(description='AMPR training / evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load checkpoint and evaluate on --test-split')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (defaults to <checkpoint_dir>/best.pt)')
    parser.add_argument('--test-split', type=str, default='test',
                        help='splits.json key to evaluate (test, test_LT_30, test_LT_95, ...)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Limit to ~50 proteins for quick smoke check (v3 only)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    log = setup_logging(config['output']['log_file'])
    log.info(f"Config: {args.config}")
    log.info(f"Branch: {config['branch']} | n_terms: {config['n_terms']}")

    seed_everything(args.seed)
    log.info(f"Seed: {args.seed}")

    # --- V3 dispatch ---
    model_version = config.get('model', {}).get('version', 'v1')
    is_v3 = (model_version == 'v3') or ('esm2_h5' in config.get('data', {}))
    if is_v3:
        log.info("[V3] Detected v3 config — using AMPRModelV3 pipeline")
        _run_v3(config, args, log)
        return

    # --- V1/V2 legacy path ---
    log.info(f"Mode: {'EVAL-ONLY' if args.eval_only else 'TRAIN+EVAL'}")
    if args.eval_only:
        log.info(f"Eval split: {args.test_split}")
    log.info(f"Device: {config['training']['device']}")

    data_cfg = config['data']
    bootstrap_split = args.test_split if args.eval_only else 'train'
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
        split=bootstrap_split,
        use_cmap=data_cfg.get('use_cmap', False),
        cmap_h5_paths=data_cfg.get('cmap_h5'),
    )
    log.info(f"{bootstrap_split.capitalize()} set: {len(dataset)} proteins")

    model = AMPRModel(
        d_hidden=config['model']['d_hidden'],
        n_terms=config['n_terms'],
        dropout_3di=config['model']['dropout_3di'],
        dropout_ppi=config['model']['dropout_ppi'],
        classifier=config['model']['classifier'],
        go_emb_dim=config['model']['go_emb_dim'],
        ppi_dim=config['model'].get('ppi_dim', 128),
        structure_modality=config['model'].get('structure_modality', 'prostt5'),
        gnn_input_dim=config['model'].get('gnn_input_dim', 26),
        gnn_hidden_dim=config['model'].get('gnn_hidden_dim', 256),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model trainable params: {n_params:,}")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        logger_obj=log,
        eval_only=args.eval_only,
    )

    if args.eval_only:
        ckpt_path = args.checkpoint or str(Path(config['output']['checkpoint_dir']) / 'best.pt')
        trainer.evaluate_split(args.test_split, history=None, checkpoint_path=ckpt_path)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
