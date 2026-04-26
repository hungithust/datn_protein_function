"""Training loop for AMPR."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ampr.data.dataset import AMPRDataset, get_dataloaders
from ampr.evaluation.metrics import compute_fmax, compute_all_metrics
from ampr.training.loss import AMPRLoss

logger = logging.getLogger('ampr')


class Trainer:
    """Train and evaluate AMPR model."""

    def __init__(self, model, dataset, config, logger_obj, eval_only=False):
        """
        Args:
            eval_only: if True, skip building train/val/test loaders, optimizer,
                       and scheduler. Use evaluate_split() to run eval on demand.
        """
        self.config = config
        self.dataset = dataset
        self.logger = logger_obj

        device_cfg = config['training'].get('device', 'auto')
        self.device = 'cuda' if (device_cfg == 'auto' and torch.cuda.is_available()) else (
            'cuda' if device_cfg == 'cuda' else 'cpu'
        )
        self.logger.info(f"[TRAIN] Device: {self.device}")

        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"[TRAIN] DataParallel on {torch.cuda.device_count()} GPUs")

        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']

        data_config = dict(config['data'])
        data_config['branch'] = config['branch']
        self.data_config = data_config

        self.go_emb = (
            dataset.go_emb_torch.to(self.device)
            if config['model']['classifier'] in ['biobert', 'both']
            else None
        )

        labels_all = np.load(config['data']['labels'])
        freq = labels_all.mean(axis=0).clip(1e-7, 1.0)
        self.term_ic = (-np.log2(freq)).astype(np.float32)
        self.logger.info(
            f"[TRAIN] Term IC: mean={self.term_ic.mean():.2f}, "
            f"max={self.term_ic.max():.2f}, n_terms={len(self.term_ic)}"
        )

        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(config['output']['results_file']).parent
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Heavy training-only setup: skip in eval-only mode
        if eval_only:
            self.train_loader = self.val_loader = self.test_loader = None
            self.loss_fn = self.optimizer = self.scheduler = None
            return

        self.loss_fn = AMPRLoss(
            dag_matrix=dataset.dag_matrix_torch.to(self.device),
            lambda_dag=config['training'].get('lambda_dag', 0.5),
        )

        self.train_loader, self.val_loader, self.test_loader, _ = get_dataloaders(
            data_config, self.batch_size
        )
        self.logger.info(f"[TRAIN] Batches per epoch: {len(self.train_loader)}")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=1e-4,
        )

        total_steps = self.epochs * len(self.train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['training']['lr'],
            total_steps=total_steps,
            pct_start=0.1,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def train(self):
        """Run full training loop, then evaluate on test set with all metrics."""
        self.logger.info(
            f"[TRAIN] Starting: {self.epochs} epochs × {len(self.train_loader)} batches"
        )

        best_fmax = 0.0
        history = []

        for epoch in range(1, self.epochs + 1):
            train_loss, loss_dict, mean_alphas = self._train_epoch()
            val_fmax = self._eval_fmax(self.val_loader)

            alpha_str = ""
            if mean_alphas is not None:
                alpha_str = (
                    f" | α=[{mean_alphas[0]:.3f}, "
                    f"{mean_alphas[1]:.3f}, {mean_alphas[2]:.3f}]"
                )

            self.logger.info(
                f"[EPOCH {epoch:02d}/{self.epochs}] "
                f"loss={train_loss:.4f} "
                f"(bce={loss_dict['bce']:.4f} dag={loss_dict['dag']:.4f})"
                f"{alpha_str} | val Fmax={val_fmax:.4f}"
            )

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'bce': loss_dict['bce'],
                'dag': loss_dict['dag'],
                'val_fmax': val_fmax,
                'alphas': mean_alphas.tolist() if mean_alphas is not None else [1/3, 1/3, 1/3],
            })

            if val_fmax > best_fmax:
                best_fmax = val_fmax
                self._save_checkpoint(epoch, val_fmax)

        self.logger.info(f"[DONE] Training complete. Best val Fmax: {best_fmax:.4f}")

        branch = self.config['branch'].lower()
        history_path = self.results_dir / f'training_history_{branch}.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"[HISTORY] Saved → {history_path}")

        # Load best checkpoint and evaluate on full 'test' split
        self._load_best_checkpoint()
        self.evaluate_split('test', history=history)

    def evaluate_split(self, split_name, history=None, checkpoint_path=None):
        """
        Run full evaluation on a named split (e.g. 'test', 'test_LT_30').
        If checkpoint_path is given, load it first; otherwise assume model is ready.
        Saves split-suffixed metrics, predictions, and plots.
        """
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        loader = self._make_loader_for_split(split_name)
        if loader is None:
            self.logger.warning(f"[EVAL] Skipping {split_name}: split is empty or missing")
            return None

        metrics, y_true, y_pred = self._full_evaluate(loader)

        branch = self.config['branch'].lower()
        suffix = self._suffix_for_split(split_name)
        suffix_part = f'_{suffix}' if suffix else ''

        self.logger.info(f"[EVAL] ─── Split: {split_name} ─── ({metrics['n_proteins']} proteins, "
                         f"{metrics['n_terms_with_positives']} terms with positives)")
        self.logger.info(f"[EVAL]  Fmax        = {metrics['fmax']:.4f}  (t={metrics['fmax_threshold']:.2f})")
        self.logger.info(f"[EVAL]  AUPR micro  = {metrics['auprc_micro']:.4f}  (DeepFRI metric)")
        self.logger.info(f"[EVAL]  AUPR macro  = {metrics['auprc_macro']:.4f}")
        self.logger.info(f"[EVAL]  Smin        = {metrics['smin']:.4f}")
        self.logger.info(f"[EVAL]  AUROC micro = {metrics['micro_auroc']:.4f}")
        self.logger.info(f"[EVAL]  AUROC macro = {metrics['macro_auroc']:.4f}")
        self.logger.info(f"[EVAL]  Coverage    = {metrics['coverage']:.4f}")

        metrics_path = self.results_dir / f'test_metrics_{branch}{suffix_part}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"[EVAL] Metrics saved → {metrics_path}")

        pred_path = self.results_dir / f'test_predictions_{branch}{suffix_part}.npz'
        np.savez_compressed(str(pred_path), y_true=y_true, y_pred=y_pred)
        self.logger.info(f"[EVAL] Predictions saved → {pred_path}")

        try:
            from ampr.evaluation.plots import generate_all_plots
            plots_dir = self.results_dir / 'plots'
            generate_all_plots(history, y_true, y_pred, str(plots_dir),
                               branch, suffix=suffix)
        except Exception as exc:
            self.logger.warning(f"[PLOT] Skipped (error: {exc})")

        return metrics

    # ── Internals ──────────────────────────────────────────────────────────

    def _suffix_for_split(self, split_name):
        if split_name in (None, '', 'test'):
            return ''
        return split_name.replace('test_', '')

    def _make_loader_for_split(self, split_name):
        """Build a DataLoader for an arbitrary split key in splits.json."""
        ds = AMPRDataset(
            seq_emb_path=self.data_config['seq_emb'],
            struct_emb_path=self.data_config['struct_emb'],
            ppi_emb_path=self.data_config['ppi_emb'],
            labels_path=self.data_config['labels'],
            dag_matrix_path=self.data_config['dag_matrix'],
            go_emb_path=self.data_config['go_emb'],
            splits_path=self.data_config['splits'],
            protein_order_path=self.data_config['protein_order'],
            branch=self.data_config['branch'],
            split=split_name,
        )
        if len(ds) == 0:
            return None
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    def _load_best_checkpoint(self):
        ckpt_path = self.checkpoint_dir / 'best.pt'
        if ckpt_path.exists():
            self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path):
        core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state = torch.load(str(ckpt_path), map_location=self.device)
        core.load_state_dict(state['model_state'])
        epoch = state.get('epoch', '?')
        fmax  = state.get('fmax', float('nan'))
        self.logger.info(f"[CKPT] Loaded {ckpt_path} (epoch={epoch}, val Fmax={fmax:.4f})")

    def _train_epoch(self):
        """Single training epoch. Returns (avg_loss, loss_dict, mean_alphas)."""
        self.model.train()
        total_loss = total_bce = total_dag = 0.0
        alpha_accumulator = []

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            x_seq  = batch['x_seq'].to(self.device)
            x_3di  = batch['x_3di'].to(self.device)
            x_ppi  = batch['x_ppi'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            logits, alphas = core(
                x_seq, x_3di, x_ppi, go_emb=self.go_emb, return_alphas=True
            )

            loss, loss_dict = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_bce  += loss_dict['bce']
            total_dag  += loss_dict['dag']
            alpha_accumulator.append(alphas.detach().mean(dim=0).cpu().numpy())

        n = max(len(self.train_loader), 1)
        mean_alphas = np.stack(alpha_accumulator).mean(axis=0) if alpha_accumulator else None

        return (
            total_loss / n,
            {'bce': total_bce / n, 'dag': total_dag / n},
            mean_alphas,
        )

    def _eval_fmax(self, loader):
        """Quick evaluation — returns Fmax only (used each epoch for val)."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                x_seq  = batch['x_seq'].to(self.device)
                x_3di  = batch['x_3di'].to(self.device)
                x_ppi  = batch['x_ppi'].to(self.device)
                labels = batch['labels'].to(self.device)

                core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                logits = core(x_seq, x_3di, x_ppi, go_emb=self.go_emb)
                probs  = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_preds:
            return 0.0

        fmax, _ = compute_fmax(
            np.concatenate(all_labels, axis=0),
            np.concatenate(all_preds,  axis=0),
        )
        return fmax

    def _full_evaluate(self, loader):
        """Full evaluation with all metrics. Returns (metrics_dict, y_true, y_pred)."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                x_seq  = batch['x_seq'].to(self.device)
                x_3di  = batch['x_3di'].to(self.device)
                x_ppi  = batch['x_ppi'].to(self.device)
                labels = batch['labels'].to(self.device)

                core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                logits = core(x_seq, x_3di, x_ppi, go_emb=self.go_emb)
                probs  = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds,  axis=0)

        metrics = compute_all_metrics(y_true, y_pred, self.term_ic)
        return metrics, y_true, y_pred

    def _save_checkpoint(self, epoch, fmax):
        ckpt_path = self.checkpoint_dir / 'best.pt'
        core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save({
            'epoch': epoch,
            'model_state': core.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'fmax': fmax,
        }, ckpt_path)
        self.logger.info(f"[CKPT] Saved epoch {epoch} — Fmax={fmax:.4f} → {ckpt_path}")
