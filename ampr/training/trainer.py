"""Training loop for AMPR."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from ampr.data.dataset import get_dataloaders
from ampr.evaluation.metrics import compute_fmax, compute_all_metrics
from ampr.training.loss import AMPRLoss

logger = logging.getLogger('ampr')


class Trainer:
    """Train and evaluate AMPR model."""

    def __init__(self, model, dataset, config, logger_obj):
        self.config = config
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

        self.loss_fn = AMPRLoss(
            dag_matrix=dataset.dag_matrix_torch.to(self.device),
            lambda_dag=config['training'].get('lambda_dag', 0.5),
        )

        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']

        data_config = dict(config['data'])
        data_config['branch'] = config['branch']
        self.train_loader, self.val_loader, self.test_loader, _ = get_dataloaders(
            data_config, self.batch_size
        )
        self.logger.info(f"[TRAIN] Batches per epoch: {len(self.train_loader)}")

        self.go_emb = (
            dataset.go_emb_torch.to(self.device)
            if config['model']['classifier'] in ['biobert', 'both']
            else None
        )

        # Annotation-based IC for Smin: IC(t) = -log2(freq(t))
        labels_all = np.load(config['data']['labels'])
        freq = labels_all.mean(axis=0).clip(1e-7, 1.0)
        self.term_ic = (-np.log2(freq)).astype(np.float32)
        self.logger.info(
            f"[TRAIN] Term IC: mean={self.term_ic.mean():.2f}, "
            f"max={self.term_ic.max():.2f}, n_terms={len(self.term_ic)}"
        )

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

        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path(config['output']['results_file']).parent
        self.results_dir.mkdir(parents=True, exist_ok=True)

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

        # Save epoch-level training history
        branch = self.config['branch'].lower()
        history_path = self.results_dir / f'training_history_{branch}.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"[HISTORY] Saved → {history_path}")

        # Load best checkpoint for test evaluation
        best_ckpt = self.checkpoint_dir / 'best.pt'
        if best_ckpt.exists():
            core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            state = torch.load(best_ckpt, map_location=self.device)
            core.load_state_dict(state['model_state'])
            self.logger.info(
                f"[TEST] Loaded best checkpoint "
                f"(epoch={state['epoch']}, val Fmax={state['fmax']:.4f})"
            )

        # Full evaluation on test set
        test_metrics, y_true, y_pred = self._full_evaluate(self.test_loader)

        self.logger.info("[TEST] ─────────────────────────────────────")
        self.logger.info(f"[TEST]  Fmax        = {test_metrics['fmax']:.4f}"
                         f"  (threshold = {test_metrics['fmax_threshold']:.2f})")
        self.logger.info(f"[TEST]  AUPRC       = {test_metrics['auprc']:.4f}")
        self.logger.info(f"[TEST]  Smin        = {test_metrics['smin']:.4f}")
        self.logger.info(f"[TEST]  AUROC micro = {test_metrics['micro_auroc']:.4f}")
        self.logger.info(f"[TEST]  AUROC macro = {test_metrics['macro_auroc']:.4f}")
        self.logger.info(f"[TEST]  Coverage    = {test_metrics['coverage']:.4f}")
        self.logger.info("[TEST] ─────────────────────────────────────")

        # Save test metrics JSON
        metrics_path = self.results_dir / f'test_metrics_{branch}.json'
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        self.logger.info(f"[TEST] Metrics saved → {metrics_path}")

        # Save raw predictions (for offline plotting / analysis)
        pred_path = self.results_dir / f'test_predictions_{branch}.npz'
        np.savez_compressed(str(pred_path), y_true=y_true, y_pred=y_pred)
        self.logger.info(f"[TEST] Predictions saved → {pred_path}")

        # Generate plots
        try:
            from ampr.evaluation.plots import generate_all_plots
            plots_dir = self.results_dir / 'plots'
            generate_all_plots(history, y_true, y_pred, str(plots_dir), branch)
        except Exception as exc:
            self.logger.warning(f"[PLOT] Skipped (error: {exc})")

    # ------------------------------------------------------------------ #

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
            for batch in tqdm(loader, desc="Test Eval", leave=False):
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