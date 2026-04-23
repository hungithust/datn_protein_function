"""Training loop for AMPR."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from ampr.data.dataset import get_dataloaders
from ampr.evaluation.metrics import compute_fmax
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

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=1e-4,
        )

        # OneCycleLR steps once per batch, total = epochs * batches
        total_steps = self.epochs * len(self.train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['training']['lr'],
            total_steps=total_steps,
            pct_start=0.1,
        )

        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """Run full training loop."""
        self.logger.info(f"[TRAIN] Starting: {self.epochs} epochs × {len(self.train_loader)} batches")

        best_fmax = 0.0

        for epoch in range(1, self.epochs + 1):
            train_loss, loss_dict, mean_alphas = self._train_epoch()
            val_fmax = self._eval_epoch(self.val_loader)

            alpha_str = ""
            if mean_alphas is not None:
                alpha_str = (f" | α=[{mean_alphas[0]:.3f}, "
                             f"{mean_alphas[1]:.3f}, {mean_alphas[2]:.3f}]")

            self.logger.info(
                f"[EPOCH {epoch:02d}/{self.epochs}] "
                f"loss={train_loss:.4f} (bce={loss_dict['bce']:.4f} dag={loss_dict['dag']:.4f})"
                f"{alpha_str} | val Fmax={val_fmax:.3f}"
            )

            if val_fmax > best_fmax:
                best_fmax = val_fmax
                self._save_checkpoint(epoch, val_fmax)

        self.logger.info(f"[DONE] Training complete. Best val Fmax: {best_fmax:.3f}")

        test_fmax = self._eval_epoch(self.test_loader)
        self.logger.info(f"[TEST] Fmax={test_fmax:.3f}")

    def _train_epoch(self):
        """Single training epoch. Returns (avg_loss, loss_dict, mean_alphas)."""
        self.model.train()
        total_loss = total_bce = total_dag = 0.0
        alpha_accumulator = []

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            x_seq   = batch['x_seq'].to(self.device)
            x_3di   = batch['x_3di'].to(self.device)
            x_ppi   = batch['x_ppi'].to(self.device)
            labels  = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # return_alphas works on bare model or DataParallel-wrapped model
            core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            logits, alphas = core(x_seq, x_3di, x_ppi, go_emb=self.go_emb, return_alphas=True)

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

        return (total_loss / n,
                {'bce': total_bce / n, 'dag': total_dag / n},
                mean_alphas)

    def _eval_epoch(self, loader):
        """Evaluate on given loader. Returns Fmax."""
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

    def _save_checkpoint(self, epoch, fmax):
        ckpt_path = self.checkpoint_dir / 'best.pt'
        core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save({
            'epoch': epoch,
            'model_state': core.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'fmax': fmax,
        }, ckpt_path)
        self.logger.info(f"[CKPT] Saved epoch {epoch} — Fmax={fmax:.3f} → {ckpt_path}")
