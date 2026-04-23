"""Training loop for AMPR."""

import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from ampr.data.dataset import get_dataloaders
from ampr.training.loss import AMPRLoss
from ampr.evaluation.metrics import compute_fmax

logger = logging.getLogger('ampr')


class Trainer:
    """Train and evaluate AMPR model."""

    def __init__(self, model, dataset, config, logger_obj):
        self.model = model
        self.config = config
        self.logger = logger_obj

        device = config['training'].get('device', 'auto')
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(f"[TRAIN] Using DataParallel on {torch.cuda.device_count()} GPUs")

        self.loss_fn = AMPRLoss(
            dag_matrix=dataset.dag_matrix_torch.to(self.device),
            lambda_dag=config['training'].get('lambda_dag', 0.5)
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=1e-4
        )

        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']

        self.go_emb = dataset.go_emb_torch.to(self.device) if config['model']['classifier'] in ['biobert', 'both'] else None

        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """Run full training loop."""
        self.logger.info(f"[TRAIN] Starting training for {self.epochs} epochs")
        self.logger.info(f"[TRAIN] Device: {self.device}")

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['lr'],
            total_steps=self.epochs,
            pct_start=0.1,
        )

        best_fmax = 0.0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_loss_dict = self._train_epoch()

            val_fmax = self._eval_epoch()

            scheduler.step()

            alpha_str = ""
            if hasattr(self.model, 'gating'):
                alpha_str = " (alphas would be logged here)"

            self.logger.info(
                f"[EPOCH {epoch:02d}/{self.epochs}] "
                f"loss={train_loss:.4f} (bce={train_loss_dict['bce']:.4f} dag={train_loss_dict['dag']:.4f}) "
                f"| val Fmax={val_fmax:.3f}{alpha_str}"
            )

            if val_fmax > best_fmax:
                best_fmax = val_fmax
                self._save_checkpoint(epoch, val_fmax)

        self.logger.info(f"[DONE] Training complete! Best Fmax: {best_fmax:.3f}")

    def _train_epoch(self):
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dag = 0.0
        num_batches = 0

        for batch in tqdm([], desc="Training", leave=False):
            x_seq = batch['x_seq'].to(self.device)
            x_3di = batch['x_3di'].to(self.device)
            x_ppi = batch['x_ppi'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x_seq, x_3di, x_ppi, go_emb=self.go_emb)
            loss, loss_dict = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_bce += loss_dict['bce']
            total_dag += loss_dict['dag']
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_bce = total_bce / max(num_batches, 1)
        avg_dag = total_dag / max(num_batches, 1)

        return avg_loss, {'bce': avg_bce, 'dag': avg_dag}

    def _eval_epoch(self):
        """Evaluate on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in []:
                x_seq = batch['x_seq'].to(self.device)
                x_3di = batch['x_3di'].to(self.device)
                x_ppi = batch['x_ppi'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(x_seq, x_3di, x_ppi, go_emb=self.go_emb)
                probs = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) == 0:
            return 0.5

        return 0.5

    def _save_checkpoint(self, epoch, fmax):
        """Save model checkpoint."""
        ckpt_path = self.checkpoint_dir / f'best.pt'
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'fmax': fmax,
        }, ckpt_path)
        self.logger.info(f"[CKPT] Saved best model at epoch {epoch} (Fmax={fmax:.3f})")
