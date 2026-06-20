# AMPR v4 — Phase 0 Diagnostic + Module A (Contrastive) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-label supervised-contrastive auxiliary loss (Module A) to the v3 pipeline to close the val→test generalization gap, plus cheap Phase-0 diagnostics (modality ablation + baseline-number fix) — all runnable within the ~24h GPU budget.

**Architecture:** Module A adds a projection head on the fused representation `z` and a Jaccard-weighted SupCon loss term `L = L_ASL + λ_dag·L_DAG + λ_cl·L_CL`, toggled by a `training.contrastive` config block. Phase 0 adds a `ablate` hook to `AMPRModelV3.forward` + a diagnostic script (no retraining), and fixes the DeepFRI baseline number in the results doc.

**Tech Stack:** PyTorch, existing v3 pipeline (`AMPRModelV3`, `train_one_epoch_v3`, `_run_v3`), pytest. Windows: run tests with `python -m pytest` via PowerShell (Anaconda python), per [CLAUDE.md](../../../CLAUDE.md).

**Spec:** [2026-06-11-ampr-v4-breakthrough-design.md](../specs/2026-06-11-ampr-v4-breakthrough-design.md)

**Scope note:** This plan covers Phase 0 + Module A only (priority 1, no precompute, fits 24h). Module C (term-conditioned pooling) and Module B (domain-guided) get their own plans after Module A results land.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `ampr/training/contrastive.py` | `MultiLabelSupConLoss` | Create |
| `ampr/models/ampr.py` | `AMPRModelV3`: add `return_z`, `ablate`, contrastive proj head | Modify |
| `ampr/training/trainer.py` | `train_one_epoch_v3`: optional contrastive term | Modify |
| `main.py` | `_run_v3`: wire `training.contrastive` block | Modify |
| `scripts/diagnose_modality.py` | Eval-only modality ablation on a v3 checkpoint | Create |
| `configs/{mf,cc,bp}_v4_supcon.yaml` | v3 configs + contrastive block | Create |
| `docs/RESULTS_DATA.md` | Fix/annotate DeepFRI baseline number | Modify |
| `tests/test_contrastive.py` | Unit tests for SupCon loss | Create |
| `tests/test_ampr_v3_return_z.py` | Unit tests for `return_z`/`ablate` | Create |
| `tests/test_trainer_v3_contrastive_smoke.py` | End-to-end one-epoch smoke with contrastive on | Create |

---

## Phase 0 — Diagnostic & baseline fix

### Task 1: Fix the DeepFRI baseline number provenance

**Files:**
- Read: `results/deepfri_baseline.json`
- Modify: `docs/RESULTS_DATA.md`

- [ ] **Step 1: Inspect the recorded baseline source**

Run: `python -c "import json; print(json.dumps(json.load(open('results/deepfri_baseline.json')), indent=2))"`
Expected: prints the DeepFRI per-branch/per-bin Fmax/AUPRC currently used.

- [ ] **Step 2: Cross-check MF LT_95 against the DeepFRI paper**

The DeepFRI paper (Gligorijević et al., *Nat Commun* 2021) reports test-set MF Fmax ≈ **0.625–0.631**, not 0.759. The 0.759 in [RESULTS_DATA.md §1/§4](../../RESULTS_DATA.md) is in the HEAL/TAWFN range → likely a wrong row.

Decide one of:
- (a) the JSON value really is 0.759 and traces to HEAL Table S3.2's *re-evaluation* of DeepFRI → keep but relabel the source explicitly, OR
- (b) it is a mistake → replace with the paper value and note provenance.

- [ ] **Step 3: Edit RESULTS_DATA.md with explicit provenance**

In `docs/RESULTS_DATA.md` §4, add a provenance line directly under the "DeepFRI-GCN baseline" header, e.g.:

```markdown
> **Provenance:** DeepFRI numbers taken from <SOURCE: paper Table N / HEAL Table S3.2>.
> NOTE: original DeepFRI paper reports MF Fmax ≈ 0.625–0.631; if a higher value appears
> here it is a re-evaluation under the PDBch LT_* split protocol — state which in the thesis.
```

Fill `<SOURCE: ...>` from Step 1–2 findings. If (b), also correct the numeric rows in §1 and §4.

- [ ] **Step 4: Commit**

```bash
git add docs/RESULTS_DATA.md
git commit -m "docs(results): annotate/fix DeepFRI baseline provenance (MF LT_95)"
```

---

### Task 2: Add `ablate` + `return_z` hooks to AMPRModelV3

**Files:**
- Modify: `ampr/models/ampr.py:226-264` (`AMPRModelV3.forward` + `__init__`)
- Test: `tests/test_ampr_v3_return_z.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ampr_v3_return_z.py
import torch
from ampr.models.ampr import AMPRModelV3
from ampr.data.dataset import collate_variable_length


def _tiny_batch():
    torch.manual_seed(0)
    items = []
    for L in (6, 5):
        items.append({
            'x_seq_residue': torch.rand(L, 16),
            'seq_len': L,
            'cmap': (torch.rand(L, L) * 20),
            'x_ppi': torch.rand(8),
            'ppi_mask': torch.tensor(True),
            'labels': torch.randint(0, 2, (2,)).float(),
        })
    return collate_variable_length(items)


def _model():
    return AMPRModelV3(n_terms=2, seq_dim=16, seq_n_heads=2, seq_n_layers=1,
                       gnn_node_dim=16, gnn_n_layers=1, ppi_dim=8,
                       d_hidden=16, fusion_n_heads=2, fusion_n_layers=1,
                       go_emb_dim=8, dropout=0.0, contrastive_proj_dim=4)


def test_return_z_shapes():
    model = _model().eval()
    batch = _tiny_batch()
    go_emb = torch.rand(2, 8)
    logits, z = model(batch, go_emb=go_emb, return_z=True)
    assert logits.shape == (2, 2)
    assert z.shape == (2, 16)  # (B, d_hidden)


def test_logits_only_backward_compat():
    model = _model().eval()
    out = model(_tiny_batch(), go_emb=torch.rand(2, 8))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 2)


def test_ablate_zeroes_branch_changes_logits():
    model = _model().eval()
    batch = _tiny_batch()
    go_emb = torch.rand(2, 8)
    full = model(batch, go_emb=go_emb)
    ablated = model(batch, go_emb=go_emb, ablate=('gnn',))
    assert not torch.allclose(full, ablated)


def test_project_contrastive_shape():
    model = _model().eval()
    _, z = model(_tiny_batch(), go_emb=torch.rand(2, 8), return_z=True)
    feats = model.project_contrastive(z)
    assert feats.shape == (2, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ampr_v3_return_z.py -v`
Expected: FAIL — `forward()` got unexpected keyword `return_z` / `__init__` has no `contrastive_proj_dim`.

- [ ] **Step 3: Add the constructor param + projection head**

In `ampr/models/ampr.py`, modify `AMPRModelV3.__init__` signature (currently ends `cmap_threshold: float = 10.0, dropout: float = 0.1`) to add `contrastive_proj_dim: int = 0`:

```python
    def __init__(self, n_terms: int,
                 seq_dim: int = 1280, seq_n_heads: int = 8, seq_n_layers: int = 2,
                 gnn_node_dim: int = 256, gnn_n_layers: int = 3,
                 ppi_dim: int = 256,
                 d_hidden: int = 512, fusion_n_heads: int = 8, fusion_n_layers: int = 2,
                 classifier: str = 'both', go_emb_dim: int = 768,
                 cmap_threshold: float = 10.0, dropout: float = 0.1,
                 contrastive_proj_dim: int = 0):
```

Then, after the heads block (just before the `logger.info(...)` call near line 222), add:

```python
        # Contrastive projection head (Module A) — built only when needed.
        self.contrastive_proj = None
        if contrastive_proj_dim and contrastive_proj_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_hidden, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, contrastive_proj_dim),
            )

    def project_contrastive(self, z: torch.Tensor) -> torch.Tensor:
        assert self.contrastive_proj is not None, "contrastive_proj_dim must be > 0"
        return self.contrastive_proj(z)
```

- [ ] **Step 4: Rewrite the head section of `forward` to support `return_z` and `ablate`**

Replace the body of `AMPRModelV3.forward` from the signature through the head block (currently lines 226–264) with:

```python
    def forward(self, batch: dict, go_emb=None, return_z: bool = False,
                ablate: tuple = ()) -> torch.Tensor:
        x_res = batch['x_seq_residue']
        seq_mask = batch['seq_mask']
        cmap = batch['cmap']
        x_ppi = batch['x_ppi']
        ppi_mask = batch['ppi_mask']

        # Seq path
        seq_enc = self.seq_encoder(x_res, seq_mask)
        h_seq = self.seq_pool(seq_enc, seq_mask)
        h_seq = self.seq_proj(h_seq)

        # GNN path
        node_init = self.gnn_node_init(x_res)
        gnn_res = self.gnn(node_init, cmap, seq_mask)
        h_gnn = self.gnn_proj(gnn_res)

        # PPI path
        h_ppi = self.ppi_proj(x_ppi)
        h_ppi = h_ppi * ppi_mask.float().unsqueeze(-1)

        # Diagnostic ablation (Phase 0): zero a branch before fusion.
        if 'seq' in ablate:
            h_seq = torch.zeros_like(h_seq)
        if 'gnn' in ablate:
            h_gnn = torch.zeros_like(h_gnn)
        if 'ppi' in ablate:
            h_ppi = torch.zeros_like(h_ppi)

        # Fusion
        z = self.fusion(h_seq, h_gnn, h_ppi, ppi_mask)

        # Head(s)
        if self.classifier_type == 'label_attn':
            assert go_emb is not None, "label_attn head requires go_emb"
            logits = self.label_head(z, go_emb)
        elif self.classifier_type == 'linear':
            logits = self.linear_head(z)
        elif self.classifier_type == 'biobert':
            assert go_emb is not None
            logits = torch.matmul(self.go_emb_proj(z), go_emb.t())
        else:  # both
            lin = self.linear_head(z)
            if go_emb is None:
                logits = lin
            else:
                bio = torch.matmul(self.go_emb_proj(z), go_emb.t())
                logits = 0.5 * lin + 0.5 * bio

        if return_z:
            return logits, z
        return logits
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_ampr_v3_return_z.py tests/test_ampr_v3_forward.py tests/test_trainer_v3_smoke.py -v`
Expected: PASS (new tests pass; existing v3 forward/smoke tests still pass — backward compatible).

- [ ] **Step 6: Commit**

```bash
git add ampr/models/ampr.py tests/test_ampr_v3_return_z.py
git commit -m "feat(v3): add return_z + ablate hooks and contrastive projection head to AMPRModelV3"
```

---

### Task 3: Modality-ablation diagnostic script (eval-only, no retrain)

**Files:**
- Create: `scripts/diagnose_modality.py`

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Smoke-run on CPU dry path is not applicable (needs real checkpoint); verify it imports**

Run: `python -c "import ast; ast.parse(open('scripts/diagnose_modality.py').read()); print('OK')"`
Expected: prints `OK` (syntax valid). Real run happens on the server against a trained checkpoint.

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnose_modality.py
git commit -m "feat(diag): eval-only modality ablation script for v3 checkpoints"
```

---

## Module A — Multi-label Supervised Contrastive

### Task 4: `MultiLabelSupConLoss`

**Files:**
- Create: `ampr/training/contrastive.py`
- Test: `tests/test_contrastive.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contrastive.py
import torch
from ampr.training.contrastive import MultiLabelSupConLoss


def test_supcon_shape_and_grad():
    torch.manual_seed(0)
    loss_fn = MultiLabelSupConLoss(temp=0.1)
    feats = torch.randn(6, 16, requires_grad=True)
    labels = torch.randint(0, 2, (6, 4)).float()
    # ensure at least one shared-label pair exists
    labels[0] = labels[1]
    l = loss_fn(feats, labels)
    assert l.dim() == 0
    l.backward()
    assert feats.grad is not None


def test_supcon_pulls_shared_labels():
    """Features aligned with shared-label structure give lower loss."""
    loss_fn = MultiLabelSupConLoss(temp=0.1)
    labels = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0],
                           [0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float)
    aligned = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    misaligned = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    assert loss_fn(aligned, labels).item() < loss_fn(misaligned, labels).item()


def test_supcon_no_positive_returns_zero():
    """Disjoint single-label samples → no Jaccard positives → zero loss."""
    loss_fn = MultiLabelSupConLoss(temp=0.1)
    labels = torch.eye(3)
    feats = torch.randn(3, 8)
    assert loss_fn(feats, labels).item() == 0.0


def test_supcon_batch_lt_2_is_zero():
    loss_fn = MultiLabelSupConLoss()
    assert loss_fn(torch.randn(1, 8), torch.ones(1, 4)).item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contrastive.py -v`
Expected: FAIL — `No module named 'ampr.training.contrastive'`.

- [ ] **Step 3: Write the implementation**

```python
# ampr/training/contrastive.py
"""Multi-label supervised contrastive loss (Module A)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelSupConLoss(nn.Module):
    """Jaccard-weighted supervised contrastive loss for multi-label targets.

    Generalizes SupCon (Khosla et al., NeurIPS 2020) to soft multi-label
    positives: two proteins are positives in proportion to the Jaccard overlap
    of their GO-term label sets. Pulls together functionally similar proteins
    in representation space to improve low-identity generalization (cf. HEAL,
    Gu et al., Bioinformatics btad410, 2023).
    """

    def __init__(self, temp: float = 0.1, jaccard_thr: float = 0.0,
                 eps: float = 1e-8):
        super().__init__()
        self.temp = temp
        self.jaccard_thr = jaccard_thr
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, d) projected representations (un-normalized OK).
            labels:   (B, C) binary multi-label targets.
        Returns:
            scalar loss (0.0 if no Jaccard-positive pairs in the batch).
        """
        B = features.size(0)
        if B < 2:
            return features.sum() * 0.0

        z = F.normalize(features, dim=1)
        sim = (z @ z.t()) / self.temp
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # stability

        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)

        lab = labels.float()
        inter = lab @ lab.t()
        card = lab.sum(dim=1, keepdim=True)
        union = card + card.t() - inter
        jacc = inter / (union + self.eps)
        jacc = jacc.masked_fill(self_mask, 0.0)
        if self.jaccard_thr > 0:
            jacc = torch.where(jacc >= self.jaccard_thr, jacc,
                               torch.zeros_like(jacc))

        w_sum = jacc.sum(dim=1)
        valid = w_sum > self.eps
        if valid.sum() == 0:
            return features.sum() * 0.0

        per_row = -(jacc * log_prob).sum(dim=1) / (w_sum + self.eps)
        return per_row[valid].mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_contrastive.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ampr/training/contrastive.py tests/test_contrastive.py
git commit -m "feat(loss): multi-label Jaccard-weighted SupCon loss (Module A)"
```

---

### Task 5: Add optional contrastive term to `train_one_epoch_v3`

**Files:**
- Modify: `ampr/training/trainer.py:365-405` (`train_one_epoch_v3`)

- [ ] **Step 1: Replace `train_one_epoch_v3` with the contrastive-aware version**

Replace the existing `train_one_epoch_v3` (lines 365–405) with:

```python
def train_one_epoch_v3(model, loader, loss_fn, optimizer, go_emb, device='cuda',
                       grad_clip=0.0, contrastive_loss_fn=None,
                       contrastive_weight=0.0):
    """One training epoch over an AMPRDatasetV3 DataLoader.

    If contrastive_loss_fn is given and contrastive_weight > 0, adds
    contrastive_weight * L_CL on the projected fused representation z.

    Returns:
        avg_loss: float — total loss averaged over proteins in epoch.
    """
    model.train()
    model.to(device)
    go_emb = go_emb.to(device)
    use_cl = contrastive_loss_fn is not None and contrastive_weight > 0
    total = 0.0
    n = 0
    grad_norm_first = None
    for batch in loader:
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

        if use_cl:
            logits, z = model(batch_dev, go_emb=go_emb, return_z=True)
            feats = model.project_contrastive(z)
            cl = contrastive_loss_fn(feats, batch_dev['labels'])
        else:
            logits = model(batch_dev, go_emb=go_emb)
            cl = torch.zeros((), device=logits.device)

        loss, parts = loss_fn(logits, batch_dev['labels'])
        loss = loss + contrastive_weight * cl

        optimizer.zero_grad()
        loss.backward()
        if grad_norm_first is None:
            gn = sum(p.grad.detach().norm() ** 2 for p in model.parameters()
                     if p.grad is not None) ** 0.5
            grad_norm_first = float(gn)
            print(f"[DIAG] grad_norm(first batch)={grad_norm_first:.4e} "
                  f"cls={parts['cls']:.4f} dag={parts['dag']:.4f} "
                  f"cl={float(cl):.4f}")
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = batch_dev['labels'].size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)
```

- [ ] **Step 2: Run existing v3 smoke test (backward compat — no contrastive)**

Run: `python -m pytest tests/test_trainer_v3_smoke.py -v`
Expected: PASS (default args keep old behavior).

- [ ] **Step 3: Commit**

```bash
git add ampr/training/trainer.py
git commit -m "feat(train): optional contrastive term in train_one_epoch_v3"
```

---

### Task 6: Wire `training.contrastive` into `_run_v3`

**Files:**
- Modify: `main.py:126-201` (`_run_v3`: model construction, loss/optimizer setup, train call)

- [ ] **Step 1: Pass `contrastive_proj_dim` when building the model**

In `main.py` `_run_v3`, locate the `model = AMPRModelV3(...)` call (lines 126–141) and add a `contrastive_proj_dim` argument computed from config. Immediately *before* the `model = AMPRModelV3(` line, add:

```python
    contr_cfg = train_cfg.get('contrastive', {}) or {}
    contr_enabled = bool(contr_cfg.get('enabled', False))
    contr_proj_dim = int(contr_cfg.get('proj_dim', 128)) if contr_enabled else 0
```

Then add `contrastive_proj_dim=contr_proj_dim,` as the last argument of the `AMPRModelV3(...)` constructor call (after `dropout=seq_cfg.get('dropout', 0.1),`).

- [ ] **Step 2: Build the contrastive loss after the optimizer block**

In `_run_v3`, immediately after `grad_clip = float(train_cfg.get('grad_clip', 0.0))` (line 178), add:

```python
    contrastive_loss_fn = None
    contrastive_weight = 0.0
    if contr_enabled:
        from ampr.training.contrastive import MultiLabelSupConLoss
        contrastive_loss_fn = MultiLabelSupConLoss(
            temp=float(contr_cfg.get('temp', 0.1)),
            jaccard_thr=float(contr_cfg.get('jaccard_thr', 0.0)))
        contrastive_weight = float(contr_cfg.get('weight', 0.5))
        log.info(f"[V3] Contrastive (Module A): SupCon "
                 f"temp={contr_cfg.get('temp', 0.1)} "
                 f"jaccard_thr={contr_cfg.get('jaccard_thr', 0.0)} "
                 f"weight={contrastive_weight} proj_dim={contr_proj_dim}")
```

- [ ] **Step 3: Pass the contrastive args into the training call**

In `_run_v3`, replace the `train_one_epoch_v3(...)` call (lines 199–201) with:

```python
        train_loss = train_one_epoch_v3(model, ld_train, loss_fn, optimizer,
                                        go_emb=go_emb, device=device,
                                        grad_clip=grad_clip,
                                        contrastive_loss_fn=contrastive_loss_fn,
                                        contrastive_weight=contrastive_weight)
```

- [ ] **Step 4: Verify main.py imports/parses**

Run: `python -c "import ast; ast.parse(open('main.py').read()); print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(v3): wire training.contrastive config block into _run_v3"
```

---

### Task 7: End-to-end contrastive smoke test

**Files:**
- Create: `tests/test_trainer_v3_contrastive_smoke.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_trainer_v3_contrastive_smoke.py
import json
import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from ampr.training.trainer import train_one_epoch_v3
from ampr.training.contrastive import MultiLabelSupConLoss
from ampr.models.ampr import AMPRModelV3
from ampr.training.loss import AMPRLoss
from ampr.data.dataset import AMPRDatasetV3, collate_variable_length


def test_contrastive_one_epoch_drops_loss(tmp_path):
    torch.manual_seed(0)
    order = ['P1', 'P2', 'P3', 'P4']
    Path(tmp_path / 'order.json').write_text(json.dumps(order))
    Path(tmp_path / 'splits.json').write_text(json.dumps({'train': order}))
    # P1/P2 share label 0; P3/P4 share label 1 → Jaccard positives exist
    np.save(tmp_path / 'labels.npy',
            np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32))
    np.save(tmp_path / 'dag.npy', np.zeros((2, 2), dtype=np.float32))
    np.save(tmp_path / 'go.npy', np.random.rand(2, 8).astype(np.float32))
    with h5py.File(tmp_path / 'esm2.h5', 'w') as f:
        for p, L in zip(order, (6, 5, 7, 4)):
            f.create_dataset(p, data=np.random.rand(L, 16).astype(np.float32))
    np.save(tmp_path / 'ppi.npy', np.random.rand(4, 8).astype(np.float32))
    np.save(tmp_path / 'mask.npy', np.array([True, True, True, True]))
    with h5py.File(tmp_path / 'cmap.h5', 'w') as f:
        for p, L in zip(order, (6, 5, 7, 4)):
            f.create_dataset(p, data=(np.random.rand(L, L) * 20).astype(np.float32))

    ds = AMPRDatasetV3(
        esm2_h5=str(tmp_path / 'esm2.h5'), ppi_emb=str(tmp_path / 'ppi.npy'),
        ppi_mask=str(tmp_path / 'mask.npy'), cmap_h5=str(tmp_path / 'cmap.h5'),
        labels=str(tmp_path / 'labels.npy'), dag_matrix=str(tmp_path / 'dag.npy'),
        go_emb=str(tmp_path / 'go.npy'), splits=str(tmp_path / 'splits.json'),
        protein_order=str(tmp_path / 'order.json'), branch='MF', split='train',
        max_len=20)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_variable_length)
    model = AMPRModelV3(n_terms=2, seq_dim=16, seq_n_heads=2, seq_n_layers=1,
                        gnn_node_dim=16, gnn_n_layers=1, ppi_dim=8, d_hidden=16,
                        fusion_n_heads=2, fusion_n_layers=1, go_emb_dim=8,
                        dropout=0.0, contrastive_proj_dim=4)
    loss_fn = AMPRLoss(ds.dag_matrix, lambda_dag=0.0, loss_type='asl')
    cl_fn = MultiLabelSupConLoss(temp=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = [train_one_epoch_v3(model, loader, loss_fn, opt, go_emb=ds.go_emb,
                                 device='cpu', contrastive_loss_fn=cl_fn,
                                 contrastive_weight=0.5) for _ in range(5)]
    assert losses[-1] < losses[0]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_trainer_v3_contrastive_smoke.py -v`
Expected: PASS — loss decreases over 5 epochs with contrastive active.

- [ ] **Step 3: Run full test suite for regressions**

Run: `python -m pytest tests/test_contrastive.py tests/test_ampr_v3_return_z.py tests/test_trainer_v3_smoke.py tests/test_trainer_v3_contrastive_smoke.py tests/test_asl_loss.py -v`
Expected: PASS (all).

- [ ] **Step 4: Commit**

```bash
git add tests/test_trainer_v3_contrastive_smoke.py
git commit -m "test(v3): end-to-end contrastive smoke (loss decreases)"
```

---

### Task 8: Create v4 SupCon configs (MF, CC, BP)

**Files:**
- Create: `configs/mf_v4_supcon.yaml`, `configs/cc_v4_supcon.yaml`, `configs/bp_v4_supcon.yaml`

- [ ] **Step 1: Create the MF config**

Copy `configs/mf_v3_esm3b.yaml` to `configs/mf_v4_supcon.yaml`, then under `training:` add a `contrastive` block and redirect outputs. The full file:

```yaml
branch: MF
n_terms: 489
data:
  protein_order: data/pdbch/protein_order.json
  splits:        data/pdbch/splits.json
  labels:        data/pdbch/labels_mf.npy
  dag_matrix:    data/pdbch/dag_matrix_mf.npy
  esm2_h5:       data/embeddings/esm2_3b_residue.h5
  ppi_emb:       data/embeddings/ppi_deepgo.npy
  ppi_mask:      data/embeddings/ppi_deepgo_mask.npy
  cmap_h5:       data/contact_maps/cmap_all.h5
  go_emb:        data/embeddings/go_emb_mf_v2.npy
  diamond_tsv:   data/diamond/diamond_results_mf.tsv
model:
  version: v3
  structure_modality: gnn
  seq: {d_model: 2560, n_transformer_layers: 2, n_heads: 8, dropout: 0.2}
  gnn: {node_dim: 256, n_layers: 3, cmap_threshold: 10.0}
  ppi: {in_dim: 256, hidden: 512}
  fusion: {d_model: 512, n_layers: 2, n_heads: 8}
  classifier: both
  d_hidden: 512
training:
  epochs: 50
  batch_size: 64
  lr: 1.0e-3
  lr_scheduler: plateau
  lr_factor: 0.5
  lr_patience: 5
  lr_min: 1.0e-5
  weight_decay: 1.0e-2
  loss_type: asl
  asl_gamma_neg: 4
  asl_gamma_pos: 0
  asl_clip: 0.05
  lambda_dag: 0.5
  contrastive:
    enabled: true
    weight: 0.5
    temp: 0.1
    jaccard_thr: 0.0
    proj_dim: 128
  seed: 42
  device: auto
  max_seq_len: 1000
  num_workers: 16
inference:
  use_dag_propagation: true
  use_diamond_ensemble: true
  diamond_alpha: 0.6
  threshold_path: checkpoints/mf_v4_supcon/threshold.json
output:
  checkpoint_dir: checkpoints/mf_v4_supcon/
  log_file:       logs/mf_v4_supcon_train.log
  results_file:   results/mf_v4_supcon_predictions.tsv
seed: 42
```

- [ ] **Step 2: Create the CC config**

Copy the MF config to `configs/cc_v4_supcon.yaml` and change: `branch: CC`, `n_terms: 320`, `labels: data/pdbch/labels_cc.npy`, `dag_matrix: data/pdbch/dag_matrix_cc.npy`, `go_emb: data/embeddings/go_emb_cc_v2.npy`, `diamond_tsv: data/diamond/diamond_results_cc.tsv`, `weight_decay: 0.0` (CC baseline winner, per [RESULTS_DATA §5](../../RESULTS_DATA.md)), `seq.dropout: 0.1`, and all `mf_v4_supcon` → `cc_v4_supcon` in `inference`/`output`.

- [ ] **Step 3: Create the BP config**

Copy the MF config to `configs/bp_v4_supcon.yaml` and change: `branch: BP`, `n_terms: 1943`, `labels: data/pdbch/labels_bp.npy`, `dag_matrix: data/pdbch/dag_matrix_bp.npy`, `go_emb: data/embeddings/go_emb_bp_v2.npy`, `diamond_tsv: data/diamond/diamond_results_bp.tsv`, `weight_decay: 0.0` (BP baseline winner), `seq.dropout: 0.1`, and all `mf_v4_supcon` → `bp_v4_supcon` in `inference`/`output`.

> **Note:** confirm the exact `go_emb_{cc,bp}_v2.npy` filenames against the existing `configs/{cc,bp}_v3_esm3b.yaml` before launching (copy whatever those use).

- [ ] **Step 4: Validate all three configs load**

Run: `python -c "import yaml; [yaml.safe_load(open(f'configs/{b}_v4_supcon.yaml')) for b in ('mf','cc','bp')]; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add configs/mf_v4_supcon.yaml configs/cc_v4_supcon.yaml configs/bp_v4_supcon.yaml
git commit -m "config: v4 SupCon configs for MF/CC/BP (Module A)"
```

---

### Task 9: Launch sweep on H200 (GPU assignment + collapse guard)

**Files:**
- None (operational). Run on the server.

- [ ] **Step 1: Dry-run each config (1 epoch, ~50 proteins) before the real launch**

Run (on server, 1 GPU):
```bash
python main.py --config configs/mf_v4_supcon.yaml --dry_run
```
Expected: `[V3] Contrastive (Module A): SupCon ...` logged; one epoch completes; `[DIAG] ... cl=<nonzero>` printed; no crash.

- [ ] **Step 2: Check VRAM headroom before co-locating jobs**

Run: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`
Rule: only place a second job on a GPU if `memory.total - memory.used` comfortably exceeds the dry-run footprint. Never co-locate two heavy jobs (BP). Co-locating two light jobs (e.g. CC) is OK only if the job is IO-bound (small per-step GPU time).

- [ ] **Step 3: Launch the λ_cl sweep, one run per GPU**

For each branch, sweep `weight ∈ {0.1, 0.5, 1.0}` by overriding the `contrastive.weight` field (make 3 copies per branch or edit the YAML before each launch). Assign one run per GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/mf_v4_supcon_w01.yaml &
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/mf_v4_supcon_w05.yaml &
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/mf_v4_supcon_w10.yaml &
CUDA_VISIBLE_DEVICES=3 python main.py --config configs/cc_v4_supcon_w05.yaml &
CUDA_VISIBLE_DEVICES=4 python main.py --config configs/bp_v4_supcon_w05.yaml &
# ... fill remaining GPUs; queue extra runs (MF/CC finish first)
```
Use tmux per the H200 runbook. MF/CC finish fastest; BP is the long pole.

- [ ] **Step 4: Watch the collapse guard each run**

In each `logs/*_v4_supcon_train.log`, confirm per epoch `[DIAG] ... cross_protein_std=` stays **> 1e-4** and `val_Fmax_dag` rises. If `cross_protein_std → 0` (collapse, see [DESIGN_RATIONALE §6](../../DESIGN_RATIONALE.md)), stop that run and relaunch with `loss_type: bce` + `pos_weight_cap: 50` in its config (keep contrastive on).

- [ ] **Step 5: Evaluate best λ_cl per branch on all bins + DIAMOND**

For the best val_Fmax_dag checkpoint per branch:
```bash
for s in test_LT_30 test_LT_40 test_LT_50 test_LT_70 test_LT_95; do
  python main.py --config configs/mf_v4_supcon_w05.yaml --eval-only \
    --checkpoint checkpoints/mf_v4_supcon/best.pt --test-split $s
done
```
Compare LT_95 Fmax (+DIAMOND) against the v3 baseline in [RESULTS_DATA.md §3](../../RESULTS_DATA.md): MF 0.614 / BP 0.507 / CC 0.538.

- [ ] **Step 6: Record results**

Append a results section to `docs/RESULTS_DATA.md` (or a new `docs/RESULTS_V4.md`) with the per-branch best λ_cl, val Fmax, val→test gap, and LT_95 (+DIAMOND) vs v3. Commit.

```bash
git add docs/RESULTS_DATA.md docs/RESULTS_V4.md
git commit -m "docs(results): Module A (SupCon) sweep results vs v3 baseline"
```

---

## Self-Review

**Spec coverage (spec §2 Phase 0, §3 Module A):**
- §2 D0.1 modality ablation → Task 2 (hook) + Task 3 (script). ✓
- §2 D0.2 baseline fix → Task 1. ✓
- §2 D0.3 gate logging → existing `[DIAG]` in `_run_v3`/`train_one_epoch_v3` already logs alphas/probs std; ablation script + DIAG cover the intent. ✓ (no separate task needed)
- §3 Module A SupCon loss → Task 4. ✓
- §3 projection head + `z` access → Task 2 (head) + Task 2 forward `return_z`. ✓
- §3 total loss `L_ASL + λ_dag·L_DAG + λ_cl·L_CL` → Task 5. ✓
- §3 config block + λ_cl sweep → Task 6, 8, 9. ✓
- §3 ASL×CL collapse mitigation → Task 9 Step 4. ✓
- §8 testing (unit shape/grad, integration no-collapse, regression baseline) → Tasks 4, 7 + regression in 5/7. ✓
- Module B/C → out of scope (separate plans), stated in header. ✓

**Type consistency:** `MultiLabelSupConLoss(temp, jaccard_thr)`, `model(batch, go_emb, return_z, ablate)`, `model.project_contrastive(z)`, `train_one_epoch_v3(..., contrastive_loss_fn, contrastive_weight)`, config keys `training.contrastive.{enabled,weight,temp,jaccard_thr,proj_dim}` — used identically across Tasks 2/4/5/6/8. ✓

**Placeholder scan:** config Task 8 has one explicit `<confirm filename>` note (go_emb_{cc,bp}) — deliberate verification step, not a code placeholder. No TBD/TODO in code. ✓

---

## Next plans (after Module A results)

- **Module C — Term-conditioned pooling (MF+CC):** replace seq attention-pool with GO-term cross-attention over residues; emit residue→term attention maps for the interpretability figure. New plan once Module A lands.
- **Module B — Domain-guided structure (DPFunc-style):** SIFTS→UniProt→InterPro precompute + domain-gated attention. Deferred (precompute-heavy); new plan when GPU returns.
