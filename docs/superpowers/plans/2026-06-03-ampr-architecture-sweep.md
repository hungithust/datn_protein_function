# AMPR Architecture Upgrade + Capacity Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a label-attention classification head to `AMPRModelV3`, then run a measured capacity/encoder sweep across the 8 H200 GPUs to pick the best configuration by validation Fmax — instead of guessing how big the model should be.

**Architecture:** A new `LabelAttentionHead` (multi-head bilinear scoring of the fused protein vector against the GO label-embedding matrix) is added as `classifier='label_attn'` alongside the existing heads, fully shape/grad unit-tested. A pure grid-expander generates one config per sweep cell; a tmux launcher runs all cells in parallel (one GPU each); a collector reads each checkpoint's `fmax_dag` and reports the winner. Scaling knobs (`d_hidden`, depth) are already config fields, so "phình to" is just config — bounded by the sweep's measured selection (spec §6).

**Tech Stack:** PyTorch, PyYAML, tmux. Local tests via Anaconda pytest.

**Spec:** [docs/superpowers/specs/2026-06-01-ampr-h200-migration-design.md](../specs/2026-06-01-ampr-h200-migration-design.md) (Phase 4)

**Prereq:** Plan 2 done — `esm2_3b_residue.h5`, `go_emb_*_v2.npy`, `go_text_*.npy`, and `*_v3_esm3b.yaml` exist
and pass `verify_inputs.py`.

**Local testing:** `python -m pytest tests/<file> -v` from repo root. Never bare `pytest`.

---

### Task 1: `LabelAttentionHead` module (TDD)

**Files:**
- Create: `ampr/models/label_attention.py`
- Test: `tests/test_label_attention.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_label_attention.py
import torch
from ampr.models.label_attention import LabelAttentionHead


def test_label_attn_shape_and_grad():
    torch.manual_seed(0)
    head = LabelAttentionHead(d_hidden=64, go_emb_dim=24, n_terms=12,
                              n_heads=4, dropout=0.0)
    z = torch.randn(3, 64, requires_grad=True)
    go_emb = torch.randn(12, 24, requires_grad=True)
    logits = head(z, go_emb)
    assert logits.shape == (3, 12)
    logits.sum().backward()
    assert z.grad is not None and go_emb.grad is not None
    assert head.bias.grad is not None


def test_label_attn_requires_divisible_heads():
    import pytest
    with pytest.raises(AssertionError):
        LabelAttentionHead(d_hidden=10, go_emb_dim=8, n_terms=4, n_heads=3)
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_label_attention.py -v`
Expected: FAIL — module `ampr.models.label_attention` not found.

- [ ] **Step 3: Implement the head**

```python
# ampr/models/label_attention.py
"""Label-attention classification head.

Scores the fused protein vector z against the GO label-embedding matrix with
multi-head bilinear attention: logit[b,c] = sum_h <Wq(z)_h, Wk(go_c)_h> / sqrt(d_head) + bias_c.
A richer alternative to the flat `z · proj(go)^T` dot-product (TALE/ATGO-style).
"""
import torch
import torch.nn as nn


class LabelAttentionHead(nn.Module):
    def __init__(self, d_hidden: int, go_emb_dim: int, n_terms: int,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.scale = self.d_head ** -0.5
        self.q = nn.Linear(d_hidden, d_hidden)
        self.k = nn.Linear(go_emb_dim, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(n_terms))

    def forward(self, z: torch.Tensor, go_emb: torch.Tensor) -> torch.Tensor:
        # z: (B, D); go_emb: (C, go_emb_dim)
        B = z.shape[0]
        C = go_emb.shape[0]
        q = self.q(self.drop(z)).view(B, self.n_heads, self.d_head)
        k = self.k(go_emb).view(C, self.n_heads, self.d_head)
        logits = torch.einsum('bhd,chd->bc', q, k) * self.scale
        return logits + self.bias
```

- [ ] **Step 4: Run it to confirm pass**

Run: `python -m pytest tests/test_label_attention.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add ampr/models/label_attention.py tests/test_label_attention.py
git commit -m "feat(model): LabelAttentionHead multi-head bilinear GO scoring (TDD)"
```

---

### Task 2: Wire `classifier='label_attn'` into `AMPRModelV3` (TDD)

**Files:**
- Modify: `ampr/models/ampr.py:161-219,221-256`
- Test: `tests/test_ampr_v3_label_attn.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ampr_v3_label_attn.py
import torch
from ampr.models.ampr import AMPRModelV3


def test_v3_label_attn_forward():
    torch.manual_seed(0)
    m = AMPRModelV3(
        n_terms=12, seq_dim=64, seq_n_heads=4, seq_n_layers=1,
        gnn_node_dim=32, gnn_n_layers=2, ppi_dim=16,
        d_hidden=64, fusion_n_heads=4, fusion_n_layers=1,
        classifier='label_attn', go_emb_dim=24, dropout=0.0,
    )
    B, L = 2, 8
    batch = {
        'x_seq_residue': torch.randn(B, L, 64, requires_grad=True),
        'seq_mask': torch.ones(B, L, dtype=torch.bool),
        'cmap': torch.rand(B, L, L) * 20,
        'x_ppi': torch.randn(B, 16),
        'ppi_mask': torch.tensor([True, False]),
    }
    go_emb = torch.randn(12, 24)
    logits = m(batch, go_emb=go_emb)
    assert logits.shape == (B, 12)
    logits.sum().backward()
    assert batch['x_seq_residue'].grad is not None
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_ampr_v3_label_attn.py -v`
Expected: FAIL — `label_attn` head not built / forward has no branch (AttributeError or wrong shape).

- [ ] **Step 3: Add the head in `__init__`**

In `AMPRModelV3.__init__`, after the existing head block (the `if classifier in ('biobert', 'both')`
block ending at `self.go_emb_proj = nn.Linear(d_hidden, go_emb_dim)`), add:

```python
        if classifier == 'label_attn':
            from ampr.models.label_attention import LabelAttentionHead
            self.label_head = LabelAttentionHead(
                d_hidden=d_hidden, go_emb_dim=go_emb_dim, n_terms=n_terms,
                n_heads=fusion_n_heads, dropout=dropout)
```

- [ ] **Step 4: Add the branch in `forward`**

In `AMPRModelV3.forward`, replace the head dispatch (the block starting `if self.classifier_type == 'linear':`)
with a version that handles `label_attn` first:

```python
        # Head(s)
        if self.classifier_type == 'label_attn':
            assert go_emb is not None, "label_attn head requires go_emb"
            return self.label_head(z, go_emb)
        if self.classifier_type == 'linear':
            return self.linear_head(z)
        if self.classifier_type == 'biobert':
            assert go_emb is not None
            return torch.matmul(self.go_emb_proj(z), go_emb.t())
        # both
        lin = self.linear_head(z)
        if go_emb is None:
            return lin
        bio = torch.matmul(self.go_emb_proj(z), go_emb.t())
        return 0.5 * lin + 0.5 * bio
```

- [ ] **Step 5: Run both forward tests**

Run: `python -m pytest tests/test_ampr_v3_label_attn.py tests/test_ampr_v3_forward.py -v`
Expected: PASS (2 passed) — the new head works and the existing `both` path is unchanged.

- [ ] **Step 6: Commit**

```bash
git add ampr/models/ampr.py tests/test_ampr_v3_label_attn.py
git commit -m "feat(model): wire classifier='label_attn' into AMPRModelV3 (TDD)"
```

---

### Task 3: Sweep grid expander (pure, TDD)

**Files:**
- Create: `scripts/gen_sweep_configs.py`
- Test: `tests/test_gen_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gen_sweep.py
from scripts.gen_sweep_configs import set_dotted, expand_grid


def test_set_dotted_nested():
    cfg = {'model': {'d_hidden': 512}}
    set_dotted(cfg, 'model.d_hidden', 1024)
    assert cfg['model']['d_hidden'] == 1024


def test_expand_grid_cartesian():
    base = {'model': {'classifier': 'both', 'd_hidden': 512},
            'data': {'go_emb': 'x'}}
    grid = {
        'model.classifier': [('both', 'both'), ('la', 'label_attn')],
        'model.d_hidden':   [('h512', 512), ('h1024', 1024)],
        'data.go_emb':      [('comb', 'go_emb_mf_v2.npy'), ('text', 'go_text_mf.npy')],
    }
    out = expand_grid(base, grid)
    assert len(out) == 8                      # 2*2*2
    names = [n for n, _ in out]
    assert 'both_h512_comb' in names
    assert 'la_h1024_text' in names
    # base is not mutated
    assert base['model']['d_hidden'] == 512
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_gen_sweep.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Generate one v3 config per sweep cell from a base config + grid.

Sweep axes (8 cells = 8 GPUs):
  classifier : {both, label_attn}
  d_hidden   : {512, 1024}
  go_emb     : {text+graph combined, text-only}

Usage:
  python scripts/gen_sweep_configs.py --base configs/mf_v3_esm3b.yaml \
    --branch mf --out_dir configs/sweep
"""
import argparse
import copy
import itertools
from pathlib import Path
import yaml


def set_dotted(cfg: dict, dotted: str, value):
    keys = dotted.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def expand_grid(base: dict, grid: dict):
    """Cartesian product over grid -> list of (name, config). base is not mutated."""
    keys = list(grid)
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = copy.deepcopy(base)
        tags = []
        for k, (tag, val) in zip(keys, combo):
            set_dotted(cfg, k, val)
            tags.append(tag)
        out.append(('_'.join(tags), cfg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--branch', required=True)
    ap.add_argument('--out_dir', default='configs/sweep')
    args = ap.parse_args()
    base = yaml.safe_load(open(args.base))
    b = args.branch
    grid = {
        'model.classifier': [('both', 'both'), ('la', 'label_attn')],
        'model.d_hidden':   [('h512', 512), ('h1024', 1024)],
        'data.go_emb':      [('comb', f'data/embeddings/go_emb_{b}_v2.npy'),
                             ('text', f'data/embeddings/go_text_{b}.npy')],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in expand_grid(base, grid):
        tag = f"{b}_{name}"
        cfg['output']['checkpoint_dir'] = f"checkpoints/sweep_{tag}/"
        cfg['output']['log_file'] = f"logs/sweep_{tag}.log"
        cfg['output']['results_file'] = f"results/sweep_{tag}.tsv"
        path = out_dir / f"{tag}.yaml"
        yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)
        print(f"[SWEEP] wrote {path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run it to confirm pass**

Run: `python -m pytest tests/test_gen_sweep.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Generate the MF sweep configs**

Run: `python scripts/gen_sweep_configs.py --base configs/mf_v3_esm3b.yaml --branch mf --out_dir configs/sweep`
Expected: 8 lines `[SWEEP] wrote configs/sweep/mf_*.yaml`.

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_sweep_configs.py tests/test_gen_sweep.py configs/sweep/
git commit -m "feat(sweep): grid expander for 8-cell capacity/encoder sweep (TDD)"
```

---

### Task 4: Parallel sweep launcher

**Files:**
- Create: `scripts/launch_sweep.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# scripts/launch_sweep.sh — run all sweep configs, one GPU each (0-7).
# Usage: bash scripts/launch_sweep.sh configs/sweep
set -euo pipefail
cd /raid/team/datn
SWEEP_DIR="${1:-configs/sweep}"
mapfile -t CFGS < <(ls "$SWEEP_DIR"/*.yaml | sort)
n=${#CFGS[@]}
echo "[SWEEP] $n configs over $(nvidia-smi -L | wc -l) GPUs"
gpu=0
for cfg in "${CFGS[@]}"; do
  base=$(basename "$cfg" .yaml)
  sess="sw_$base"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpu python main.py --config $cfg 2>&1 | tee logs/${base}.run.log"
  echo "[SWEEP] $base -> GPU $gpu"
  gpu=$(( (gpu + 1) % 8 ))
done
echo "[SWEEP] launched. monitor: watch -n1 nvidia-smi ; tmux ls"
```

Note: with 8 cells and 8 GPUs each cell gets a dedicated GPU. If a branch has >8 cells, cells wrap onto
shared GPUs (still correct, just serialized within a GPU).

- [ ] **Step 2: Pre-flight one config, then launch**

Run:
```bash
python scripts/verify_inputs.py --config configs/sweep/mf_both_h512_comb.yaml && \
bash scripts/launch_sweep.sh configs/sweep
```
Expected: `ALL PASS`, then 8 `[SWEEP] … -> GPU n` lines; `tmux ls` shows 8 `sw_mf_*` sessions.

- [ ] **Step 3: Confirm jobs are training**

Run: `sleep 90; grep -h "Epoch 1/" logs/sweep_mf_*.log | head`
Expected: each cell prints a `[V3] Epoch 1/50: … val_Fmax_dag=…` line; `nvidia-smi` shows all 8 GPUs busy.

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_sweep.sh
git commit -m "feat(sweep): tmux launcher mapping sweep cells across 8 GPUs"
```

---

### Task 5: Sweep result collector (pure, TDD)

**Files:**
- Create: `scripts/collect_sweep.py`
- Test: `tests/test_collect_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_sweep.py
from scripts.collect_sweep import best_fmax_from_log


def test_best_fmax_picks_max():
    log = (
        "[V3] Epoch 1/50: loss=0.4 val_Fmax_raw=0.30 val_Fmax_dag=0.31\n"
        "[V3] Epoch 2/50: loss=0.3 val_Fmax_raw=0.40 val_Fmax_dag=0.42\n"
        "[V3] Epoch 3/50: loss=0.2 val_Fmax_raw=0.39 val_Fmax_dag=0.41\n"
    )
    assert abs(best_fmax_from_log(log) - 0.42) < 1e-9


def test_best_fmax_no_match_returns_none():
    assert best_fmax_from_log("no epochs here") is None
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_collect_sweep.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Collect sweep results: best val_Fmax_dag per cell -> ranked table + winner.

Usage:
  python scripts/collect_sweep.py --logs_glob 'logs/sweep_mf_*.log'
"""
import argparse
import glob
import re

_FMAX = re.compile(r'val_Fmax_dag=([0-9.]+)')


def best_fmax_from_log(text: str):
    vals = [float(m.group(1)) for m in _FMAX.finditer(text)]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_glob', required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(args.logs_glob)):
        with open(path) as f:
            best = best_fmax_from_log(f.read())
        rows.append((path, best))
    rows.sort(key=lambda r: (r[1] is not None, r[1] or -1), reverse=True)
    print(f"{'config log':<48} best_val_Fmax_dag")
    for path, best in rows:
        print(f"{path:<48} {best if best is not None else 'n/a'}")
    if rows and rows[0][1] is not None:
        print(f"\n[WINNER] {rows[0][0]}  Fmax_dag={rows[0][1]:.4f}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run it to confirm pass**

Run: `python -m pytest tests/test_collect_sweep.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Collect after the sweep finishes**

Run: `python scripts/collect_sweep.py --logs_glob 'logs/sweep_mf_*.log'`
Expected: a ranked table and a `[WINNER]` line. Promote the winning config's knobs into the final
`configs/mf_v3_esm3b.yaml` (and replicate the chosen settings for bp/cc).

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_sweep.py tests/test_collect_sweep.py
git commit -m "feat(sweep): collect best val Fmax per cell + report winner (TDD)"
```

---

## Self-Review

**Spec coverage (Phase 4):**
- 4a label-attention head → Tasks 1, 2 ✓
- 4c scale-up knobs (`d_hidden`, depth) → exercised via sweep grid (config-only, no code) ✓
- 4e capacity/encoder sweep across 8 GPUs → Tasks 3, 4, 5 ✓
- 4b wider fusion → fusion depth is already a config field (`fusion.n_layers`); can be added as a sweep axis
  by extending the Task-3 grid. The 4-token ProstT5 variant remains **stretch (spec §7), not in this plan.**
- 4d regularization → `dropout` is a config field already plumbed through `AMPRModelV3`; weight-decay would
  be a one-line optimizer change tracked as a follow-up if the sweep shows overfitting.

**Placeholder scan:** No TBD/TODO. Edit sites in `ampr.py` are pinned to exact anchor lines and the
replacement blocks are shown in full.

**Type/name consistency:** `LabelAttentionHead(d_hidden, go_emb_dim, n_terms, n_heads, dropout)` identical
in module, test, and the `AMPRModelV3.__init__` call (uses `fusion_n_heads` for `n_heads`, consistent with
the forward test which sets `fusion_n_heads=4`, `d_hidden=64` → divisible). `set_dotted`/`expand_grid`,
`best_fmax_from_log` signatures match across tests and `main()`. Sweep file/dir names
(`configs/sweep/{branch}_{name}.yaml`, `logs/sweep_{tag}.log`) are consistent across Tasks 3, 4, 5.
Regex `val_Fmax_dag=([0-9.]+)` matches the exact log string emitted by `main.py` `_run_v3`
(`val_Fmax_dag={fmax_dag:.4f}`).
