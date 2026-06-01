# AMPR H200 Migration + Baseline Reproduce — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get AMPR v3 MF/BP/CC training running on the 8×H200 hackathon server from the existing Kaggle embeddings, with input-alignment verification, in one sitting.

**Architecture:** SSH + `tmux`, one branch per GPU. Pull precomputed embeddings via the Kaggle API directly onto `/raid` (no local hop), verify alignment, then launch three training jobs with H200-sized batches. This is **Track 1** of the spec — Tracks 2/3 (embedding + architecture upgrades) are separate plans.

**Tech Stack:** NGC `pytorch:24.10-py3` container, Kaggle CLI, tmux, PyTorch, h5py, PyYAML.

**Spec:** [docs/superpowers/specs/2026-06-01-ampr-h200-migration-design.md](../specs/2026-06-01-ampr-h200-migration-design.md) (Phases 0–2)

**Execution note:** Tasks run **on the H200 server** (Linux), not the Windows dev box. "Tests" here are
verification commands with expected output, since the work is operational (remote GPU), not local pytest.
Commit code/doc artifacts from the repo checkout on the server (or commit locally and `git pull` on server).

---

### Task 1: Server bootstrap script

**Files:**
- Create: `scripts/server_setup.sh`

- [ ] **Step 1: Write the bootstrap script**

```bash
#!/usr/bin/env bash
# scripts/server_setup.sh — one-time H200 environment bootstrap.
# Idempotent: safe to re-run. Run from inside the JupyterLab/NGC container.
set -euo pipefail

REPO_DIR=/raid/team/datn
REPO_URL=https://github.com/hungithust/datn_protein_function

echo "[SETUP] repo -> $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "[SETUP] python extras (into NGC image)"
pip install -q \
  transformers==4.41.2 obonet biopython h5py pyyaml tqdm sentence-transformers

echo "[SETUP] kaggle cli + credentials"
pip install -q kaggle
mkdir -p "$HOME/.kaggle"
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "[SETUP][WARN] place kaggle.json at $HOME/.kaggle/kaggle.json then re-run"
else
  chmod 600 "$HOME/.kaggle/kaggle.json"
fi

echo "[SETUP] GPU sanity"
python - <<'PY'
import torch
print("[SETUP] cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
assert torch.cuda.device_count() == 8, "expected 8 H200"
print("[SETUP] gpu0:", torch.cuda.get_device_name(0))
PY

echo "[SETUP] DGL-free check for v3 GNN path"
python - <<'PY'
import ast, pathlib
src = pathlib.Path("ampr/models/gnn_encoder.py").read_text()
assert "import dgl" not in src and "from dgl" not in src, \
    "gnn_encoder imports DGL — cu121 wheel will not match CUDA-13 image; refactor or install matching wheel"
print("[SETUP] gnn_encoder is DGL-free OK")
PY

echo "[SETUP] done"
```

- [ ] **Step 2: Run it on the server**

Run: `bash scripts/server_setup.sh`
Expected (final lines):
```
[SETUP] cuda: True gpus: 8
[SETUP] gpu0: NVIDIA H200
[SETUP] gnn_encoder is DGL-free OK
[SETUP] done
```
If the DGL check fails, stop and resolve (refactor encoder or install a CUDA-13-matching build) before training.

- [ ] **Step 3: Commit**

```bash
git add scripts/server_setup.sh
git commit -m "feat(h200): server bootstrap script (env + GPU + DGL-free check)"
```

---

### Task 2: Kaggle data pull script

**Files:**
- Create: `scripts/pull_kaggle_data.sh`

- [ ] **Step 1: Write the pull script**

Dataset slugs taken from `notebooks/kaggle_phase3_train_mf.ipynb` (owner `hungnguyenviet04`). Adjust
slugs if the live `kaggle datasets list -m` shows different names.

```bash
#!/usr/bin/env bash
# scripts/pull_kaggle_data.sh — download precomputed AMPR datasets to /raid.
set -euo pipefail
DATA=/raid/team/datn/data
mkdir -p "$DATA/embeddings" "$DATA/contact_maps" "$DATA/pdbch" "$DATA/_dl"
cd "$DATA/_dl"

pull () {  # $1 = dataset slug, $2 = subdir under _dl
  echo "[PULL] $1"
  kaggle datasets download -d "$1" -p "$2" --unzip
}

pull hungnguyenviet04/ampr-phase3-embeddings   emb1
pull hungnguyenviet04/ampr-phase3-embeddings-2 emb2
pull hungnguyenviet04/cmap-all                 cmap
pull hungnguyenviet04/ampr-pdbch-phase0        pdbch

echo "[PULL] place files at config-expected paths"
ln -sf "$DATA/_dl/emb1/esm2_residue.h5"        "$DATA/embeddings/esm2_residue.h5"
ln -sf "$DATA/_dl/emb2/ppi_deepgo.npy"         "$DATA/embeddings/ppi_deepgo.npy"
ln -sf "$DATA/_dl/emb2/ppi_deepgo_mask.npy"    "$DATA/embeddings/ppi_deepgo_mask.npy"
ln -sf "$DATA/_dl/cmap/cmap_all.h5"            "$DATA/contact_maps/cmap_all.h5"
for f in labels_mf labels_bp labels_cc dag_matrix_mf dag_matrix_bp dag_matrix_cc; do
  ln -sf "$DATA/_dl/pdbch/$f.npy"  "$DATA/pdbch/$f.npy"
done
ln -sf "$DATA/_dl/pdbch/splits.json"         "$DATA/pdbch/splits.json"
ln -sf "$DATA/_dl/pdbch/protein_order.json"  "$DATA/pdbch/protein_order.json"
for b in mf bp cc; do
  ln -sf "$DATA/_dl/pdbch/go_emb_$b.npy" "$DATA/embeddings/go_emb_$b.npy"
done
echo "[PULL] done — review any missing symlink targets above"
ls -lL "$DATA/embeddings" "$DATA/contact_maps" "$DATA/pdbch"
```

- [ ] **Step 2: Run it**

Run: `bash scripts/pull_kaggle_data.sh`
Expected: each `[PULL]` completes; final `ls -lL` shows real files (not broken symlinks) for
`esm2_residue.h5`, `cmap_all.h5`, `ppi_deepgo.npy`, `labels_mf.npy`, `splits.json`, `go_emb_mf.npy`.
If any symlink is broken (red in `ls`), the slug or inner filename differs — fix the slug/path and re-run.

- [ ] **Step 3: Commit**

```bash
git add scripts/pull_kaggle_data.sh
git commit -m "feat(h200): kaggle data pull + symlink to config paths"
```

---

### Task 3: Input-alignment verification script

**Files:**
- Create: `scripts/verify_inputs.py`

This ports the notebook's VERIFY 1–3 logic into a standalone gate. Per spec principle (regenerate over
force-fit): if a check fails because an artifact is stale/misaligned, that artifact gets re-precomputed in
Plan 2 — do not patch shims here.

- [ ] **Step 1: Write the verification script**

```python
#!/usr/bin/env python
"""Verify AMPR v3 inputs are mutually aligned before training.

Checks (per branch config):
  - protein_order length == labels rows == ppi rows == ppi_mask len
  - labels cols == dag cols == go_emb rows == n_terms
  - config dims: seq.d_model == ESM-2 residue dim; ppi.in_dim == ppi dim
  - every split protein has an ESM-2 and a cmap entry
  - DAG orientation matches loss.py (child->parent has fewer TPR violations)

Usage:
  python scripts/verify_inputs.py --config configs/mf_v3.yaml
Exit code 0 = all pass, 1 = any fail.
"""
import argparse, json, sys
import numpy as np, yaml, h5py


def load_order(path):
    o = json.loads(open(path).read())
    if isinstance(o, dict):
        o = [k for k, _ in sorted(o.items(), key=lambda kv: kv[1])]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg['data']
    n_terms = cfg['n_terms']
    seq_dim_cfg = cfg['model']['seq']['d_model']
    ppi_dim_cfg = cfg['model']['ppi']['in_dim']

    order = load_order(d['protein_order'])
    labels = np.load(d['labels'])
    ppi = np.load(d['ppi_emb'])
    ppi_mask = np.load(d['ppi_mask'])
    dag = np.load(d['dag_matrix'])
    go_emb = np.load(d['go_emb'])
    splits = json.loads(open(d['splits']).read())

    fails = []
    def check(name, cond, detail=""):
        tag = 'PASS' if cond else 'FAIL'
        print(f"[{tag}] {name}  {detail}")
        if not cond:
            fails.append(name)

    N = len(order)
    check("protein_order==labels rows", labels.shape[0] == N, f"{labels.shape[0]} vs {N}")
    check("ppi rows==N", ppi.shape[0] == N, f"{ppi.shape[0]} vs {N}")
    check("ppi_mask len==N", ppi_mask.shape[0] == N, f"{ppi_mask.shape[0]} vs {N}")
    check("labels cols==n_terms", labels.shape[1] == n_terms, f"{labels.shape[1]} vs {n_terms}")
    check("dag is n_terms x n_terms", dag.shape == (n_terms, n_terms), str(dag.shape))
    check("go_emb rows==n_terms", go_emb.shape[0] == n_terms, f"{go_emb.shape[0]} vs {n_terms}")
    check("ppi dim==cfg ppi.in_dim", ppi.shape[1] == ppi_dim_cfg, f"{ppi.shape[1]} vs {ppi_dim_cfg}")

    prot2idx = {p: i for i, p in enumerate(order)}
    for k in ('train', 'valid'):
        miss = [p for p in splits.get(k, []) if p not in prot2idx]
        check(f"split '{k}' subset of protein_order", not miss, f"missing={len(miss)}")

    with h5py.File(d['esm2_h5'], 'r') as fe, h5py.File(d['cmap_h5'], 'r') as fc:
        ek, ck = set(fe.keys()), set(fc.keys())
        any_pid = next(iter(splits['train']))
        seq_dim = fe[any_pid].shape[1]
        check("ESM-2 residue dim==cfg seq.d_model", seq_dim == seq_dim_cfg, f"{seq_dim} vs {seq_dim_cfg}")
        for k in ('train', 'valid'):
            ids = splits.get(k, [])
            check(f"'{k}' all have ESM-2", all(p in ek for p in ids),
                  f"missing={sum(p not in ek for p in ids)}")
            check(f"'{k}' all have cmap", all(p in ck for p in ids),
                  f"missing={sum(p not in ck for p in ids)}")

    L = labels.astype(np.float32)
    viol_a = float(((L @ dag) * (1 - L)).sum())     # dag[child,parent] (matches loss.py)
    viol_b = float(((L @ dag.T) * (1 - L)).sum())
    check("DAG orientation matches loss.py (A<B)", viol_a < viol_b, f"A={viol_a:.0f} B={viol_b:.0f}")

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it for all three branches**

Run: `python scripts/verify_inputs.py --config configs/mf_v3.yaml`
Expected: every line `[PASS]`, final `ALL PASS`, exit 0.
Repeat for `bp_v3.yaml`, `cc_v3.yaml`.

Note: with the *current* Kaggle `esm2_residue.h5` (650M, 1280d) and `mf_v3.yaml` (`seq.d_model: 1280`)
the dim check passes. After Plan 2 swaps to ESM2-3B you re-run this against the `*_esm3b.yaml` configs.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_inputs.py
git commit -m "feat(h200): standalone input-alignment verification gate"
```

---

### Task 4: H200-sized training config

**Files:**
- Modify: `configs/mf_v3.yaml:24-25,35` (and the same lines in `bp_v3.yaml`, `cc_v3.yaml`)

- [ ] **Step 1: Raise batch size and workers for H200**

In `configs/mf_v3.yaml`, change the `training` block:

```yaml
training:
  epochs: 50
  batch_size: 128       # was 16 — H200 141GB; cmap LxL + ESM residue fit easily at 128
  lr: 1.0e-3
  loss_type: asl
  asl_gamma_neg: 4
  asl_gamma_pos: 0
  asl_clip: 0.05
  lambda_dag: 0.5
  seed: 42
  device: auto
  max_seq_len: 1000
  num_workers: 16       # was 2 — 192-core box
```

Apply the identical `batch_size: 128` / `num_workers: 16` change to `configs/bp_v3.yaml` and
`configs/cc_v3.yaml`.

- [ ] **Step 2: Smoke-test one config with a dry run**

Run: `python main.py --config configs/mf_v3.yaml --dry_run`
Expected: log shows `[V3] --dry_run: train=50, val=20`, one epoch completes, a `[DIAG]` line prints,
no OOM, no shape error.

- [ ] **Step 3: Commit**

```bash
git add configs/mf_v3.yaml configs/bp_v3.yaml configs/cc_v3.yaml
git commit -m "feat(h200): bump v3 batch_size 128 / num_workers 16 for H200"
```

---

### Task 5: tmux launch orchestration

**Files:**
- Create: `scripts/launch_baseline.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# scripts/launch_baseline.sh — one detached tmux session per branch, one GPU each.
# Usage: bash scripts/launch_baseline.sh
set -euo pipefail
cd /raid/team/datn

declare -A GPU=( [mf]=0 [bp]=1 [cc]=2 )
for b in mf bp cc; do
  g=${GPU[$b]}
  sess="train_$b"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$g python main.py --config configs/${b}_v3.yaml 2>&1 | tee logs/${b}_v3_h200.log"
  echo "[LAUNCH] $sess on GPU $g -> logs/${b}_v3_h200.log"
done
echo "[LAUNCH] attach with: tmux attach -t train_mf   (detach: Ctrl-b d)"
echo "[LAUNCH] watch GPUs:  watch -n1 nvidia-smi"
```

- [ ] **Step 2: Pre-flight verify, then launch**

Run:
```bash
python scripts/verify_inputs.py --config configs/mf_v3.yaml && \
python scripts/verify_inputs.py --config configs/bp_v3.yaml && \
python scripts/verify_inputs.py --config configs/cc_v3.yaml && \
bash scripts/launch_baseline.sh
```
Expected: three `ALL PASS`, then three `[LAUNCH]` lines. `tmux ls` shows `train_mf`, `train_bp`, `train_cc`.

- [ ] **Step 3: Confirm training is live**

Run: `sleep 60; tail -n 5 logs/mf_v3_h200.log`
Expected: a `[V3] Epoch 1/50: loss=... val_Fmax_raw=... val_Fmax_dag=...` line (or in-epoch progress),
and `nvidia-smi` shows GPUs 0–2 active. MF baseline target: val Fmax (DAG-prop) ≈ 0.678 (per prior run).

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_baseline.sh
git commit -m "feat(h200): tmux launcher for MF/BP/CC baseline on GPUs 0-2"
```

---

### Task 6: Operator runbook

**Files:**
- Create: `docs/RUNBOOK_H200.md`

- [ ] **Step 1: Write the runbook**

```markdown
# RUNBOOK — AMPR on 8×H200 (Hackathon)

## Access
- JupyterLab: http://<IP>:8888/lab  (password from VTS) — inspection only
- SSH: `ssh <user>@<IP>` — all training via tmux

## Cold start (one-time)
1. `ssh <user>@<IP>` ; ensure repo at `/raid/team/datn` or run `scripts/server_setup.sh`
2. Upload `kaggle.json` to `~/.kaggle/`, then `bash scripts/server_setup.sh`
3. `bash scripts/pull_kaggle_data.sh`
4. `python scripts/verify_inputs.py --config configs/mf_v3.yaml`  (repeat bp/cc) → ALL PASS

## Train baseline (Track 1)
- `bash scripts/launch_baseline.sh`  → GPUs 0–2, sessions train_{mf,bp,cc}
- Monitor: `tmux attach -t train_mf` (detach Ctrl-b d) ; `watch -n1 nvidia-smi`
- Logs: `logs/{mf,bp,cc}_v3_h200.log` ; checkpoints: `checkpoints/{mf,bp,cc}_v3/best.pt`

## GPU map
- 0–2: baseline (Track 1).  3–7: free for upgrade precompute/sweep (Plans 2–3).

## Gotchas
- Jobs survive browser close (tmux). They survive container restart only if started under tmux that
  itself survives — prefer re-launching after a restart; `/raid` data persists regardless.
- OOM: lower `batch_size`. Never edit driver/CUDA/Docker-root (hackathon rule).
- Stale/misaligned artifact → regenerate it (Plan 2), do not shim.
```

- [ ] **Step 2: Commit**

```bash
git add docs/RUNBOOK_H200.md
git commit -m "docs(h200): operator runbook for migration + baseline"
```

---

## Self-Review

**Spec coverage (Phases 0–2):**
- Phase 0 bootstrap → Task 1 ✓ (env, GPU sanity, DGL-free check)
- Phase 1 data transfer + verify → Tasks 2, 3 ✓
- Phase 2 baseline training (batch/workers bump, tmux per branch) → Tasks 4, 5 ✓
- Runbook deliverable → Task 6 ✓
- Spec "regenerate over force-fit" principle → referenced in Tasks 3 & 6 ✓
- Phases 3–4 (embeddings, architecture, sweep) → **out of scope here; Plans 2 & 3.**

**Placeholder scan:** Dataset slugs are real values from the notebook with an explicit "adjust if live list
differs" instruction (a genuine runtime unknown, not a placeholder). No TBD/TODO/"add error handling".

**Type/name consistency:** `verify_inputs.py --config` signature matches usage in Tasks 5 & 6;
config keys (`batch_size`, `num_workers`, `seq.d_model`, `ppi.in_dim`) match `main.py` `_run_v3` and the
config files; tmux session names (`train_mf/bp/cc`) and log paths consistent across Tasks 5 & 6.
