# AMPR Embedding Upgrades (ESM2-3B + SapBERT GO + GO-graph) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the input embeddings with the strongest feasible encoders from FASTA-only inputs — ESM2-3B per-residue, a SapBERT GO-term encoder, and a GO-ontology graph embedding concatenated onto the GO labels — sharded across the 8 H200 GPUs.

**Architecture:** Pure-function helpers (shard selection, GO text building, graph spectral embedding, L2-concat) are unit-tested locally with pytest on the Anaconda env; the GPU-bound encoding loops are smoke-run on the server. New `*_v3_esm3b.yaml` configs point the existing v3 pipeline at the new artifacts. `go_emb_dim` is read from the loaded array shape (`main.py` `_run_v3`), so no model code changes are needed for the new GO dim.

**Tech Stack:** PyTorch, HuggingFace Transformers (ESM-2, SapBERT), h5py, numpy/scipy, obonet, PyYAML.

**Spec:** [docs/superpowers/specs/2026-06-01-ampr-h200-migration-design.md](../specs/2026-06-01-ampr-h200-migration-design.md) (Phase 3)

**Prereq:** Plan 1 done — `/raid/team/datn` populated, `verify_inputs.py` exists, FASTA present at
`data/pdbch/nrPDB-GO_2019.06.18_sequences.fasta` (pull it from Kaggle alongside the other artifacts if not).

**Local testing:** `python -m pytest tests/<file> -v` from repo root (Anaconda env, per CLAUDE.md). Never bare `pytest`.

---

### Task 1: Shard-selection helper (pure, TDD)

**Files:**
- Modify: `scripts/precompute_esm2_residue.py`
- Test: `tests/test_shard_select.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shard_select.py
from scripts.precompute_esm2_residue import select_shard


def test_shard_partitions_disjoint_and_complete():
    ids = [f"p{i}" for i in range(10)]
    shards = [select_shard(ids, s, 3) for s in range(3)]
    # disjoint
    seen = sum(shards, [])
    assert sorted(seen) == sorted(ids)
    # roughly balanced
    assert all(2 <= len(s) <= 4 for s in shards)


def test_shard_single_returns_all():
    ids = ["a", "b", "c"]
    assert select_shard(ids, 0, 1) == ids
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_shard_select.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_shard'`.

- [ ] **Step 3: Implement `select_shard`**

Add near the top of `scripts/precompute_esm2_residue.py` (after imports):

```python
def select_shard(ordered_ids, shard: int, nshards: int):
    """Deterministic contiguous partition: shard i of nshards over ordered_ids."""
    if nshards <= 1:
        return list(ordered_ids)
    n = len(ordered_ids)
    lo = (n * shard) // nshards
    hi = (n * (shard + 1)) // nshards
    return list(ordered_ids[lo:hi])
```

- [ ] **Step 4: Run it to confirm pass**

Run: `python -m pytest tests/test_shard_select.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/precompute_esm2_residue.py tests/test_shard_select.py
git commit -m "feat(precompute): add deterministic select_shard helper (TDD)"
```

---

### Task 2: `--model` and `--shard` flags on the ESM-2 precompute

**Files:**
- Modify: `scripts/precompute_esm2_residue.py:55-64,88-109`

- [ ] **Step 1: Parametrize the iterator by model name**

Change the iterator signature (currently `_esm2_iterator(seqs, ordered_ids, existing, batch, max_len)`)
and the two hardcoded `facebook/esm2_t33_650M_UR50D` references:

```python
def _esm2_iterator(seqs: dict, ordered_ids: list, existing: set,
                   batch: int, max_len: int, model_name: str):
    """Yield (pid, residue_emb) by running the given ESM-2 model in batches."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"[ESM2] device={device} model={model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
```

(The body that builds `hs = out.last_hidden_state` is unchanged; it already adapts to the model's hidden
dim — 2560 for 3B.)

- [ ] **Step 2: Add CLI flags and wire sharding**

Replace the `main()` arg block + the iterator call:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--protein_order', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max_len', type=int, default=1022)
    ap.add_argument('--model', default='facebook/esm2_t33_650M_UR50D',
                    help='HF model id, e.g. facebook/esm2_t36_3B_UR50D')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    seqs = _load_fasta(args.fasta)
    ordered = json.loads(Path(args.protein_order).read_text())
    if isinstance(ordered, dict):
        ordered = [k for k, _ in sorted(ordered.items(), key=lambda kv: kv[1])]
    ordered = select_shard(ordered, args.shard, args.nshards)
    logger.info(f"[ESM2] shard {args.shard}/{args.nshards} -> {len(ordered)} proteins")

    existing = set()
    if Path(args.out).exists():
        with h5py.File(args.out, 'r') as f:
            existing = set(f.keys())

    write_residue_h5(
        _esm2_iterator(seqs, ordered, existing, args.batch, args.max_len, args.model),
        args.out)
    with h5py.File(args.out, 'r') as f:
        logger.info(f"[ESM2] DONE — {len(f.keys())} proteins in {args.out}")
```

- [ ] **Step 3: Confirm existing smoke test still passes**

Run: `python -m pytest tests/test_precompute_esm2_smoke.py -v`
Expected: PASS (if it loads the real model it may skip/require GPU — confirm it does not error on the new
signature; if it calls `_esm2_iterator`, update that call to pass a `model_name`).

- [ ] **Step 4: Commit**

```bash
git add scripts/precompute_esm2_residue.py
git commit -m "feat(precompute): --model + --shard flags for multi-GPU ESM-2"
```

---

### Task 3: Run sharded ESM2-3B precompute on 5 GPUs

**Files:**
- Create: `scripts/launch_esm3b_precompute.sh`

- [ ] **Step 1: Write the sharded launcher**

```bash
#!/usr/bin/env bash
# scripts/launch_esm3b_precompute.sh — ESM2-3B residue embeddings across GPUs 3-7.
set -euo pipefail
cd /raid/team/datn
FASTA=data/pdbch/nrPDB-GO_2019.06.18_sequences.fasta
ORDER=data/pdbch/protein_order.json
OUT=data/embeddings/esm2_3b_residue.h5   # one shared file (resumable, skips existing keys)
MODEL=facebook/esm2_t36_3B_UR50D
N=5
for i in 0 1 2 3 4; do
  gpu=$((i + 3))
  sess="esm3b_$i"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpu python scripts/precompute_esm2_residue.py \
       --fasta $FASTA --protein_order $ORDER --out $OUT \
       --model $MODEL --batch 8 --max_len 1022 --shard $i --nshards $N \
       2>&1 | tee logs/esm3b_shard$i.log"
  echo "[ESM3B] shard $i on GPU $gpu"
done
echo "[ESM3B] watch: tail -f logs/esm3b_shard*.log ; verify count after: python - <<'PY'
import h5py; print(len(h5py.File('$OUT').keys()))
PY"
```

Note: all shards append to one resumable HDF5 (`write_residue_h5` skips existing keys), and shards are
disjoint id ranges, so there are no write collisions on distinct keys. If you prefer zero shared-file risk,
write per-shard files (`--out esm2_3b_shard$i.h5`) and merge after.

- [ ] **Step 2: Launch and verify count**

Run: `bash scripts/launch_esm3b_precompute.sh`
After completion (`tmux ls` shows sessions ended): 
```bash
python -c "import h5py; f=h5py.File('data/embeddings/esm2_3b_residue.h5'); k=list(f.keys()); print(len(k), f[k[0]].shape)"
```
Expected: ~36 641 keys, second value `(L, 2560)`.

- [ ] **Step 3: Commit**

```bash
git add scripts/launch_esm3b_precompute.sh
git commit -m "feat(precompute): sharded ESM2-3B launcher (GPUs 3-7)"
```

---

### Task 4: GO text builder (pure, TDD)

**Files:**
- Create: `scripts/precompute_go_text.py`
- Test: `tests/test_go_text_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_go_text_builder.py
from scripts.precompute_go_text import build_go_texts


def test_build_go_texts_name_and_def():
    nodes = {
        "GO:0001": {"name": "catalytic activity",
                    "def": '"Catalysis of a reaction." [GOC:x]'},
        "GO:0002": {"name": "binding"},          # no def
    }
    texts = build_go_texts(["GO:0001", "GO:0002", "GO:9999"], nodes)
    assert texts[0] == "catalytic activity. Catalysis of a reaction."
    assert texts[1] == "binding."
    assert texts[2] == "GO:9999."                # missing node -> id as name
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_go_text_builder.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement the script with the pure builder + a generic encoder**

```python
#!/usr/bin/env python
"""Encode GO terms (name + definition) with a configurable HF text encoder.

Generalizes the old BioBERT-only encoder. Default = SapBERT (entity-aware).
Output dim is inferred from the model (768 for SapBERT/PubMedBERT-base,
1024 for *-large), and main.py reads go_emb_dim from the saved array shape.

Usage:
  python scripts/precompute_go_text.py \
    --go_terms data/pdbch/go_terms_mf.json \
    --obo data/pdbch/go-basic.obo \
    --out data/embeddings/go_text_mf.npy \
    --model cambridgeltl/SapBERT-from-PubMedBERT-fulltext
"""
import argparse, json
from pathlib import Path
import numpy as np


def build_go_texts(go_term_ids, graph_nodes):
    """Pure: GO ids + node dict -> 'name. definition.' strings."""
    texts = []
    for gid in go_term_ids:
        node = graph_nodes.get(gid, {})
        name = node.get("name", gid)
        raw = node.get("def", "")
        defn = raw.split('"')[1] if '"' in raw else ""
        texts.append(f"{name}. {defn}".strip())
    return texts


def encode(texts, model_name, batch=16, max_len=128):
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    dim = model.config.hidden_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=max_len, return_tensors='pt').to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out[i:i + pooled.size(0)] = pooled.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--go_terms', required=True)
    ap.add_argument('--obo', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    args = ap.parse_args()
    import obonet
    graph = obonet.read_obo(args.obo)
    go_ids = json.loads(Path(args.go_terms).read_text())
    texts = build_go_texts(go_ids, dict(graph.nodes(data=True)))
    emb = encode(texts, args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, emb)
    print(f"[GO-TEXT] saved {args.out} shape={emb.shape} model={args.model}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run it to confirm the builder test passes**

Run: `python -m pytest tests/test_go_text_builder.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Encode all three branches on the server (GPU 3)**

```bash
for b in mf bp cc; do
  CUDA_VISIBLE_DEVICES=3 python scripts/precompute_go_text.py \
    --go_terms data/pdbch/go_terms_$b.json --obo data/pdbch/go-basic.obo \
    --out data/embeddings/go_text_$b.npy
done
```
Expected: three `[GO-TEXT] saved … shape=(489|1943|320, 768)`.

- [ ] **Step 6: Commit**

```bash
git add scripts/precompute_go_text.py tests/test_go_text_builder.py
git commit -m "feat(go): configurable SapBERT GO-text encoder (TDD pure builder)"
```

---

### Task 5: GO-graph spectral embedding (pure, TDD)

**Files:**
- Create: `scripts/precompute_go_graph.py`
- Test: `tests/test_go_graph_embedding.py`

Rationale: a dependency-light, deterministic topology embedding = truncated SVD of the symmetric
normalized GO-DAG adjacency (Node2Vec is a heavier, stochastic alternative; SVD is testable and reproducible).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_go_graph_embedding.py
import numpy as np
from scripts.precompute_go_graph import graph_embedding


def test_graph_embedding_shape_and_determinism():
    rng = np.random.default_rng(0)
    n = 20
    dag = (rng.random((n, n)) < 0.2).astype(np.float32)
    np.fill_diagonal(dag, 0)
    e1 = graph_embedding(dag, dim=8)
    e2 = graph_embedding(dag, dim=8)
    assert e1.shape == (n, 8)
    assert np.allclose(e1, e2)            # deterministic


def test_graph_embedding_dim_caps_at_rank():
    dag = np.zeros((5, 5), dtype=np.float32)
    dag[1, 0] = dag[2, 0] = 1.0          # tiny graph
    e = graph_embedding(dag, dim=16)     # dim > nodes
    assert e.shape == (5, 16)            # zero-padded, no crash
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_go_graph_embedding.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""GO-ontology graph embedding via truncated SVD of the normalized DAG adjacency.

Deterministic and dependency-light (numpy/scipy). Concatenated later with the
SapBERT text embedding to give each GO term both semantic and topological signal.

Usage:
  python scripts/precompute_go_graph.py \
    --dag data/pdbch/dag_matrix_mf.npy \
    --out data/embeddings/go_graph_mf.npy --dim 128
"""
import argparse
from pathlib import Path
import numpy as np


def graph_embedding(dag: np.ndarray, dim: int = 128) -> np.ndarray:
    """Symmetrize -> add self-loops -> sym-normalize -> truncated SVD -> (N, dim)."""
    n = dag.shape[0]
    A = ((dag + dag.T) > 0).astype(np.float64)
    np.fill_diagonal(A, 1.0)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    A_norm = (A * dinv[:, None]) * dinv[None, :]
    # Deterministic SVD (sign-fixed by largest-magnitude component per column)
    U, S, _ = np.linalg.svd(A_norm)
    k = min(dim, n)
    emb = U[:, :k] * S[:k]
    for j in range(k):
        if emb[np.argmax(np.abs(emb[:, j])), j] < 0:
            emb[:, j] = -emb[:, j]
    if k < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - k)))
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dim', type=int, default=128)
    args = ap.parse_args()
    dag = np.load(args.dag)
    emb = graph_embedding(dag, args.dim)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, emb)
    print(f"[GO-GRAPH] saved {args.out} shape={emb.shape}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run it to confirm pass**

Run: `python -m pytest tests/test_go_graph_embedding.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Generate for all branches (CPU is fine)**

```bash
for b in mf bp cc; do
  python scripts/precompute_go_graph.py --dag data/pdbch/dag_matrix_$b.npy \
    --out data/embeddings/go_graph_$b.npy --dim 128
done
```
Expected: three `[GO-GRAPH] saved … shape=(489|1943|320, 128)`.

- [ ] **Step 6: Commit**

```bash
git add scripts/precompute_go_graph.py tests/test_go_graph_embedding.py
git commit -m "feat(go): deterministic GO-graph SVD embedding (TDD)"
```

---

### Task 6: Combine GO text + graph (pure, TDD)

**Files:**
- Create: `scripts/build_go_combined.py`
- Test: `tests/test_go_combined.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_go_combined.py
import numpy as np
from scripts.build_go_combined import l2norm_concat


def test_l2norm_concat_shape_and_unit_blocks():
    text = np.random.randn(4, 6).astype(np.float32)
    graph = np.random.randn(4, 3).astype(np.float32)
    out = l2norm_concat(text, graph)
    assert out.shape == (4, 9)
    # each block L2-normalized per row
    assert np.allclose(np.linalg.norm(out[:, :6], axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(out[:, 6:], axis=1), 1.0, atol=1e-5)


def test_l2norm_concat_row_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        l2norm_concat(np.zeros((4, 6), np.float32), np.zeros((3, 3), np.float32))
```

- [ ] **Step 2: Run it to confirm failure**

Run: `python -m pytest tests/test_go_combined.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Concatenate L2-normalized GO text + graph embeddings -> final GO label matrix.

Usage:
  python scripts/build_go_combined.py \
    --text data/embeddings/go_text_mf.npy \
    --graph data/embeddings/go_graph_mf.npy \
    --out data/embeddings/go_emb_mf_v2.npy
"""
import argparse
from pathlib import Path
import numpy as np


def l2norm_concat(text: np.ndarray, graph: np.ndarray) -> np.ndarray:
    if text.shape[0] != graph.shape[0]:
        raise ValueError(f"row mismatch: text {text.shape[0]} vs graph {graph.shape[0]}")
    def norm(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    return np.concatenate([norm(text), norm(graph)], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True)
    ap.add_argument('--graph', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = l2norm_concat(np.load(args.text), np.load(args.graph))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"[GO-COMBINED] saved {args.out} shape={out.shape}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run it to confirm pass**

Run: `python -m pytest tests/test_go_combined.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Build combined GO embeddings**

```bash
for b in mf bp cc; do
  python scripts/build_go_combined.py \
    --text data/embeddings/go_text_$b.npy \
    --graph data/embeddings/go_graph_$b.npy \
    --out data/embeddings/go_emb_${b}_v2.npy
done
```
Expected: three `[GO-COMBINED] saved … shape=(489|1943|320, 896)` (768 text + 128 graph).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_go_combined.py tests/test_go_combined.py
git commit -m "feat(go): L2-norm concat of text+graph GO embeddings (TDD)"
```

---

### Task 7: ESM2-3B + upgraded-GO configs

**Files:**
- Create: `configs/mf_v3_esm3b.yaml`, `configs/bp_v3_esm3b.yaml`, `configs/cc_v3_esm3b.yaml`

- [ ] **Step 1: Write `configs/mf_v3_esm3b.yaml`**

```yaml
branch: MF
n_terms: 489
data:
  protein_order: data/pdbch/protein_order.json
  splits:        data/pdbch/splits.json
  labels:        data/pdbch/labels_mf.npy
  dag_matrix:    data/pdbch/dag_matrix_mf.npy
  esm2_h5:       data/embeddings/esm2_3b_residue.h5     # NEW: 2560d
  ppi_emb:       data/embeddings/ppi_deepgo.npy
  ppi_mask:      data/embeddings/ppi_deepgo_mask.npy
  cmap_h5:       data/contact_maps/cmap_all.h5
  go_emb:        data/embeddings/go_emb_mf_v2.npy        # NEW: SapBERT+graph (896d)
  diamond_tsv:   data/diamond/diamond_results_mf.tsv
model:
  version: v3
  structure_modality: gnn
  seq: {d_model: 2560, n_transformer_layers: 2, n_heads: 8, dropout: 0.1}  # NEW d_model
  gnn: {node_dim: 256, n_layers: 3, cmap_threshold: 10.0}
  ppi: {in_dim: 256, hidden: 512}
  fusion: {d_model: 512, n_layers: 2, n_heads: 8}
  classifier: both
  d_hidden: 512
training:
  epochs: 50
  batch_size: 64          # 2560d residue is ~2x memory of 1280d; 64 safe on H200
  lr: 1.0e-3
  loss_type: asl
  asl_gamma_neg: 4
  asl_gamma_pos: 0
  asl_clip: 0.05
  lambda_dag: 0.5
  seed: 42
  device: auto
  max_seq_len: 1000
  num_workers: 16
inference:
  use_dag_propagation: true
  use_diamond_ensemble: true
  diamond_alpha: 0.6
  threshold_path: checkpoints/mf_v3_esm3b/threshold.json
output:
  checkpoint_dir: checkpoints/mf_v3_esm3b/
  log_file:       logs/mf_v3_esm3b_train.log
  results_file:   results/mf_v3_esm3b_predictions.tsv
seed: 42
```

For `bp_v3_esm3b.yaml` / `cc_v3_esm3b.yaml`: copy this file, set `branch`, `n_terms` (1943 / 320),
the `_bp`/`_cc` data paths, and the `checkpoint_dir`/`log_file`/`results_file` to the matching branch.
Note: `go_emb_dim` is **not** a config field — `main.py` `_run_v3` reads it from `go_emb.shape[1]`, so the
new 896d GO matrix needs no further wiring.

- [ ] **Step 2: Verify alignment with the Plan-1 gate**

Run: `python scripts/verify_inputs.py --config configs/mf_v3_esm3b.yaml`
Expected: `ALL PASS` — in particular `ESM-2 residue dim==cfg seq.d_model` now checks 2560==2560, and the
`go_emb rows==n_terms` check passes for the 896d matrix.

- [ ] **Step 3: Dry-run train smoke**

Run: `python main.py --config configs/mf_v3_esm3b.yaml --dry_run`
Expected: one epoch completes, no shape error, `[DIAG]` line prints (proves 2560d + 896d GO flow end-to-end).

- [ ] **Step 4: Commit**

```bash
git add configs/mf_v3_esm3b.yaml configs/bp_v3_esm3b.yaml configs/cc_v3_esm3b.yaml
git commit -m "feat(config): ESM2-3B + SapBERT/graph-GO v3 configs"
```

---

## Self-Review

**Spec coverage (Phase 3):**
- 3a ESM2-3B sharded precompute → Tasks 1, 2, 3 ✓
- 3c SapBERT GO text encoder → Task 4 ✓
- 3d GO-graph embedding → Task 5 ✓; text+graph concat → Task 6 ✓
- New `*_v3_esm3b.yaml` + alignment re-verify → Task 7 ✓
- "regenerate over force-fit": every artifact rebuilt from FASTA/obo/dag, none shimmed ✓
- 3b ProstT5 → **stretch, deferred (spec §7); not in this plan.**

**Placeholder scan:** No TBD/TODO. SapBERT/ESM-3B model ids are concrete HF ids. bp/cc config deltas are
spelled out explicitly (branch, n_terms, paths) rather than "similar to MF".

**Type/name consistency:** `select_shard(ids, shard, nshards)` signature identical in Tasks 1, 2, 3;
`build_go_texts(go_ids, graph_nodes)`, `graph_embedding(dag, dim)`, `l2norm_concat(text, graph)` match
across their tests and `main()` callers; output filenames (`esm2_3b_residue.h5`, `go_text_$b.npy`,
`go_graph_$b.npy`, `go_emb_${b}_v2.npy`) are consistent between producing tasks and the Task-7 config.
`go_emb_dim` correctly described as shape-derived (matches `main.py` `_run_v3` line `go_emb_dim = go_emb.shape[1]`).
