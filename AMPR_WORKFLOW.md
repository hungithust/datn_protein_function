# AMPR Workflow Guide

Quick reference for implementing AMPR. This document maps high-level tasks to implementation steps.

## Phase 1: Data Preparation (Colab cell)

**Goal:** Download raw data + precompute embeddings → `.npy` files

### 1.1 Download Raw Data

```bash
# In Colab cell:
!python scripts/01_download_data.py --output data/

# Downloads:
#   - pdb_seqres.txt.gz
#   - pdb_chain_go.tsv.gz
#   - bc-95.out
#   - go-basic.obo
#   - ppi_{mf,bp,cc}.bin (optional, from DeepGO)
```

**Outputs:** `data/nrPDB-GO_annot.tsv`, `data/nrPDB-GO_sequences.fasta`, `data/nrPDB-GO_train/valid/test.txt`

**Logging:** Each script logs counts, shapes, file sizes.

### 1.2 Create TFRecords (no contact maps)

```bash
# Convert FASTA + GO → TFRecord (one-time conversion)
!python scripts/seq2tfrecord.py \
  --annot data/nrPDB-GO_annot.tsv \
  --fasta data/nrPDB-GO_sequences.fasta \
  --split data/nrPDB-GO_train.txt \
  --out_prefix data/tfrecords/GO_train \
  --num_shards 10 --num_threads 2

# Repeat for valid/test:
!python scripts/seq2tfrecord.py ... --split data/nrPDB-GO_valid.txt --out_prefix data/tfrecords/GO_valid
!python scripts/seq2tfrecord.py ... --split data/nrPDB-GO_test.txt  --out_prefix data/tfrecords/GO_test
```

**Outputs:** `data/tfrecords/GO_{train,valid,test}_*-of-*.tfrecords`

**Logging:** [SHARD XX/YY] lines, final count of written vs skipped proteins.

### 1.3 Precompute Embeddings (2-4 hours on T4)

```bash
# Run once, saves .npy files
!python scripts/02_precompute_embeddings.py --config configs/mf.yaml
```

**Outputs:**
```
data/embeddings/
├── seq_embeddings.npy         (N, 1024) ProteinBERT
├── struct_embeddings.npy      (N, 1024) ProstT5
├── ppi_embeddings.npy         (N, 128)  Node2Vec
└── go_emb_mf/bp/cc.npy        (C, 768)  BioBERT GO definitions
```

**Logging:** [EMBED] batch progress, final shape + MB size for each file.

### 1.4 Build DAG Matrix

```bash
!python scripts/03_build_dag_matrix.py --obo data/go-basic.obo --out data/
```

**Outputs:**
```
data/dag_matrices/
├── dag_matrix_mf.npy  (C_mf, C_mf)
├── dag_matrix_bp.npy  (C_bp, C_bp)
└── dag_matrix_cc.npy  (C_cc, C_cc)
```

**Logging:** [DAG] number of parent-child edges per ontology.

---

## Phase 2: Training (loop 3x for MF/BP/CC)

**Goal:** Train AMPR model for one GO branch

### 2.1 Train MF Model

```bash
!python main.py --config configs/mf.yaml
```

**Config `configs/mf.yaml` specifies:**
```yaml
branch: MF
n_terms: 489
data:
  seq_emb: data/embeddings/seq_embeddings.npy
  struct_emb: data/embeddings/struct_embeddings.npy
  ppi_emb: data/embeddings/ppi_embeddings.npy
  labels: data/tfrecords/GO_train_*-of-*.tfrecords  OR .npy
  dag_matrix: data/dag_matrices/dag_matrix_mf.npy
  go_emb: data/embeddings/go_emb_mf.npy
  splits: data/splits.json

model:
  d_hidden: 512
  dropout_3di: 0.15
  dropout_ppi: 0.25
  classifier: both      # "linear" | "biobert" | "both"

training:
  epochs: 50
  batch_size: 256
  lr: 1e-3
  lambda_dag: 0.5
  device: auto          # auto-detect GPU/CPU

output:
  checkpoint_dir: checkpoints/mf/
  log_file: logs/mf_train.log
```

**Logging:**
```
[EPOCH 01/50] lr=1.0e-3 | loss=0.3421 (bce=0.2891 dag=0.0530)
              α_seq=0.612 α_3di=0.251 α_ppi=0.137
              val Fmax=0.412 | AUPRC=0.389
[EPOCH 02/50] lr=9.8e-4 | loss=0.3187 (bce=0.2701 dag=0.0486)
              α_seq=0.589 α_3di=0.278 α_ppi=0.133
              val Fmax=0.431 | AUPRC=0.401
...
[BEST] Epoch 23: Fmax=0.456 → saved checkpoints/mf/best.pt
[FINAL] Test Fmax=0.451 | Smin=6.23 | AUPRC=0.417
```

**Outputs:**
```
checkpoints/mf/best.pt          # best validation checkpoint
logs/mf_train.log               # full training log
results/mf_predictions.tsv      # test set predictions + labels
```

### 2.2 Repeat for BP and CC

```bash
!python main.py --config configs/bp.yaml
!python main.py --config configs/cc.yaml
```

Same structure, different branch + C terms.

---

## Phase 3: Evaluation & Analysis

**Goal:** Analyze results, visualize alpha weights, prepare thesis figures

### 3.1 Per-Branch Results

```python
# In notebook, load results
import pandas as pd

results = {
    'MF': pd.read_csv('results/mf_predictions.tsv'),
    'BP': pd.read_csv('results/bp_predictions.tsv'),
    'CC': pd.read_csv('results/cc_predictions.tsv'),
}

# Compare:
#   - Fmax, Smin, AUPRC per branch
#   - Number of proteins in test set
#   - Alpha weight evolution (from logs)
```

### 3.2 Modality Contribution Analysis

Parse log files → extract alpha weights per epoch:

```python
# Extract from logs/mf_train.log
# [EPOCH XX] α_seq=0.612 α_3di=0.251 α_ppi=0.137
# Visualize: how do weights shift during training?
```

### 3.3 Ablation Study

Train multiple model variants:

| Config | classifier | modalities | Expected result |
|---|---|---|---|
| mf.yaml | both | all 3 | Best (ensemble) |
| mf_linear_only.yaml | linear | all 3 | Baseline |
| mf_biobert_only.yaml | biobert | all 3 | Zero-shot variant |
| mf_seq_only.yaml | linear | seq only | Ablate PPI+3Di |

Compare Fmax across variants → table for thesis.

---

## Phase 4: Reporting Results

### 4.1 Main Results Table

| Branch | Classifier | Fmax | Smin | AUPRC | α_seq | α_3di | α_ppi |
|---|---|---|---|---|---|---|---|
| MF | Both | 0.456 | 6.23 | 0.417 | 0.612 | 0.251 | 0.137 |
| MF | Linear | 0.449 | 6.41 | 0.410 | N/A | N/A | N/A |
| BP | Both | 0.412 | 7.65 | 0.389 | 0.601 | 0.268 | 0.131 |
| CC | Both | 0.478 | 5.87 | 0.442 | 0.623 | 0.231 | 0.146 |

### 4.2 Figures

1. **Alpha weight evolution** — line plots over epochs (per branch)
2. **Loss curves** — BCE vs DAG loss decomposition
3. **Modality contribution** — bar chart of final alpha values
4. **ROC/PR curves** — for each branch
5. **GO term distribution** — how many terms per protein in train/valid/test

---

## Troubleshooting

### OOM on Colab T4

- Reduce `batch_size` from 256 → 128
- Reduce `d_hidden` from 512 → 256
- Use Kaggle (2x T4) instead

### TFRecord not found

- Check split file exists: `data/nrPDB-GO_train.txt`
- Run seq2tfrecord.py with correct `--out_prefix`

### Alpha weights not changing

- Might be stuck at init weights
- Check learning rate (should be > 1e-4)
- Verify loss is decreasing (BCE term)

### Missing proteins in TFRecords

- seq2tfrecord.py logs [STATS] skipped count
- Likely: sequence too short (<10 AA) or not in FASTA
- Acceptable if <1% of proteins skipped

---

## Time Budget

| Step | Runtime | Hardware |
|---|---|---|
| Download data | ~30 min | Network I/O |
| Precompute embeddings | 2-4 hours | T4 GPU |
| Create TFRecords | ~30 min | CPU (gzip) |
| Build DAG matrix | <1 min | CPU |
| Train 1 model (50 epochs) | 1-2 hours | T4 GPU |
| **Total (3 branches)** | ~8-10 hours | One Colab session |

---

## Code Review Checklist (before thesis submission)

- [ ] All 3 models (MF/BP/CC) train without error
- [ ] Loss curves decrease monotonically (or plateau)
- [ ] Validation Fmax improves over first 10 epochs
- [ ] Alpha weights are non-zero and change during training
- [ ] Test set Fmax ≥ 0.40 for all branches
- [ ] DAG loss term is contributing (not 0)
- [ ] Logs are clear and reproducible
- [ ] No hardcoded paths (all relative or parameterized)
- [ ] Comments explain WHY not WHAT
- [ ] Git history is clean (logical commits)

---

**Status:** Brainstorm phase complete. Ready for implementation on Colab.
