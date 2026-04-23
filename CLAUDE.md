# AMPR: Adaptive Multimodal Protein Representation

**Thesis Topic:** Deep learning-based protein function prediction with Adaptive Multimodal Representation  
**Student:** Nguyen Viet Hung (20224998)  
**Execution Environment:** Google Colab (Python 3.12.13, T4 15GB) or Kaggle (Python 3.12.12, 2x T4)  
**Status:** Brainstorming → Implementation → Thesis writing

## Quick Links

- **Design Spec:** [docs/superpowers/specs/2026-04-23-ampr-design.md](docs/superpowers/specs/2026-04-23-ampr-design.md)
- **Implementation Plan:** [.claude/plans/context-t-i-ang-tri-n-toasty-dove.md](.claude/plans/context-t-i-ang-tri-n-toasty-dove.md)

## Project Overview

AMPR is an **Adaptive Multimodal Protein Representation** framework for predicting Gene Ontology (GO) terms from protein sequences. The model addresses three key challenges:

1. **Missing modality robustness** — some proteins lack structural/PPI data; model uses adaptive gating to weight available modalities
2. **GO hierarchy consistency** — enforces True Path Rule via custom DAG-constrained loss
3. **Efficient training on limited hardware** — all embeddings precomputed; train only small fusion + classifier layers

### 3 GO Branches

Train separate models for:
- **MF (Molecular Function)** — ~489 terms
- **BP (Biological Process)** — ~1,943 terms  
- **CC (Cellular Component)** — ~320 terms

Same code, different config YAML.

## Architecture

```
ProteinBERT(1024d)  ─┐
                     ├→ Linear(→512d) ──┐
ProstT5(1024d)      ─┤                  │
                     │                  ├→ [3 vectors] → Gating MLP
Node2Vec(128d)      ─┼→ Linear(→512d) ──┤  → α weights
                     │                  │
BioBERT-GO(768d) ───┴→ Linear(→512d) ──┘  (fixed GO weights OR
                                           trainable Linear head)
```

Weighted sum: `Z = α_seq·h_seq + α_3di·h_3di + α_ppi·h_ppi` (512d)

**Loss:** `L = BCE + λ·DAG_loss` (enforces child-parent GO relationships)

## Code Organization

```
ampr/                          # Python package
├── data/                       # Dataset + GO hierarchy
├── embeddings/                 # Precompute scripts
├── models/                     # Fusion, classifier, AMPR
├── training/                   # Loss, trainer, logging
└── evaluation/                 # Metrics (Fmax, Smin, AUPRC)

configs/                        # Per-branch YAML
├── mf.yaml
├── bp.yaml
└── cc.yaml

scripts/                        # Data processing
├── seq2tfrecord.py            # Convert FASTA+GO → TFRecord (no contact maps)
├── 01_download_data.py        # wget sequences, annotations
├── 02_precompute_embeddings.py # ProteinBERT, ProstT5, BioBERT
├── 03_build_dag_matrix.py     # go-basic.obo → matrix
└── 04_run_node2vec.py         # DGL PPI → embeddings

notebooks/
├── colab_run.ipynb            # Colab entry point
└── kaggle_run.ipynb           # Kaggle entry point

main.py                         # CLI: python main.py --config configs/mf.yaml
```

## Principles

1. **Module clarity** — each subpackage has one clear responsibility
2. **Config-driven** — YAML controls all training hyperparameters, no hardcoding
3. **Logging explicit** — every step logs shapes, counts, progress
4. **GPU/CPU auto** — `device: auto` in config detects CUDA; DataParallel for multi-GPU
5. **Notebook-native** — scripts run via `!python ...` cells in Colab/Kaggle
6. **No local execution** — this is a Colab-first project

## Running on Colab/Kaggle

### Step 1: Setup (colab_run.ipynb)

```python
# Clone repo + install deps
!git clone https://github.com/YOUR_REPO/ampr d:/datn
!pip install -r d:/datn/requirements.txt

# Download raw data from DeepFRI sources
!python d:/datn/scripts/01_download_data.py --output data/
```

### Step 2: Precompute Embeddings (one-time, ~2-4 hours)

```python
# ProteinBERT + ProstT5 + BioBERT + Node2Vec
!python d:/datn/scripts/02_precompute_embeddings.py --config d:/datn/configs/mf.yaml
```

### Step 3: Create TFRecords (no contact maps)

```python
# Convert FASTA + GO labels to TFRecord
!python d:/datn/scripts/seq2tfrecord.py \
  --annot data/nrPDB-GO_annot.tsv \
  --fasta data/nrPDB-GO_sequences.fasta \
  --split data/nrPDB-GO_train.txt \
  --out_prefix data/tfrecords/GO_train \
  --num_shards 10 --num_threads 2
```

### Step 4: Train (loop for MF, BP, CC)

```python
# Train MF model
!python d:/datn/main.py --config d:/datn/configs/mf.yaml

# Outputs:
#   - checkpoints/mf/best.pt
#   - logs/mf_train.log
#   - results/mf_predictions.tsv
```

## Key Decisions

| Decision | Reason |
|---|---|
| ProteinBERT instead of ESM-2 | Lighter (1024d vs 1280d), runs on T4 |
| ProstT5 instead of 3D structure | No PDB files needed; 3Di tokens sufficient |
| Precomputed embeddings | Avoid re-running expensive models each epoch |
| 512d projection for all modalities | Fair fusion; weighted sum requires same dimension |
| BioBERT GO + Linear classifier | Ablation study: compare zero-shot vs supervised |
| DAG loss with True Path Rule | Enforce biological consistency in predictions |
| 3 separate models (MF/BP/CC) | Standard CAFA evaluation; easier to analyze |

## Testing & Verification

**Unit tests** (optional but recommended):
- `tests/test_data.py` — load_GO_annot, seq2onehot
- `tests/test_models.py` — forward pass on synthetic batch
- `tests/test_metrics.py` — Fmax computation

**Integration test:**
1. Train on tiny subset (100 proteins, 10 epochs) → ~5 min on Colab T4
2. Verify loss decreases, alpha weights change
3. Verify metrics (Fmax, AUPRC) compute without error

## Dependencies & Versions

```
Python 3.12.x
torch==2.3.1
transformers==4.41.2
dgl==2.1.0
obonet==1.0.0
tensorflow==2.16.1
numpy==1.26.4
scikit-learn==1.5.0
pyyaml==6.0.1
tqdm==4.66.4
```

All pinned for reproducibility. No local testing — verify on Colab.

## Logging Format

Every script logs to stdout + file:

```
[ANNOT] Loaded 12312 proteins
[FASTA] Loaded 12312 sequences (min 60, max 1000 AA)
[SPLIT] train.txt → 9876 proteins
[EMBED] ProteinBERT batch 1/48: 256 sequences → 256×1024 tensor
[MODEL] AMPRModel (512d hidden, 3 modalities, 489 MF terms)
[TRAIN] Epoch 1/50: loss=0.342 (bce=0.289 dag=0.053)
        α=[0.612, 0.251, 0.137] | val Fmax=0.412
[EVAL] Test Fmax=0.418, Smin=6.542, AUPRC=0.401
```

## Notes for Implementation

1. **No test() function** — but input/output shapes logged clearly at each step
2. **Config flexibility** — can switch classifier (linear ↔ biobert) without code change
3. **Reproducibility** — `np.random.seed()` + `torch.manual_seed()` pinned in config
4. **Hardware** — auto-detects GPU; falls back to CPU (slow but works)
5. **Time budget** — precompute: 2-4h (one-time), train 1 model: 1-2h (on T4)

---

**Last updated:** 2026-04-23  
**Maintained by:** Claude Code (brainstorming + planning)
