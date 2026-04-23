# AMPR: Adaptive Multimodal Protein Representation — Design Spec

**Date:** 2026-04-23  
**Author:** Nguyen Viet Hung  
**Thesis:** Dự đoán chức năng protein dựa trên Deep Learning

---

## Context

Protein function prediction là bài toán cốt lõi trong computational biology. Các SOTA hiện tại (DeepFRI, DeepGOPlus) gặp 3 vấn đề chính: (1) missing modality — không phải protein nào cũng có đủ cấu trúc 3D và PPI data, (2) long-tail class imbalance trong Gene Ontology với >27,000 terms, (3) vi phạm True Path Rule của GO DAG hierarchy trong predictions.

AMPR giải quyết bằng: Adaptive Gating (tự động weight các modality có sẵn), ProstT5 thay cho 3D structure (không cần PDB coordinates), BioBERT-encoded GO terms làm zero-shot classifier, và custom DAG-constrained loss.

**Môi trường thực thi:** Tất cả code chạy trên Google Colab (Python 3.12.13, T4 15GB VRAM) hoặc Kaggle (Python 3.12.12, 2x T4). Không có local execution.

---

## Architecture Overview

3 model riêng biệt cho 3 GO branches: MF, BP, CC. Dùng chung code, khác config YAML.

```
Input per protein:
  x_seq  (1024,) ── Linear(1024→512) + ReLU + LayerNorm ──→ h_seq  (512,)
  x_3di  (1024,) ── Linear(1024→512) + ReLU + LayerNorm ──→ h_3di  (512,)
  x_ppi  (128,)  ── Linear(128→512)  + ReLU + LayerNorm ──→ h_ppi  (512,)
                          ↓ Modality Dropout (train only)
                   [h_seq, h_3di, h_ppi] → (1536,)
                          ↓ Gating MLP
                   Linear(1536→128) + ReLU + Linear(128→3) + Softmax
                          ↓ α = [α_seq, α_3di, α_ppi]
                   Z = α_seq·h_seq + α_3di·h_3di + α_ppi·h_ppi  (512,)
                          ↓
             ┌────────────┴────────────┐
             │                         │
   Linear(512→C)              W_go.T @ Z  (BioBERT, frozen)
   (trainable head)           (zero-shot head)
             └────────────┬────────────┘
                    logits (C,) → sigmoid → predictions
```

**Embedding dimensions — justification:**
- Tất cả modalities được project về cùng 512d trước fusion vì weighted sum yêu cầu cùng chiều, và projection là learnable transformation (không phải padding)
- PPI upsampled 128→512d: Node2Vec graph topology signal ít chiều hơn sequence/structure nhưng cần "fair playing field" trong gating
- GO terms: BioBERT 768d → 512d via Linear projection. 27,000+ terms có intrinsic dimensionality thấp do cấu trúc DAG phân cấp và definition text ngắn (~20-80 tokens)

---

## Project Structure

```
ampr/                           # Python package
├── data/
│   ├── dataset.py              # AMPRDataset: load .npy, return tensors
│   └── go_graph.py             # parse go-basic.obo → DAG adjacency matrix
├── embeddings/
│   ├── protein_bert.py         # ProteinBERT → (N, 1024) .npy
│   ├── prost_t5.py             # ProstT5 → (N, 1024) .npy
│   ├── ppi_node2vec.py         # DGL ppi.bin → Node2Vec → (N, 128) .npy
│   └── biobert_go.py           # BioBERT GO definitions → (C, 768) .npy
├── models/
│   ├── projections.py          # per-modality Linear+ReLU+LayerNorm
│   ├── fusion.py               # GatingNetwork + weighted sum
│   ├── classifier.py           # LinearHead + BioBERTHead
│   └── ampr.py                 # AMPRModel assembles all components
├── training/
│   ├── loss.py                 # AMPRLoss: BCE + λ*DAG_loss
│   └── trainer.py              # Trainer: loop, logging, checkpoint, DataParallel
└── evaluation/
    └── metrics.py              # Fmax, Smin, AUPRC

configs/
├── mf.yaml
├── bp.yaml
└── cc.yaml

scripts/
├── 01_download_data.py         # wget sequences, GO, PPI — run in notebook
├── 02_preprocess_go.py         # create_nrPDB_GO_annot → .fasta + .tsv
├── 03_precompute_embeddings.py # ProteinBERT + ProstT5 + BioBERT → .npy
├── 04_run_node2vec.py          # DGL graph → Node2Vec → ppi_embeddings.npy
└── 05_build_dag_matrix.py      # go-basic.obo → dag_matrix_{mf,bp,cc}.npy

notebooks/
├── colab_run.ipynb             # Colab entry point (T4 single GPU)
└── kaggle_run.ipynb            # Kaggle entry point (2x T4, DataParallel)

main.py                         # python main.py --config configs/mf.yaml
requirements.txt                # pinned for Python 3.12
```

---

## Data Pipeline

### Raw Data Sources

| File | Source | Purpose |
|---|---|---|
| `pdb_seqres.txt.gz` | PDB FTP | Protein sequences |
| `pdb_chain_go.tsv.gz` | SIFTS | GO annotations |
| `bc-95.out` | DeepFRI repo | Sequence clustering 95% |
| `go-basic.obo` | OBO Foundry | GO DAG hierarchy |
| `ppi_{mf,bp,cc}.bin` | DeepGO repo | PPI DGL graphs |

### Precomputed .npy Files (shared across MF/BP/CC)

| File | Shape | Notes |
|---|---|---|
| `seq_embeddings.npy` | (N, 1024) | ProteinBERT [CLS] token |
| `struct_embeddings.npy` | (N, 1024) | ProstT5 mean pooling |
| `ppi_embeddings.npy` | (N, 128) | Node2Vec; zero-vector if missing |
| `labels_mf.npy` | (N, C_mf) | binary, float32 |
| `labels_bp.npy` | (N, C_bp) | binary, float32 |
| `labels_cc.npy` | (N, C_cc) | binary, float32 |
| `dag_matrix_mf.npy` | (C_mf, C_mf) | A[i,j]=1 if j is parent of i |
| `dag_matrix_bp.npy` | (C_bp, C_bp) | |
| `dag_matrix_cc.npy` | (C_cc, C_cc) | |
| `go_emb_mf.npy` | (C_mf, 768) | BioBERT GO definitions |
| `go_emb_bp.npy` | (C_bp, 768) | |
| `go_emb_cc.npy` | (C_cc, 768) | |
| `splits.json` | — | train/valid/test protein IDs |

### Logging Format (mỗi script phải log)

```
[DATA] Loading seq_embeddings.npy    → shape (12312, 1024), dtype float32
[DATA] Missing PPI: 1823/12312 proteins → zero-vector fallback
[DATA] Train/Valid/Test split: 9876 / 1234 / 1202
[EMBED] ProteinBERT processing batch 1/48 (256 sequences)
[EMBED] Saved seq_embeddings.npy (12312, 1024) — 49.2 MB
```

---

## Loss Function

```
L_total = λ1 * L_BCE + λ2 * L_DAG

L_BCE = BCEWithLogitsLoss(logits, labels)

L_DAG = mean(max(0, P_child - P_parent)²)
      where P = sigmoid(logits)
      child-parent pairs from DAG matrix A
```

Default: `λ1=1.0, λ2=0.5`

---

## Config YAML (example: MF)

```yaml
branch: MF
n_terms: 489

data:
  seq_emb: dataset/seq_embeddings.npy
  struct_emb: dataset/struct_embeddings.npy
  ppi_emb: dataset/ppi_embeddings.npy
  labels: dataset/labels_mf.npy
  dag_matrix: dataset/dag_matrix_mf.npy
  go_emb: dataset/go_emb_mf.npy
  splits: dataset/splits.json

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
  device: auto          # auto-detect: cuda / cpu

output:
  checkpoint_dir: checkpoints/mf/
  log_file: logs/mf_train.log
```

---

## Training Log Format

```
[EPOCH 01/50] lr=1.0e-3 | loss=0.3421 (bce=0.2891 dag=0.1060)
              α_seq=0.612 α_3di=0.251 α_ppi=0.137
              valid Fmax=0.412 | AUPRC=0.389
[EPOCH 02/50] lr=9.8e-4 | loss=0.3187 (bce=0.2701 dag=0.0972)
              α_seq=0.589 α_3di=0.278 α_ppi=0.133
              valid Fmax=0.431 | AUPRC=0.401
```

---

## Hardware & Compatibility

| Platform | Python | GPU | DataParallel |
|---|---|---|---|
| Google Colab | 3.12.13 | T4 15GB | No |
| Kaggle | 3.12.12 | 2x T4 15GB | Yes (auto-detect) |

**Key pinned versions (requirements.txt):**
```
torch==2.3.1
transformers==4.41.2
dgl==2.1.0
obonet==1.0.0
numpy==1.26.4
scikit-learn==1.5.0
pyyaml==6.0.1
tqdm==4.66.4
```

DataParallel logic trong `trainer.py`:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    log("[TRAIN] Using DataParallel on", torch.cuda.device_count(), "GPUs")
```

---

## Evaluation Metrics

- **Fmax**: primary metric, threshold-optimized F1 over all GO terms
- **Smin**: semantic distance metric (uses GO hierarchy)
- **AUPRC**: area under precision-recall curve

Computed per branch (MF/BP/CC) và reported riêng biệt.

---

## Ablation Study (thesis comparison table)

| Model Variant | Classifier | Modalities |
|---|---|---|
| AMPR-Linear | Linear head | seq + 3di + ppi |
| AMPR-BioBERT | BioBERT weights | seq + 3di + ppi |
| AMPR-SeqOnly | Linear head | seq only |
| AMPR-NoPPI | Linear head | seq + 3di |

Alpha weights `[α_seq, α_3di, α_ppi]` logged mỗi epoch → visualize modality contribution qua training.
