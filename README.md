# AMPR: Adaptive Multimodal Protein Representation

Deep learning-based protein function prediction with Adaptive Multimodal Representation.

**Thesis:** Dự đoán chức năng protein dựa trên Deep Learning  
**Student:** Nguyen Viet Hung (20224998)  
**Execution:** Google Colab (T4 GPU) or Kaggle (2x T4)

## Quick Start

### 1. Setup (Colab/Kaggle)

```bash
# Install dependencies
!pip install -r requirements.txt

# Clone repo
!git clone https://github.com/YOUR_REPO/ampr /content/ampr
cd /content/ampr
```

### 2. Download Data

```bash
!python scripts/01_download_data.py --output data/
```

Downloads: sequences, GO annotations, PPI data, GO hierarchy.

### 3. Precompute Embeddings (2-4 hours, one-time)

```bash
!python scripts/02_precompute_embeddings.py --config configs/mf.yaml
```

Generates: `data/embeddings/{seq,struct,ppi,go_emb}_*.npy`

### 4. Create TFRecords (no contact maps)

```bash
!python scripts/seq2tfrecord.py \
  --annot data/nrPDB-GO_annot.tsv \
  --fasta data/nrPDB-GO_sequences.fasta \
  --split data/nrPDB-GO_train.txt \
  --out_prefix data/tfrecords/GO_train \
  --num_shards 10 --num_threads 2
```

Repeat for `valid/test` splits.

### 5. Train Models

```bash
# MF branch
!python main.py --config configs/mf.yaml

# BP branch
!python main.py --config configs/bp.yaml

# CC branch
!python main.py --config configs/cc.yaml
```

Outputs:
- `checkpoints/{mf,bp,cc}/best.pt` — trained model
- `logs/{mf,bp,cc}_train.log` — training log
- `results/{mf,bp,cc}_predictions.tsv` — test predictions

## Architecture

```
Sequence (1024d)  ─┐
                   ├→ Project to 512d
Structure (1024d) ─┤
                   │  → Gating Network
PPI (128d)        ─┤  → α_seq, α_3di, α_ppi
                   │
                   ├→ Weighted Sum (512d)
GO Terms (768d) ──┘  → Classifier (Linear + BioBERT)
                        → Predictions
```

**Loss:** BCE + λ·DAG_loss (enforces GO hierarchy)

## Project Structure

```
ampr/
├── data/          # Dataset loading
├── embeddings/    # Embedding computation
├── models/        # Neural architectures
├── training/      # Training loop + loss
└── evaluation/    # Metrics (Fmax, Smin, AUPRC)

configs/           # MF/BP/CC configs
scripts/           # Data processing scripts
main.py            # Entry point
requirements.txt   # Dependencies
CLAUDE.md          # Project notes (for Claude Code)
```

## Key Features

- **Modular design** — clear separation of concerns
- **Config-driven** — YAML controls all hyperparameters
- **Colab-first** — no local execution required
- **Logging explicit** — shapes, counts logged at each step
- **GPU auto-detect** — seamless CPU fallback
- **Multi-GPU support** — DataParallel for Kaggle 2x T4

## Documentation

- [Design Spec](docs/superpowers/specs/2026-04-23-ampr-design.md) — Architecture + data flow
- [Workflow Guide](AMPR_WORKFLOW.md) — Step-by-step execution
- [CLAUDE.md](CLAUDE.md) — Project context for Claude Code

## Evaluation Metrics

- **Fmax** — optimal F1 score (primary metric)
- **Smin** — semantic distance metric
- **AUPRC** — area under precision-recall curve

Computed per branch (MF/BP/CC) independently.

## Dependencies

```
torch==2.3.1
transformers==4.41.2
tensorflow==2.16.1
dgl==2.1.0
obonet==1.0.0
numpy==1.26.4
```

All pinned for Python 3.12.x (Colab/Kaggle).

## Troubleshooting

| Issue | Solution |
|---|---|
| OOM on T4 | Reduce batch_size to 128 |
| Missing sequences | Check FASTA file not corrupted |
| NaN loss | Check label normalization |
| Slow training | Verify GPU usage: `!nvidia-smi` |

## Timeline

| Phase | Time | Hardware |
|---|---|---|
| Download + Precompute | 2-4h | T4 GPU |
| TFRecord creation | 30m | CPU |
| Train 1 model | 1-2h | T4 GPU |
| **Total (3 branches)** | **8-10h** | One Colab session |

## Author

Nguyen Viet Hung (nguyenviethungsoicthust@gmail.com)

## Notes

- This is a **Colab-first project** — no local testing
- All scripts run via `!python` in Jupyter cells
- Input/output shapes logged explicitly for verification
- Reproducible across runs: pinned seeds + versions
