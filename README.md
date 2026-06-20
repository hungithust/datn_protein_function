# AMPR: Adaptive Multimodal Protein Representation

Deep learning-based protein function prediction with Adaptive Multimodal Representation.

**Thesis:** Dự đoán chức năng protein dựa trên Deep Learning
**Student:** Nguyen Viet Hung (20224998)

AMPR predicts Gene Ontology (GO) terms for proteins by fusing three modalities —
**ESM-2** sequence embeddings, a **contact-map GNN** (structure), and **DeepGO PPI**
embeddings — with a cross-modal adaptive-gating fusion, then enforces the GO hierarchy
(True Path Rule) via a DAG loss. Separate models per GO branch: **MF** (489 terms),
**BP** (1943), **CC** (320). At inference, model probabilities are DAG-propagated and
ensembled with a **DIAMOND** sequence-homology baseline.

> **Pipeline = V3** (`AMPRModelV3`, entry `main.py → _run_v3 / _eval_v3`). The older
> ProteinBERT/ProstT5/Node2Vec/TFRecord pipeline is deprecated — see
> [docs/HANH_TRINH_THU_NGHIEM.md](docs/HANH_TRINH_THU_NGHIEM.md) for the full migration story.

## Results (held-out PDBch test, 3-seed ensemble + DIAMOND, Fmax)

| Branch | LT_30 (novel) | **LT_95 (full)** | DeepFRI (LT_95) | HEAL SOTA (LT_95) |
|---|---|---|---|---|
| **MF** | 0.524 | **0.654** | 0.626 ✅ beats | 0.749 |
| **BP** | — | **0.539** | 0.540 ≈ ties | 0.594 |
| **CC** | 0.515 | **0.566** | 0.612 ❌ trails | 0.687 |

- **MF ensemble beats DeepFRI at every identity bin**; BP essentially ties; CC still
  trails (limited DIAMOND homology coverage at low identity). All three remain below HEAL.
- The official model is **PDB-30K, no SWISS-MODEL pretrain** (large-scale homology-model
  pretrain was tested and *hurt* — clean negative result, see [docs/PHASE_V6_RESULTS.md](docs/PHASE_V6_RESULTS.md)).
- Backbone is a separate axis: **650M for MF/CC** (≈/> 3B), **3B for BP** (the hardest,
  longest-tail branch still benefits from capacity).
- Full identity curve + all metrics: [docs/RESULTS_DATA.md](docs/RESULTS_DATA.md) ·
  baseline provenance (HEAL supp. Tables S3.1/S3.2): [docs/REPORT_v4_summary.md](docs/REPORT_v4_summary.md).

> ⚠️ Earlier drafts claimed "BP/CC beat DeepFRI" — that was an artifact of a **wrong
> DeepFRI baseline** later corrected from the HEAL supplementary. Do not cite the old numbers.

## Inference — predict GO terms for your own proteins

Full-modality inference needs, per protein, a **sequence** (FASTA) and a **structure**
(a PDB file at `<pdb_dir>/<protein_id>.pdb`; AlphaFold models work). ESM-2 requires a GPU.

```bash
pip install -r requirements.txt          # torch, transformers, h5py, biopython, pyyaml

# 1. put sequences in proteins.fasta and structures in pdbs/<id>.pdb
# 2. download a trained checkpoint into checkpoints/<branch>_v3_esm3b/best.pt
#    (see Releases / Google Drive link), then:

python scripts/predict.py --branch mf \
  --fasta proteins.fasta --pdb_dir pdbs/ \
  --out predictions_mf.tsv --threshold 0.3
```

Output `predictions_mf.tsv`:

```
protein_id    go_term     score
P12345        GO:0016787  0.91
P12345        GO:0003824  0.88
...
```

- `--branch {mf,bp,cc}` — which GO sub-ontology to predict.
- `--threshold 0.3` — keep terms above this score; or `--topk 20` for the top-k per protein.
- PPI is unavailable for novel proteins, so that modality is masked off automatically.
- Scores are DAG-propagated (parent terms ≥ child terms).

A runnable walkthrough is in [notebooks/inference_demo.ipynb](notebooks/inference_demo.ipynb).

## Architecture

```
ESM-2 residue (650M=1280d / 3B=2560d) ─→ Transformer encoder ─┐
                                                              │
Contact map ─→ GCN (node 256d, 3 layers, 10Å threshold) ──────┼→ CrossModal
                                                              │   Fusion (512d,
DeepGO PPI (256d) + availability mask ────────────────────────┘   adaptive gating)
                                                                        │
GO embedding (SapBERT text + GO-graph, combined 896d) ─→ classifier ←───┘
                                                          (linear + GO-emb head = "both")
                                                                        │
                                                                  per-term logits
```

- **Backbones frozen** — only fusion + classifier are trained (cheap, fits one GPU).
- **Loss:** `L = cls + λ·DAG_loss` (λ=0.5). `cls` = ASL with the *combined* GO embedding,
  or **BCE + pos_weight** for long-tail branches (BP) where ASL dead-gradient collapses.
- **GO embedding must be combined** (text+graph, 896d) — text-only collapses.

## Quick Start (8×H200 server / Kaggle)

The project trains on a dedicated 8×H200 server (NVIDIA Open Hackathon). All embeddings
are **precomputed once**, then only the small fusion+classifier is trained.

### 1. Setup + pull data

```bash
bash scripts/server_setup.sh          # env, dirs, deps
bash scripts/pull_kaggle_data.sh      # FASTA, annotations, splits, obo
python scripts/verify_inputs.py       # gate: confirm all artifacts present & aligned
```

### 2. Build artifacts (one-time)

```bash
# Labels, splits, DAG matrices from DeepFRI annotation + go-basic.obo
python scripts/build_labels_from_annot.py
python scripts/build_splits_from_deepfri.py
python scripts/build_dag_from_obo.py

# Sequence embeddings (ESM-2 residue-level, sharded) — pick 650M or 3B
bash scripts/launch_esm3b_precompute.sh        # → data/embeddings/esm2_*_residue.h5

# PPI (DeepGO) + availability mask
python scripts/build_ppi_from_deepgo.py        # → ppi_deepgo.npy (+ _mask.npy)

# GO embedding: SapBERT text + GO-graph, then combine
python scripts/precompute_go_text.py
python scripts/precompute_go_graph.py
python scripts/build_go_combined.py            # → go_emb_<branch>_v2.npy (896d)

# Contact maps (test set) + DIAMOND homology baseline
python scripts/precompute_cmap_test.py         # → data/contact_maps/cmap_all.h5
bash scripts/run_diamond.sh                    # → data/diamond/diamond_results_<branch>.tsv
```

### 3. Train (3 seeds per branch for the ensemble)

```bash
for seed in 42 123 2024; do
  python main.py --config configs/mf_v3_esm3b.yaml --seed $seed
done
# repeat with bp_v3_esm3b.yaml / cc_v3_esm3b.yaml
```

Outputs per run: `checkpoints/<branch>_v3_esm3b/best.pt`, `logs/<branch>_*_train.log`.

### 4. Evaluate (3-seed ensemble + DIAMOND, all identity bins)

```bash
cks="checkpoints/mf_v3_esm3b_s42/best.pt checkpoints/mf_v3_esm3b_s123/best.pt checkpoints/mf_v3_esm3b_s2024/best.pt"
for split in test test_LT_30 test_LT_40 test_LT_50 test_LT_70 test_LT_95; do
  python scripts/ensemble_eval.py --config configs/mf_v3_esm3b.yaml \
    --checkpoints $cks --split $split
done
```

Single-model eval: `python main.py --config <cfg> --eval-only --checkpoint <best.pt>`.

## Project Structure

```
ampr/
├── data/          # AMPRDatasetV3 (residue h5 + cmap + ppi + labels)
├── embeddings/    # precompute helpers
├── models/        # AMPRModelV3: GCN, PPI head, CrossModalFusion, classifier
├── training/      # loss (ASL/BCE + DAG), trainer, contrastive (deprecated)
└── evaluation/    # Fmax, Smin, AUPRC; stratified by identity bin

configs/           # <branch>_v3_esm3b.yaml (official), v4/v5/v6 experiments
scripts/           # precompute, build, ensemble_eval, predict, diamond
main.py            # entry: _run_v3 (train) / _eval_v3 (eval)
docs/              # specs, plans, results, experiment journey
```

## Documentation

- [docs/HANH_TRINH_THU_NGHIEM.md](docs/HANH_TRINH_THU_NGHIEM.md) — **experiment journey & lessons** (start here)
- [docs/PHASE_V6_RESULTS.md](docs/PHASE_V6_RESULTS.md) — SWISS-MODEL data-scaling (negative result)
- [docs/RESULTS_DATA.md](docs/RESULTS_DATA.md) — full per-bin metric appendix
- [AMPR_WORKFLOW.md](AMPR_WORKFLOW.md) — step-by-step execution guide
- [CLAUDE.md](CLAUDE.md) — project context for Claude Code

## Evaluation Metrics

- **Fmax** — protein-centric optimal F1 (primary; reported per identity bin LT_30..LT_95)
- **Smin** — semantic-distance metric (information-content weighted)
- **AUPRC** — area under precision-recall curve (macro)

Computed per branch (MF/BP/CC) independently, stratified by max train-test sequence identity.

## Dependencies

```
torch==2.3.1
transformers==4.41.2     # ESM-2, SapBERT
dgl==2.1.0               # PPI / GO-graph
obonet==1.0.0           # go-basic.obo → DAG
h5py                    # residue embeddings + contact maps
numpy==1.26.4
```

Python 3.12.x. Backbones (ESM-2, SapBERT) are frozen — no fine-tuning required.

## Troubleshooting

| Issue | Solution |
|---|---|
| Fmax stuck at ~0.0209 | Dead-gradient collapse — use **BCE + pos_weight** (not ASL text-only); see journey §1 |
| OOM on 2560d (3B) | Lower `batch_size` to 64 (already default for esm3b) |
| `verify_inputs.py` fails | An artifact is missing/misaligned — regenerate, don't shim |
| GPU 0-1 busy (node-07) | Use `CUDA_VISIBLE_DEVICES=2..7` |
| val ↑ but test flat | Expected — the core problem is the val→test gap (~0.20), not val |

## Author

Nguyen Viet Hung (nguyenviethungsoicthust@gmail.com)
