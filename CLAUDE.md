# AMPR: Adaptive Multimodal Protein Representation

**Thesis Topic:** Deep learning-based protein function prediction with Adaptive Multimodal Representation  
**Student:** Nguyen Viet Hung (20224998)  
**Execution Environment:** 8×H200 server (NVIDIA Open Hackathon 2026); legacy runs on Kaggle 2×T4  
**Status:** V3 pipeline trained + evaluated → thesis writing. Official model = PDB-30K (no SWISS-MODEL pretrain).

> **Current pipeline = V3** (`AMPRModelV3`, entry `main.py → _run_v3 / _eval_v3`). The
> ProteinBERT/ProstT5/Node2Vec/TFRecord design below the "Key Decisions" line is the
> **original brainstorm and is superseded** — see the migration story in
> [docs/HANH_TRINH_THU_NGHIEM.md](docs/HANH_TRINH_THU_NGHIEM.md). When in doubt, trust the
> configs (`configs/*_v3_esm3b.yaml`) and scripts over older prose.

## Windows Environment (IMPORTANT for subagents)

- **Python env:** `python` = `D:\anaconda3\python.exe` (Anaconda). Always use `python -m pytest`, never bare `pytest` (bare pytest points to wrong Python312 env and will fail silently).
- **Shell:** Use PowerShell tool for running pytest/python commands. Bash tool has a `rtk` PreToolUse hook that rewrites `pytest` → `rtk pytest` which may swallow output.
- **pytest config:** `pytest.ini` is at repo root — `pytest tests/` works from project root.
- **obonet:** Must be installed in Anaconda env: `pip install obonet==1.0.0` if missing.

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

## Architecture (V3 — current)

```
ESM-2 residue (650M=1280d / 3B=2560d) ─→ Transformer encoder ─┐
                                                              │
Contact map ─→ GCN (node 256d, 3 layers, 10Å threshold) ──────┼→ CrossModalFusion
                                                              │   (512d, adaptive
DeepGO PPI (256d) + availability mask ────────────────────────┘    gating → α weights)
                                                                        │
GO embedding (SapBERT text + GO-graph, combined 896d) ─→ classifier ←───┘
                                                          (classifier="both":
                                                           linear + GO-emb head)
                                                                        │
                                                                  per-term logits
```

- **Backbones frozen** (ESM-2, SapBERT) — train only fusion + classifier.
- **Fusion:** cross-modal attention + adaptive gating; PPI masked off when unavailable.
- **Loss:** `L = cls + λ·DAG_loss` (λ=0.5). `cls` = **ASL** with the *combined* GO embedding
  (MF/CC), or **BCE + per-class pos_weight** for BP (1943-term long tail — ASL dead-gradients).
- **GO embedding must be combined `_v2` (896d)** — text-only collapses (Fmax 0.0209).
- **Inference:** DAG-propagate probs → DIAMOND homology ensemble (α=0.6) → 3-seed ensemble.

## Code Organization

```
ampr/                          # Python package
├── data/                       # AMPRDatasetV3 (residue h5 + cmap + ppi + labels)
├── embeddings/                 # precompute helpers
├── models/                     # AMPRModelV3: GCN, PPI head, CrossModalFusion, classifier
├── training/                   # loss (ASL/BCE + DAG), trainer; contrastive (deprecated)
└── evaluation/                 # Fmax, Smin, AUPRC; stratified by LT_* identity bin

configs/                        # per-branch YAML
├── {mf,bp,cc}_v3_esm3b.yaml    # OFFICIAL configs
├── *_v4_supcon.yaml            # Module A contrastive (negative result)
├── *_v5_drop04*.yaml           # dropout-0.4 anti-overfit
└── *_v6_*.yaml                 # SWISS-MODEL expansion (negative result)

scripts/                        # build + precompute + eval (see AMPR_WORKFLOW.md)
├── precompute_esm2_residue.py / launch_esm3b_precompute.sh   # ESM-2 → h5
├── build_ppi_from_deepgo.py                                  # PPI + mask
├── precompute_go_text.py + precompute_go_graph.py + build_go_combined.py  # GO emb
├── build_dag_from_obo.py / build_labels_from_annot.py / build_splits_from_deepfri.py
├── precompute_cmap_test.py / run_diamond.sh                  # structure + homology
├── ensemble_eval.py                                          # 3-seed + DIAMOND eval
├── verify_inputs.py / verify_label_parity.py                 # GATES
└── predict.py                                                # inference on new proteins

main.py                         # CLI: python main.py --config configs/mf_v3_esm3b.yaml --seed 42
                                #      --eval-only / --checkpoint / --test-split / --init-from
```

> The old `seq2tfrecord.py`, `01_download_data.py`, `02_precompute_embeddings.py`,
> `03_build_dag_matrix.py`, `04_run_node2vec.py` belong to the deprecated TFRecord pipeline.

## Principles

1. **Module clarity** — each subpackage has one clear responsibility
2. **Config-driven** — YAML controls all hyperparameters, no hardcoding
3. **Logging explicit** — every step logs shapes, counts, α weights, val Fmax_dag
4. **Precompute once** — all embeddings cached to h5/npy; train only fusion+classifier
5. **Regenerate, don't shim** — if `verify_inputs.py` flags a stale artifact, rebuild it
6. **Server-first** — runs on 8×H200 via SSH+tmux (GPUs 2-7 on node-07)

## Running (see AMPR_WORKFLOW.md for the full guide)

```bash
# 0. setup + gate
bash scripts/server_setup.sh && bash scripts/pull_kaggle_data.sh && python scripts/verify_inputs.py

# 1. build artifacts (one-time) — labels, DAG, ESM-2 h5, PPI, combined GO emb, cmap, DIAMOND
#    (individual scripts listed under Code Organization)

# 2. train 3 seeds per branch
for seed in 42 123 2024; do python main.py --config configs/mf_v3_esm3b.yaml --seed $seed; done

# 3. ensemble eval across identity bins
python scripts/ensemble_eval.py --config configs/mf_v3_esm3b.yaml --checkpoints <s42 s123 s2024> --split test_LT_95
```

## Key Decisions

| Decision | Reason |
|---|---|
| ESM-2 (650M MF/CC, 3B BP), frozen | Strong PLM; 650M ≈/> 3B except hardest branch (BP keeps 3B) |
| Contact-map GCN for structure | Real experimental structure signal; homology-model cmaps hurt (v6 negative) |
| DeepGO PPI + availability mask | 3rd modality; gate masks it off for proteins without interactions |
| Combined GO embedding (SapBERT+graph, 896d) | Text-only collapses under ASL; combined is stable |
| BCE+pos_weight for BP | ASL dead-gradients on 1943-term long tail at every stage |
| DAG loss (True Path Rule) | Enforce parent≥child GO consistency |
| 3-seed ensemble + DIAMOND | +0.02–0.035 Fmax, no arch change; standard CAFA-style eval |
| PDB-30K, no SWISS-MODEL pretrain | Large-scale homology-model pretrain hurt (clean v6 negative) |
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
transformers==4.41.2     # ESM-2, SapBERT
dgl==2.1.0               # PPI / GO-graph
obonet==1.0.0           # go-basic.obo → DAG
h5py                    # residue embeddings + contact maps
numpy==1.26.4
scikit-learn==1.5.0
pyyaml==6.0.1
tqdm==4.66.4
```

Backbones frozen (no fine-tuning). `tensorflow` was only for the deprecated TFRecord pipeline.

## Logging Format

Every script logs to stdout + file:

```
[INPUTS] verify_inputs: labels/cmap/ppi/go_emb aligned to protein_order.json ✓
[MODEL] AMPRModelV3 (seq 2560d, GCN 256d, PPI 256d, fusion 512d, 489 MF terms)
[TRAIN] epoch 01/50: loss=0.342 (cls=0.289 dag=0.053)
        α_seq=0.61 α_struct=0.25 α_ppi=0.14 | val Fmax_dag=0.412
[BEST]  epoch 29 → checkpoints/mf_v3_esm3b/best.pt (val Fmax_dag=0.760)
[EVAL]  LT_95 ens+DIAMOND: Fmax=0.654 Smin=0.619 AUPRC=0.563
```

If Fmax pins at **0.0209** with probs≈0 → dead-gradient collapse: use BCE+pos_weight,
ensure the GO embedding is the combined `_v2`, check grad_clip.

## Current Results (3-seed ensemble + DIAMOND, full test LT_95, Fmax)

| Branch | AMPR | DeepFRI | HEAL (SOTA) |
|---|---|---|---|
| MF | **0.654** (beats DeepFRI all bins) | 0.626 | 0.749 |
| BP | **0.539** (≈ ties) | 0.540 | 0.594 |
| CC | **0.566** (trails) | 0.612 | 0.687 |

Baselines from HEAL supplementary Tables S3.1/S3.2 (same PDBch test, same LT_* bins). The old
`results/deepfri_baseline.json` was wrong — do not cite it. Full appendix:
[docs/RESULTS_DATA.md](docs/RESULTS_DATA.md); journey + lessons:
[docs/HANH_TRINH_THU_NGHIEM.md](docs/HANH_TRINH_THU_NGHIEM.md).

## Notes for Implementation

1. **Eval path** — `main.py --eval-only` (single) or `scripts/ensemble_eval.py` (3-seed + DIAMOND)
2. **Config flexibility** — switch `classifier` / `loss_type` / backbone via YAML, no code change
3. **Reproducibility** — `--seed {42,123,2024}`; seeds pinned in config for the ensemble
4. **Gates first** — run `verify_inputs.py` / `verify_label_parity.py` before any training
5. **Time budget** — precompute (one-time) hours; train 1 model ~1-2h on one H200

---

**Last updated:** 2026-06-17  
**Maintained by:** Claude Code (planning + implementation)
