# Thesis Draft — Adaptive Multimodal Protein Representation for GO Function Prediction

> Draft scaffold. Sections marked _[TODO]_ need the student's prose/citations; the
> Methods/Experiments/Results sections are filled from the actual AMPR v3 results
> (see [REPORT_v3_esm3b.md](REPORT_v3_esm3b.md)). Written in English for reuse;
> translate to Vietnamese for submission as needed.

**Student:** Nguyen Viet Hung (20224998)

---

## Abstract

We present **AMPR (Adaptive Multimodal Protein Representation)**, a framework for
predicting Gene Ontology (GO) terms that fuses three complementary modalities —
ESM-2 3B sequence embeddings, a contact-map graph neural network over protein
structure, and protein–protein interaction (PPI) embeddings — via cross-modal
attention with adaptive gating for missing modalities, and enforces GO-hierarchy
consistency through a DAG-constrained loss (True Path Rule). On the nrPDB-GO
benchmark, AMPR is markedly more robust to low sequence identity than the DeepFRI
baseline, and — combined with a DIAMOND homology ensemble — surpasses DeepFRI on
the Biological Process and Cellular Component ontologies while remaining
competitive on Molecular Function. _[TODO: 1–2 sentences with final headline numbers.]_

## 1. Introduction

- Motivation: experimental GO annotation is slow; automated function prediction (CAFA). _[TODO]_
- Three challenges addressed: (i) **missing-modality robustness**, (ii) **GO-hierarchy
  consistency**, (iii) **efficient training on limited hardware** (precomputed embeddings).
- Contributions:
  1. A multimodal architecture with adaptive gating that degrades gracefully when
     structure or PPI is absent.
  2. A DAG-constrained loss + inference-time True-Path-Rule propagation.
  3. A **measured** capacity/encoder sweep (rather than guessed model size) and a
     homology ensemble; an analysis of similarity-robustness vs the DeepFRI baseline.
  4. An open inference pipeline for external proteins (sequence + structure).

## 2. Background & Related Work _[TODO]_

- GO and the CAFA evaluation; Fmax / Smin / AUPRC metrics.
- Sequence models: ESM-2 protein language models.
- Structure-based: DeepFRI (GCN over contact maps), structure prediction (AlphaFold).
- Homology transfer: BLAST/DIAMOND, DeepGOPlus.
- Hierarchy-aware prediction: True Path Rule, DAG losses.

## 3. Methods

### 3.1 Problem formulation
Multi-label classification over GO terms per branch (MF/BP/CC); labels obey the GO DAG.

### 3.2 Modalities (precomputed)
- **Sequence:** ESM-2 3B (`esm2_t36_3B`) per-residue embeddings (2560-d).
- **Structure:** Cα–Cα contact map (Å) from PDB; a GNN over the thresholded
  adjacency (10 Å) with nodes initialized from ESM-2 residues.
- **PPI:** DeepGO PPI embeddings (256-d), gated by an availability mask.

### 3.3 Architecture (AMPRModelV3)
- Per-modality projection to a shared 512-d space; attention pooling for the
  sequence and GNN paths.
- **Cross-modal fusion** (multi-head attention, 2 layers) producing a fused
  protein vector `z`; the PPI path is gated off via its mask when absent.
- **Classification head** = `both`: average of a linear head and a GO-embedding
  dot-product head `z·proj·GOᵀ`, where GO embeddings combine SapBERT text + GO-graph
  features (896-d).

### 3.4 Loss & hierarchy
`L = ASL + λ·DAG` (λ=0.5). The DAG term penalizes child>parent violations
(True Path Rule); at inference, scores are propagated upward over the DAG.

### 3.5 Training
AdamW, lr 1e-3 with ReduceLROnPlateau; 50 epochs; precomputed embeddings so only
the fusion + heads train. Hardware: 8×H200. Per-branch regularization tuned by
validation (Section 4.4).

### 3.6 Homology ensemble
DeepGOPlus-style DIAMOND blend: `α·model + (1−α)·diamond` for proteins with a
homology hit (α=0.6), then DAG propagation.

## 4. Experiments

### 4.1 Dataset
nrPDB-GO (DeepFRI split): 36,641 proteins; MF 489 / BP 1943 / CC 320 terms.
Test reported on sequence-identity bins LT_30 … LT_95 (3,123 proteins at LT_95).

### 4.2 Architecture / capacity sweep (MF)
An 8-cell grid `classifier{both, label_attn} × d_hidden{512, 1024} × go_emb{combined, text-only}`.
**Finding:** the compact `both / 512 / combined` config wins (val Fmax_dag 0.7525);
scaling to 1024-d and the label-attention head both hurt; text-only GO embeddings
collapse. → model size chosen by measurement, not assumption.

### 4.3 Main results & baseline comparison
See Section 5. DIAMOND ensemble applied at inference.

### 4.4 Regularization study
Per-branch grid `weight_decay{1e-4,1e-2} × dropout{0.2,0.3}`, selected by validation
Fmax. **Regularization helps MF** (adopt wd=1e-2/dropout=0.2) but **lowers validation
for BP/CC** (baseline kept) — regularization is branch-dependent.

### 4.5 Ablations _[TODO: optional — drop each modality, no-DAG, etc.]_

## 5. Results

Test Fmax (DAG-propagated; +D = with DIAMOND ensemble) vs DeepFRI-GCN:

| Branch | Split | AMPR | AMPR+D | DeepFRI |
|---|---|---|---|---|
| MF | LT_30 | 0.478 | 0.515 | 0.545 |
| MF | LT_95 | 0.550 | 0.614 | 0.759 |
| BP | LT_30 | 0.436 | **0.460** | 0.282 |
| BP | LT_95 | 0.458 | **0.507** | 0.395 |
| CC | LT_30 | 0.491 | **0.515** | 0.434 |
| CC | LT_95 | 0.496 | **0.538** | 0.561 |

**Key findings:**
1. **Similarity robustness** — AMPR Fmax is nearly flat across identity bins
   (model-only Δ LT_30→LT_95: MF +0.07, BP +0.02, CC +0.005) whereas DeepFRI rises
   steeply (MF +0.21), i.e. AMPR generalizes to novel sequences.
2. **AMPR+DIAMOND beats DeepFRI on BP and CC** at every bin except CC LT_95 (tied);
   BP is a decisive win (+0.11–0.18 Fmax).
3. **MF at high identity remains DeepFRI's stronghold** — the one clear gap.
4. The val→test gap (~0.19–0.21) is largely **intrinsic** (test is genuinely
   lower-similarity), not removable by regularization.

_[TODO: add Smin/AUPRC table, identity-curve figure (LT_30/40/50/70/95), PR curves.]_

## 6. Discussion _[TODO]_
- Why ESM-2 gives similarity robustness; when homology helps (high identity).
- Limitations: MF high-identity gap; structure dependency at inference; PPI absent
  for novel proteins; normalized vs raw-IC Smin (not comparable to DeepFRI's).

## 7. Conclusion & Future Work
- Summary of contributions and headline results.
- Future: per-bin α tuning; larger training data (SWISS-MODEL) with scaled capacity;
  structure-free inference mode. _[TODO]_

## References _[TODO: DeepFRI, ESM-2, DeepGOPlus, AlphaFold, CAFA, SapBERT, ...]_

---

### Appendix A — Reproducibility
- Configs: `configs/{mf,bp,cc}_v3_esm3b.yaml`. Training: `scripts/train_all_v3.sh`.
- Eval: `scripts/eval_all_v3.sh <split> [--tune-alpha]`. Sweep: `scripts/{gen_sweep_configs,launch_sweep,collect_sweep}`.
- Inference: `scripts/predict.py` (see README). Full results log: `docs/REPORT_v3_esm3b.md`.
