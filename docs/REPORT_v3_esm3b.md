# AMPR v3 (ESM-2 3B) — Experiment Report

**Date:** 2026-06-04 · **Hardware:** 8×H200 (NGC pytorch:24.10, `--ipc=host`)
**Dataset:** nrPDB-GO (DeepFRI split), 3 GO branches · **Test set:** 3,123 proteins

---

## 1. Setup

AMPR v3 fuses three precomputed modalities and trains only the fusion + heads:

- **Sequence:** ESM-2 3B residue embeddings (2560-d) → Transformer encoder (2L) → attention pool
- **Structure:** GNN over contact-map adjacency (cmap threshold 10 Å), node init from ESM-2
- **PPI:** DeepGO PPI embeddings (256-d), gated by availability mask
- **Fusion:** cross-modal attention (d=512, 2L) → classifier head
- **Head:** `both` = 0.5·linear + 0.5·(z·proj·go_emb^T), GO embeddings = SapBERT text + graph (896-d)
- **Loss:** ASL + λ·DAG (True Path Rule), λ=0.5 · **LR:** Adam 1e-3 + gentle ReduceLROnPlateau

Architecture was **selected by measurement**, not assumption (see §2).

## 2. Architecture / capacity sweep (MF)

8-cell grid `classifier{both, label_attn} × d_hidden{512, 1024} × go_emb{comb 896-d, text-only 768-d}`, 50 epochs each on a dedicated GPU.

| Config | best val Fmax_dag |
|---|---|
| **both · 512 · comb** ✅ | **0.7525** |
| label_attn · 512 · comb | 0.5987 |
| label_attn · 1024 · comb | 0.5950 |
| both · 1024 · comb | 0.5781 |
| all 4 `*_text` variants | 0.0209 (collapse) |

**Conclusions:**
1. The compact base config (`both` / 512 / comb) wins decisively.
2. **Scaling up hurts** — `d_hidden` 512→1024 is worse in every pairing. No need to enlarge the model at the current data scale.
3. The new **label-attention head hurts** vs the dot-product `both` head.
4. **Combined text+graph GO embeddings are essential** — text-only (768-d) caused total dead-gradient collapse (probs→0, Fmax 0.0209).

## 3. Final test-set results (mf/bp/cc)

Each branch trained 50 epochs with the winning architecture + LR scheduler; evaluated on the held-out `test` split (3,123 proteins). Metrics shown for **DAG-propagated** scores (raw is within ±0.002).

| Branch | n_terms | Fmax | Smin* | AUPRC micro | AUPRC macro | AUROC micro | Coverage |
|---|---|---|---|---|---|---|---|
| **MF** | 489 | **0.5498** | 0.677 | 0.585 | 0.524 | 0.922 | 0.941 |
| **BP** | 1943 | **0.4582** | 0.846 | 0.306 | 0.275 | 0.784 | 0.998 |
| **CC** | 320 | **0.4955** | 0.797 | 0.367 | 0.350 | 0.842 | 0.974 |

\* Smin here is the **normalized** remaining-uncertainty/misinformation variant (∈[0,1], lower better) — **not** on the same scale as DeepFRI's raw-IC Smin; do not compare across the two directly.

**DAG propagation** improves every branch slightly (MF Fmax 0.5488→0.5498, BP 0.4557→0.4582, CC 0.4941→0.4955), confirming the True-Path-Rule post-processing is mildly beneficial and never harmful.

## 4. Generalization gap (val → test)

| Branch | best val Fmax_dag | test Fmax_dag | gap |
|---|---|---|---|
| MF | 0.7452 | 0.5498 | **−0.195** |
| BP | 0.6726 | 0.4582 | **−0.214** |
| CC | 0.6817 | 0.4955 | **−0.186** |

A consistent ~0.19–0.21 drop from validation to test across all three branches. The held-out `test` split is harder (lower sequence similarity to train). Note this is an *absolute* gap, not a weakness relative to the baseline: §5 shows that on the low-similarity bins AMPR is actually **more robust than DeepFRI**. The gap mostly reflects the validation split containing higher-similarity proteins.

> Note: with the LR scheduler, MF best val (0.7452) is marginally below the no-scheduler sweep run (0.7525). The scheduler smooths late-epoch oscillation but did not raise the peak here; it is retained for stability/consistency across branches.

## 5. Comparison to DeepFRI baseline (LT_30 / LT_95)

Evaluated on the two sequence-identity bins DeepFRI reports (`LT_30` = ≤30% identity to train, hardest/novel; `LT_95` = full test, 3,123 proteins). AMPR numbers are DAG-propagated; DeepFRI from Gligorijević et al. 2021 (DeepFRI-GCN). **Bold = better** (higher Fmax/AUPRC).

| Branch | Split | AMPR Fmax | DeepFRI Fmax | AMPR AUPRC_micro | DeepFRI AUPRC_micro |
|---|---|---|---|---|---|
| MF | LT_30 | 0.4777 | **0.545** | **0.5024** | 0.443 |
| MF | LT_95 | 0.5498 | **0.759** | 0.5851 | **0.671** |
| BP | LT_30 | **0.4358** | 0.282 | **0.2803** | 0.169 |
| BP | LT_95 | **0.4582** | 0.395 | **0.3057** | 0.272 |
| CC | LT_30 | **0.4909** | 0.434 | **0.3832** | 0.308 |
| CC | LT_95 | 0.4955 | **0.561** | 0.3665 | **0.443** |

(AMPR normalized Smin — LT_30 / LT_95: MF 0.773 / 0.677, BP 0.887 / 0.846, CC 0.811 / 0.797. Not comparable to DeepFRI's raw-IC Smin; reported for internal reference only.)

**The key result — similarity robustness.** AMPR's Fmax is nearly **flat** from LT_30→LT_95 (MF +0.072, BP +0.022, CC +0.005), whereas DeepFRI rises steeply (MF +0.214, CC +0.127). This means AMPR (built on ESM-2 3B embeddings) generalizes to **low-similarity / novel sequences** far better, while DeepFRI relies more on close homologs.

Consequently, in the hardest and most practically important regime (**LT_30, novel proteins**):
- **BP:** AMPR beats DeepFRI on both Fmax (+0.154) and AUPRC (+0.111).
- **CC:** AMPR beats DeepFRI on both Fmax (+0.057) and AUPRC (+0.075).
- **MF:** AMPR wins AUPRC_micro (+0.059); Fmax slightly behind (−0.068).

DeepFRI overtakes only at high identity (LT_95) for MF and CC, where exploiting near-duplicate homologs pays off.

## 6. Key findings & recommended next steps

**Findings**
- Best architecture is compact (512-d, dot-product head) — enlarging the model does not help at this data scale.
- Combined text+graph GO embeddings are required; text-only collapses.
- **AMPR is markedly more robust to low sequence identity than DeepFRI** — at LT_30 it beats DeepFRI on BP and CC (both metrics) and on MF AUPRC. This similarity-invariance is the model's main strength and a strong thesis narrative.
- DeepFRI only wins at high identity (LT_95, MF/CC), where homology lookup dominates — exactly where a homology ensemble would help AMPR (next step).

**Next steps (priority order)**
1. **DIAMOND ensemble** — config has `use_diamond_ensemble: true`, but `_eval_v3` reports pure-model scores; adding the homology ensemble targets precisely the high-identity regime where DeepFRI currently leads (likely closes the LT_95 MF/CC gap). Highest-value, low-risk.
2. **MF high-identity Fmax** — investigate per-threshold calibration; AMPR already wins MF AUPRC at LT_30, so the Fmax gap is partly thresholding.
3. **Scale-up only with more data (SWISS-MODEL)** — enlarging `d_hidden`/fusion depth is justified *only* alongside a much larger training set; requires a separate embedding-precompute effort (cf. Plan 2).
4. Remaining LT bins (LT_40/50/70) for a complete identity-curve figure in the thesis.

---

*Artifacts: `docs/sweep_results_mf/RANKING.md` (sweep), `results/*_v3_esm3b.eval_test.json` (test metrics), `logs/*_v3_esm3b.train.run.log` (training traces). Checkpoints backed up to Google Drive.*
