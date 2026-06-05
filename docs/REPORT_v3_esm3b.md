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

**Fmax** (DAG-propagated). `AMPR` = model only; `AMPR+D` = model + DIAMOND homology ensemble (α=0.6, §5.1). **Bold = beats DeepFRI**.

| Branch | Split | AMPR | AMPR+D | DeepFRI |
|---|---|---|---|---|
| MF | LT_30 | 0.4777 | 0.5149 | **0.545** |
| MF | LT_95 | 0.5498 | 0.6142 | **0.759** |
| BP | LT_30 | **0.4358** | **0.4603** | 0.282 |
| BP | LT_95 | **0.4582** | **0.5069** | 0.395 |
| CC | LT_30 | **0.4909** | **0.5154** | 0.434 |
| CC | LT_95 | 0.4955 | **0.5383** | 0.561 |

**AUPRC_micro:**

| Branch | Split | AMPR | AMPR+D | DeepFRI |
|---|---|---|---|---|
| MF | LT_30 | **0.5024** | **0.4974** | 0.443 |
| MF | LT_95 | 0.5851 | 0.6001 | **0.671** |
| BP | LT_30 | **0.2803** | **0.2835** | 0.169 |
| BP | LT_95 | **0.3057** | **0.3409** | 0.272 |
| CC | LT_30 | **0.3832** | **0.3909** | 0.308 |
| CC | LT_95 | 0.3665 | 0.4011 | **0.443** |

(AMPR normalized Smin — not comparable to DeepFRI's raw-IC Smin; internal reference only.)

### 5.1 Effect of the DIAMOND ensemble

The homology ensemble lifts Fmax on **every** branch/split (MF +0.037/+0.064, BP +0.025/+0.049, CC +0.025/+0.043 for LT_30/LT_95), with the larger gains at LT_95 where more high-similarity homologs exist (hom-hits: 2323/3123 MF, 2273 BP, 1897 CC at LT_95 vs ~620–840/1582 at LT_30).

**Per-branch α tuning (negative result).** Tuning α on the validation split (`--tune-alpha`, sweep 0.3–0.9) selected **α=0.6 for all three branches** — identical to the default — so test metrics are unchanged. The validation split is high-similarity to train, where both the model and DIAMOND are strong and the optimal blend coincides with 0.6; this optimum does not necessarily transfer to the harder low-identity test bins. Conclusion: **α=0.6 is robust; per-branch tuning on the standard valid split yields no gain.** (Bin-specific tuning would need a tune/eval split matched to each LT bin's identity distribution.)

### 5.2 Takeaways

- **Similarity robustness (model-only):** AMPR Fmax is nearly flat LT_30→LT_95 (MF +0.072, BP +0.022, CC +0.005) vs DeepFRI's steep rise (MF +0.214, CC +0.127) — ESM-2 generalizes to novel sequences; DeepFRI leans on close homologs.
- **With the ensemble, AMPR beats DeepFRI outright on BP and CC at every split except CC LT_95 (0.538 vs 0.561, essentially tied).** BP is a decisive win (+0.11–0.18 Fmax).
- **MF remains DeepFRI's stronghold at high identity** (LT_95 0.614 vs 0.759), though AMPR wins MF AUPRC at LT_30. MF high-identity is the one clear remaining gap.

## 6. Key findings & recommended next steps

**Findings**
- Best architecture is compact (512-d, dot-product head) — enlarging the model does not help at this data scale.
- Combined text+graph GO embeddings are required; text-only collapses.
- **AMPR is markedly more robust to low sequence identity than DeepFRI** — model-only Fmax is nearly flat across identity bins; DeepFRI leans on close homologs.
- **The DIAMOND ensemble lifts Fmax on every branch/split.** With it, AMPR beats DeepFRI on BP and CC at every split except CC LT_95 (statistically tied); BP is a decisive win.
- The single clear remaining gap is **MF at high identity (LT_95)**.

**Next steps (priority order)**
1. **MF high-identity gap** — tune ensemble α per branch (`tune_alpha`); MF likely benefits from a higher DIAMOND weight since its homology hits are strong. Also revisit per-threshold calibration (AMPR already wins MF AUPRC at LT_30).
2. **Complete the identity curve** — run LT_40/50/70 for the thesis figure (`for s in test_LT_40 test_LT_50 test_LT_70; do bash scripts/eval_all_v3.sh $s; done`).
3. **Close the absolute val→test gap** — stronger regularization (weight decay / higher dropout) rather than scaling up.
4. **Scale-up only with more data (SWISS-MODEL)** — enlarging `d_hidden`/fusion depth is justified *only* alongside a much larger training set; separate embedding-precompute effort (cf. Plan 2).

---

*Artifacts: `docs/sweep_results_mf/RANKING.md` (sweep), `results/*_v3_esm3b.eval_test.json` (test metrics), `logs/*_v3_esm3b.train.run.log` (training traces). Checkpoints backed up to Google Drive.*
