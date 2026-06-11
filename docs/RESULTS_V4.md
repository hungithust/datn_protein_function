# AMPR v4 — Module A (Multi-label SupCon) results

**Date:** 2026-06-11 · **Branch tested:** MF · **Verdict:** negative result (no test improvement)

Module A adds a Jaccard-weighted supervised contrastive loss (SupCon, Khosla et al.
NeurIPS 2020; multi-label generalization is our own, motivated by HEAL btad410) on the
fused representation `z`, via `weight · L_CL`. Config is identical to the v3 MF baseline
except for the `training.contrastive` block.

## 1. Loss-scale sensitivity (the key finding)

ASL classification loss is tiny (`cls ≈ 0.036`, mean over 489 terms at 0.88% positive
rate, γ⁻=4); SupCon is O(1) (`cl ≈ 4.3`). They differ by ~120×, so the contrastive term
dominates the gradient unless `weight ~ 1e-3`. The runbook's initial grid {0.1, 0.5, 1.0}
was the wrong scale — all collapsed classification.

| λ_cl | contrastive share of loss | val Fmax_dag |
|---|---|---|
| 0 (baseline) | 0% | 0.760 |
| **0.001** | ~12% of cls | **0.783** |
| 0.005 | ~60% of cls | 0.719 |
| 0.02 | ~2.4× cls | 0.575 |
| 0.1 | 92% of total | 0.353 |
| 0.5 | 98% | 0.420 |
| 1.0 | 99% | 0.292 |

→ Only λ_cl ≈ 1e-3 keeps classification dominant. This is a clean **hyperparameter
sensitivity** table for the thesis.

## 2. Best config (λ_cl=0.001) vs baseline — test, all bins (+DIAMOND, α=0.6)

| Bin | baseline v3 +D | v4 SupCon w0001 +D | Δ |
|---|---|---|---|
| LT_30 | 0.5149 | 0.5108 | −0.004 |
| LT_40 | 0.5253 | 0.5218 | −0.004 |
| LT_50 | 0.5495 | 0.5459 | −0.004 |
| LT_70 | 0.5840 | 0.5815 | −0.003 |
| LT_95 | 0.6142 | 0.6127 | −0.002 |

val_Fmax_dag: baseline 0.760 → w0001 0.783 (**+0.023**). test: unchanged-to-slightly-worse
at **every** bin (within noise, uniformly negative).

## 3. Conclusion

Module A raises **validation** Fmax but does **not** transfer to **test** at any
sequence-identity bin — including LT_30, where contrastive was hypothesized to help
low-identity generalization most. Net effect: the val→test gap **widens** (0.204 → 0.233).

This is an honest **negative result**: a Jaccard-weighted supervised contrastive auxiliary
does not improve test generalization for MF here; it increases val overfitting. The
val→test gap (~0.20), not modality usage, remains the core open problem (see modality
ablation §4 below — all three modalities contribute, so the gap is not a dead-branch issue).

## 4. Phase-0 modality ablation (eval-only, v3 baseline, test_LT_95)

Zeroing a branch before fusion (lower bound — model was trained with all three):

| Branch | full | −gnn | −ppi | seq_only |
|---|---|---|---|---|
| MF | 0.5498 | 0.3021 | 0.4912 | 0.0875 |
| BP | 0.4582 | 0.3900 | 0.3731 | 0.1625 |
| CC | 0.4955 | 0.3992 | 0.4145 | 0.2085 |

All three modalities contribute substantially (GNN dominant on MF). This **refutes** the
spec §0 working hypothesis that structure/PPI might be near-dead — multimodality is
justified. Good standalone ablation figure for the thesis.

## Source

- Training logs: `logs/mf_v4_supcon_w{0001,0005,002,01,05,10}_train.log`
- Eval: `results/mf_v4_supcon_w0001_predictions.eval_test_LT_*.json`
- Ablation: `results/{mf,bp,cc}_v3_esm3b_predictions.ablate_test_LT_95.json`
- Baseline reference: `docs/RESULTS_DATA.md`
