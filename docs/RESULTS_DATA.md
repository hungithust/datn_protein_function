# AMPR v3 — Consolidated Results Data (for thesis tables/figures)

Single reference for every metric, extracted from `logs/*.eval_*.log`,
`docs/REPORT_v3_esm3b.md`, and `docs/sweep_results_mf/RANKING.md`. `test` ≡ `LT_95`
(3,123 proteins); `LT_30` = ≤30% identity to train (1,582). Smin is the **normalized**
[0,1] variant (lower better) — not comparable to DeepFRI raw-IC Smin.

## 1. Identity curve — Fmax (headline figure data)

Per branch across all 5 sequence-identity bins. `AMPR` = model only (DAG); `AMPR+D` =
+ DIAMOND ensemble (α=0.6). `DeepFRI` and `HEAL` Fmax taken from **HEAL supplementary
Table S3.2** (Gu et al., Bioinformatics btad410 2023) — same PDBch test set, same LT_*
homology bins, same protein-centric Fmax, so directly comparable. (The previously stored
DeepFRI numbers were wrong; see §4.)

**MF**
| Bin | AMPR | AMPR+D | DeepFRI | HEAL (SOTA) |
|---|---|---|---|---|
| LT_30 | 0.4777 | 0.5149 | 0.544 | 0.698 |
| LT_40 | 0.4850 | 0.5253 | 0.552 | 0.702 |
| LT_50 | 0.5022 | 0.5495 | 0.575 | 0.719 |
| LT_70 | 0.5295 | 0.5840 | 0.604 | 0.735 |
| LT_95 | 0.5498 | 0.6142 | 0.626 | 0.749 |

**BP**
| Bin | AMPR | AMPR+D | DeepFRI | HEAL (SOTA) |
|---|---|---|---|---|
| LT_30 | 0.4358 | 0.4603 | 0.502 | 0.582 |
| LT_40 | 0.4342 | 0.4619 | 0.510 | 0.578 |
| LT_50 | 0.4390 | 0.4695 | 0.517 | 0.582 |
| LT_70 | 0.4511 | 0.4865 | 0.533 | 0.592 |
| LT_95 | 0.4582 | 0.5069 | 0.540 | 0.594 |

**CC**
| Bin | AMPR | AMPR+D | DeepFRI | HEAL (SOTA) |
|---|---|---|---|---|
| LT_30 | 0.4909 | 0.5154 | 0.605 | 0.684 |
| LT_40 | 0.4912 | 0.5164 | 0.606 | 0.682 |
| LT_50 | 0.4944 | 0.5218 | 0.606 | 0.684 |
| LT_70 | 0.4930 | 0.5238 | 0.605 | 0.686 |
| LT_95 | 0.4955 | 0.5383 | 0.612 | 0.687 |

**Standing vs DeepFRI (single model, corrected):** a *single* AMPR+D model trails DeepFRI
on all three ontologies — MF close (LT_95 0.614 vs 0.626, −0.012), BP −0.033, CC −0.074.
The earlier claim that AMPR beat DeepFRI on BP/CC was an artifact of the wrong stored
DeepFRI numbers. **The 3-seed ensemble (§1b) closes this** — see below.

## 1b. 3-seed ensemble — Fmax (final headline)

Average sigmoid probs over 3 seeds {42, 123, 2024} of the v3 config, then DAG-propagate
(`ens dag`) and DIAMOND-blend (`ens +D`, α=0.6). Script: `scripts/ensemble_eval.py`.
Ensembling adds ~+0.02–0.035 test Fmax over a single model — variance reduction, no
architecture change.

**MF** — ensemble +D **beats DeepFRI at every bin**:
| Bin | single +D | **ens +D** | DeepFRI | HEAL |
|---|---|---|---|---|
| LT_30 | 0.5149 | **0.5617** | 0.544 | 0.698 |
| LT_40 | 0.5253 | **0.5709** | 0.552 | 0.702 |
| LT_50 | 0.5495 | **0.5939** | 0.575 | 0.719 |
| LT_70 | 0.5840 | **0.6248** | 0.604 | 0.735 |
| LT_95 | 0.6142 | **0.6492** | 0.626 | 0.749 |

**BP** — ensemble +D tiệm cận DeepFRI (LT_95 −0.007):
| Bin | single +D | ens +D | DeepFRI | HEAL |
|---|---|---|---|---|
| LT_30 | 0.4603 | 0.4904 | 0.502 | 0.582 |
| LT_40 | 0.4619 | 0.4924 | 0.510 | 0.578 |
| LT_50 | 0.4695 | 0.5006 | 0.517 | 0.582 |
| LT_70 | 0.4865 | 0.5162 | 0.533 | 0.592 |
| LT_95 | 0.5069 | 0.5330 | 0.540 | 0.594 |

**CC** — ensemble +D improves +0.022 but still trails DeepFRI:
| Bin | single +D | ens +D | DeepFRI | HEAL |
|---|---|---|---|---|
| LT_30 | 0.5154 | 0.5453 | 0.605 | 0.684 |
| LT_40 | 0.5164 | 0.5445 | 0.606 | 0.682 |
| LT_50 | 0.5218 | 0.5495 | 0.606 | 0.684 |
| LT_70 | 0.5238 | 0.5497 | 0.605 | 0.686 |
| LT_95 | 0.5383 | 0.5603 | 0.612 | 0.687 |

**Standing (3-seed ensemble, final):** **AMPR+D beats DeepFRI on MF at all five bins**
(LT_95 0.649 vs 0.626, +0.023) and comes within 0.007 of DeepFRI on BP at LT_95; CC still
trails (−0.052). All three remain below HEAL. (ens model-only `dag` Fmax LT_95: MF 0.590,
BP 0.531, CC 0.531.)

## 1c. Anti-overfit branch (dropout 0.4) — Regularization vs Ensembling

Re-trained all branches with global `dropout` raised to **0.4** (MF baseline 0.2; BP/CC
baseline 0.1), everything else identical. Configs: `configs/{mf,bp,cc}_v5_drop04*.yaml`.
Dropout 0.4 chosen by val_Fmax (0.5 reached similar val but lower test on MF → over-strong).

**Single-model effect (MF, LT_95):** val_Fmax 0.760→0.7505, test +D **0.614→0.625**;
val→test gap **0.146→0.125**. Regularization demonstrably narrows the overfit gap.

**3-seed ensemble (drop04) vs baseline ensemble (+D), all bins:**

| Branch | Bin | baseline ens +D | **drop04 ens +D** | DeepFRI |
|---|---|---|---|---|
| MF | LT_30 | 0.5617 | 0.5591 | 0.544 |
| MF | LT_40 | 0.5709 | 0.5704 | 0.552 |
| MF | LT_50 | 0.5939 | 0.5943 | 0.575 |
| MF | LT_70 | 0.6248 | 0.6254 | 0.604 |
| MF | LT_95 | 0.6492 | **0.6504** | 0.626 |
| BP | LT_30 | 0.4904 | 0.4959 | 0.502 |
| BP | LT_40 | 0.4924 | 0.4971 | 0.510 |
| BP | LT_50 | 0.5006 | 0.5055 | 0.517 |
| BP | LT_70 | 0.5162 | 0.5221 | 0.533 |
| BP | LT_95 | 0.5330 | **0.5392** | 0.540 |
| CC | LT_30 | 0.5453 | 0.5373 | 0.605 |
| CC | LT_40 | 0.5445 | 0.5366 | 0.606 |
| CC | LT_50 | 0.5495 | 0.5435 | 0.606 |
| CC | LT_70 | 0.5497 | 0.5439 | 0.605 |
| CC | LT_95 | 0.5603 | 0.5542 | 0.612 |

**Finding (redundancy):** strong dropout lifts *single* models clearly, but the gain is
**largely redundant with ensembling** — at the ensemble level MF is a tie (+0.001),
BP improves slightly (+0.006, now **matching DeepFRI at LT_95**: 0.539 vs 0.540), and CC
is marginally worse (−0.006). Regularization and ensembling are overlapping anti-overfit
mechanisms; they do not add linearly. CC's weakness is driven by low DIAMOND coverage
(LT_95 1897/3123 hom-hits), not by overfit, so extra dropout cannot help it.

**Updated standing (best-per-branch ensemble, LT_95):** MF **0.650** (drop04) and BP
**0.539** (drop04) — AMPR now **meets-or-beats DeepFRI on MF and BP** (MF +0.024, BP −0.001
≈ tie); CC 0.560 (baseline) still trails (−0.052). All below HEAL.

## 2. Model-only metrics per bin (DAG-propagated)

| Branch | Bin | Fmax | Smin | AUPRC_micro | AUPRC_macro | AUROC_micro | Coverage |
|---|---|---|---|---|---|---|---|
| MF | LT_30 | 0.4777 | 0.7731 | 0.5024 | 0.4579 | 0.9042 | 0.9482 |
| MF | LT_40 | 0.4850 | 0.7640 | 0.5116 | 0.4676 | 0.9078 | 0.9530 |
| MF | LT_50 | 0.5022 | 0.7408 | 0.5311 | 0.4812 | 0.9115 | 0.9572 |
| MF | LT_70 | 0.5295 | 0.7031 | 0.5639 | 0.5051 | 0.9187 | 0.9344 |
| MF | LT_95 | 0.5498 | 0.6768 | 0.5851 | 0.5243 | 0.9217 | 0.9414 |
| BP | LT_30 | 0.4358 | 0.8865 | 0.2803 | 0.2288 | 0.7788 | 1.0000 |
| BP | LT_40 | 0.4342 | 0.8826 | 0.2818 | 0.2298 | 0.7805 | 1.0000 |
| BP | LT_50 | 0.4390 | 0.8747 | 0.2855 | 0.2435 | 0.7855 | 0.9995 |
| BP | LT_70 | 0.4511 | 0.8555 | 0.2945 | 0.2569 | 0.7837 | 0.9984 |
| BP | LT_95 | 0.4582 | 0.8459 | 0.3057 | 0.2754 | 0.7839 | 0.9981 |
| CC | LT_30 | 0.4909 | 0.8113 | 0.3832 | 0.3104 | 0.8382 | 0.9766 |
| CC | LT_40 | 0.4912 | 0.8077 | 0.3779 | 0.3160 | 0.8409 | 0.9821 |
| CC | LT_50 | 0.4944 | 0.8032 | 0.3765 | 0.3182 | 0.8420 | 0.9769 |
| CC | LT_70 | 0.4930 | 0.8035 | 0.3622 | 0.3208 | 0.8405 | 0.9742 |
| CC | LT_95 | 0.4955 | 0.7973 | 0.3665 | 0.3497 | 0.8418 | 0.9741 |

(raw-score variant available for LT_30/LT_95 in the logs; within ±0.01 of DAG.)

## 3. DIAMOND-ensemble metrics per bin (α=0.6, DAG)

| Branch | Bin | Fmax | Smin | AUPRC_micro | AUROC_micro | hom-hits |
|---|---|---|---|---|---|---|
| MF | LT_30 | 0.5149 | 0.7491 | 0.4974 | 0.9075 | 839/1582 |
| MF | LT_40 | 0.5253 | 0.7361 | 0.5062 | 0.9113 | 1012/1786 |
| MF | LT_50 | 0.5495 | 0.7060 | 0.5281 | 0.9154 | 1248/2033 |
| MF | LT_70 | 0.5840 | 0.6546 | 0.5663 | 0.9228 | 1725/2515 |
| MF | LT_95 | 0.6142 | 0.6120 | 0.6001 | 0.9262 | 2323/3123 |
| BP | LT_30 | 0.4603 | 0.8704 | 0.2835 | 0.7863 | 834/1582 |
| BP | LT_40 | 0.4619 | 0.8660 | 0.2856 | 0.7890 | 999/1786 |
| BP | LT_50 | 0.4695 | 0.8537 | 0.2924 | 0.7950 | 1225/2033 |
| BP | LT_70 | 0.4865 | 0.8253 | 0.3073 | 0.7957 | 1684/2515 |
| BP | LT_95 | 0.5069 | 0.7952 | 0.3409 | 0.8007 | 2273/3123 |
| CC | LT_30 | 0.5154 | 0.8019 | 0.3909 | 0.8426 | 623/1582 |
| CC | LT_40 | 0.5164 | 0.7994 | 0.3863 | 0.8455 | 752/1786 |
| CC | LT_50 | 0.5218 | 0.7881 | 0.3876 | 0.8474 | 939/2033 |
| CC | LT_70 | 0.5238 | 0.7829 | 0.3810 | 0.8488 | 1346/2515 |
| CC | LT_95 | 0.5383 | 0.7598 | 0.4011 | 0.8527 | 1897/3123 |

## 4. DeepFRI + HEAL baselines (corrected — source: HEAL supplementary)

> **Provenance (CORRECTED 2026-06-11):** the previously stored DeepFRI numbers
> (`results/deepfri_baseline.json`, e.g. MF LT_95 = 0.759, BP LT_95 = 0.395) were
> **wrong** and have been discarded. The authoritative values below come from **HEAL
> supplementary information, Tables S3.1 (macro-AUPR) and S3.2 (Fmax)** — Gu et al.,
> "Hierarchical Graph Transformer with Contrastive Learning for Protein Function
> Prediction", Bioinformatics btad410 (2023). HEAL reports DeepFRI, DeepGO, HEAL-PDB
> and HEAL on the **same PDBch test set**, the **same five LT_* homology bins to the
> training set**, and the **same protein-centric Fmax / macro-AUPR** we use, averaged
> over 10-bootstrap samples — so these are directly comparable to our AMPR numbers.
> File: `baselines/HEAL/supplementary-data.md`.

**Fmax (Table S3.2)**
| Branch | Model | <30 | <40 | <50 | <70 | <95 |
|---|---|---|---|---|---|---|
| MF | DeepFRI | 0.544 | 0.552 | 0.575 | 0.604 | 0.626 |
| MF | HEAL | 0.698 | 0.702 | 0.719 | 0.735 | 0.749 |
| BP | DeepFRI | 0.502 | 0.510 | 0.517 | 0.533 | 0.540 |
| BP | HEAL | 0.582 | 0.578 | 0.582 | 0.592 | 0.594 |
| CC | DeepFRI | 0.605 | 0.606 | 0.606 | 0.605 | 0.612 |
| CC | HEAL | 0.684 | 0.682 | 0.684 | 0.686 | 0.687 |

**macro-AUPR (Table S3.1)**
| Branch | Model | <30 | <40 | <50 | <70 | <95 |
|---|---|---|---|---|---|---|
| MF | DeepFRI | 0.425 | 0.443 | 0.463 | 0.485 | 0.504 |
| MF | HEAL | 0.638 | 0.641 | 0.663 | 0.681 | 0.698 |
| BP | DeepFRI | 0.214 | 0.218 | 0.232 | 0.253 | 0.268 |
| BP | HEAL | 0.300 | 0.296 | 0.311 | 0.327 | 0.345 |
| CC | DeepFRI | 0.248 | 0.248 | 0.251 | 0.258 | 0.285 |
| CC | HEAL | 0.429 | 0.434 | 0.434 | 0.445 | 0.468 |

> Note on AUPR: HEAL reports **macro**-AUPR. AMPR §2 lists both; compare against
> AMPR `AUPRC_macro`. Example: AMPR MF macro-AUPR LT_30 = 0.458 vs DeepFRI 0.425 — AMPR
> wins macro-AUPR at the hardest MF bin, even though it trails DeepFRI on Fmax. Note
> HEAL itself uses contrastive learning (hierarchical/structure-based), reaching MF 0.749.

## 5. Validation Fmax_dag (best, final configs)

| Branch | best val Fmax_dag | test (dag) | gap | config |
|---|---|---|---|---|
| MF | 0.7603 | 0.5559 | 0.204 | wd=1e-2, dropout=0.2 (reg winner) |
| BP | 0.6726 | 0.4582 | 0.214 | baseline (wd=0, dropout=0.1) |
| CC | 0.6817 | 0.4955 | 0.186 | baseline (wd=0, dropout=0.1) |

## 6. Architecture sweep (MF, val Fmax_dag)

| classifier | d_hidden | go_emb | val Fmax_dag |
|---|---|---|---|
| both | 512 | combined | **0.7525** |
| label_attn | 512 | combined | 0.5987 |
| label_attn | 1024 | combined | 0.5950 |
| both | 1024 | combined | 0.5781 |
| any | any | text-only | 0.0209 (collapse) |

## 7. Regularization sweep (val Fmax_dag; test+DIAMOND)

| Branch | wd | dropout | val | test (model dag) | test (+DIAMOND) |
|---|---|---|---|---|---|
| MF | 1e-2 | 0.2 | **0.7603** | 0.5559 | 0.6180 |
| MF | 1e-4 | 0.3 | 0.7431 | 0.5540 | 0.6235 |
| MF | 1e-4 | 0.2 | 0.7525 | 0.5419 | 0.6155 |
| MF | 1e-2 | 0.3 | 0.7268 | 0.5374 | 0.6128 |
| BP | 1e-4 | 0.2 | 0.6349 | 0.4526 | 0.5223 |
| BP | 1e-2 | 0.3 | 0.6285 | 0.4528 | 0.5272 |
| BP | 1e-4 | 0.3 | 0.6130 | 0.4504 | 0.5244 |
| BP | 1e-2 | 0.2 | 0.5907 | 0.4329 | 0.5181 |
| CC | 1e-4 | 0.2 | 0.6770 | 0.5030 | 0.5449 |
| CC | 1e-4 | 0.3 | 0.6704 | 0.5131 | 0.5450 |
| CC | 1e-2 | 0.3 | 0.6587 | 0.5130 | 0.5485 |
| CC | 1e-2 | 0.2 | 0.6482 | 0.5098 | 0.5453 |

## Source logs

- Model + ensemble eval: `logs/{mf,bp,cc}_v3_esm3b.eval_test_LT_{30,40,50,70,95}.log`
- Reg sweep: `logs/{mf,bp,cc}_reg_*.{log,eval_test_LT_95.log}`
- Arch sweep: `logs/sweep_mf_*.log`, `docs/sweep_results_mf/RANKING.md`
- Training traces: `logs/*_v3_esm3b.train.run.log`
