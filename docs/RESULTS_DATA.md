# AMPR v3 — Consolidated Results Data (for thesis tables/figures)

Single reference for every metric, extracted from `logs/*.eval_*.log`,
`docs/REPORT_v3_esm3b.md`, and `docs/sweep_results_mf/RANKING.md`. `test` ≡ `LT_95`
(3,123 proteins); `LT_30` = ≤30% identity to train (1,582). Smin is the **normalized**
[0,1] variant (lower better) — not comparable to DeepFRI raw-IC Smin.

## 1. Identity curve — Fmax (headline figure data)

Per branch across all 5 sequence-identity bins. `AMPR` = model only (DAG); `AMPR+D` =
+ DIAMOND ensemble (α=0.6); `DeepFRI` = DeepFRI-GCN (Gligorijević 2021).

**MF**
| Bin | AMPR | AMPR+D | DeepFRI |
|---|---|---|---|
| LT_30 | 0.4777 | 0.5149 | 0.545 |
| LT_40 | 0.4850 | 0.5253 | 0.587 |
| LT_50 | 0.5022 | 0.5495 | 0.626 |
| LT_70 | 0.5295 | 0.5840 | 0.717 |
| LT_95 | 0.5498 | 0.6142 | 0.759 |

**BP**
| Bin | AMPR | AMPR+D | DeepFRI |
|---|---|---|---|
| LT_30 | 0.4358 | **0.4603** | 0.282 |
| LT_40 | 0.4342 | **0.4619** | 0.302 |
| LT_50 | 0.4390 | **0.4695** | 0.327 |
| LT_70 | 0.4511 | **0.4865** | 0.365 |
| LT_95 | 0.4582 | **0.5069** | 0.395 |

**CC**
| Bin | AMPR | AMPR+D | DeepFRI |
|---|---|---|---|
| LT_30 | 0.4909 | **0.5154** | 0.434 |
| LT_40 | 0.4912 | **0.5164** | 0.462 |
| LT_50 | 0.4944 | **0.5218** | 0.493 |
| LT_70 | 0.4930 | 0.5238 | 0.541 |
| LT_95 | 0.4955 | 0.5383 | 0.561 |

**Robustness (Fmax slope LT_30→LT_95):** AMPR+D MF +0.099, BP +0.047, CC +0.023 —
much flatter than DeepFRI (MF +0.214, BP +0.113, CC +0.127). AMPR+D beats DeepFRI on
**all 5 BP bins** and **CC LT_30/40/50** (loses CC LT_70/95, close); MF trails on Fmax
at every bin (but wins MF AUPRC_micro at LT_30, see §4).

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

## 4. DeepFRI-GCN baseline (Gligorijević et al. 2021)

> **Provenance:** DeepFRI numbers taken from `results/deepfri_baseline.json`, whose
> `_source` cites Gligorijević et al., Nat Commun 2021 (Table 1, DeepFRI-GCN row,
> https://www.nature.com/articles/s41467-021-23303-9).
> NOTE: the original DeepFRI paper reports MF Fmax ≈ 0.625–0.631 on its test set. The
> MF LT_95 value of **0.759** here is therefore NOT the paper's headline number — it is
> consistent with a re-evaluation under the PDBch LT_* sequence-identity split protocol
> (LT_95 ≈ paper's least-stringent / highest-identity bin). The JSON `_comment` itself
> flags that these numbers must be re-verified against the paper before final submission.
> In the thesis, state explicitly whether each DeepFRI figure is the paper's reported
> value or a re-evaluation under our LT_* protocol.

| Branch | Bin | Fmax | AUPRC_micro |
|---|---|---|---|
| MF | LT_30 / 40 / 50 / 70 / 95 | 0.545 / 0.587 / 0.626 / 0.717 / 0.759 | 0.443 / 0.483 / 0.521 / 0.628 / 0.671 |
| BP | LT_30 / 40 / 50 / 70 / 95 | 0.282 / 0.302 / 0.327 / 0.365 / 0.395 | 0.169 / 0.185 / 0.203 / 0.243 / 0.272 |
| CC | LT_30 / 40 / 50 / 70 / 95 | 0.434 / 0.462 / 0.493 / 0.541 / 0.561 | 0.308 / 0.339 / 0.367 / 0.418 / 0.443 |

> AMPR wins MF AUPRC_micro at LT_30 (0.502 vs 0.443) despite trailing on MF Fmax.

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
