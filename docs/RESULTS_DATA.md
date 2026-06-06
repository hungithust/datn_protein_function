# AMPR v3 — Consolidated Results Data (for thesis tables/figures)

Single reference for every metric, extracted from `logs/*.eval_*.log`,
`docs/REPORT_v3_esm3b.md`, and `docs/sweep_results_mf/RANKING.md`. `test` ≡ `LT_95`
(3,123 proteins); `LT_30` = ≤30% identity to train (1,582 proteins).
Smin is the **normalized** [0,1] variant (lower better) — not comparable to DeepFRI raw-IC Smin.

## 1. Model-only test metrics (all metrics, raw vs DAG)

| Branch | Split | Var | Fmax | Smin | AUPRC_micro | AUPRC_macro | AUROC_micro | Coverage |
|---|---|---|---|---|---|---|---|---|
| MF | LT_30 | raw | 0.4769 | 0.7719 | 0.5124 | 0.4584 | 0.9025 | 0.9482 |
| MF | LT_30 | dag | 0.4777 | 0.7731 | 0.5024 | 0.4579 | 0.9042 | 0.9482 |
| MF | LT_95 | raw | 0.5488 | 0.6716 | 0.5974 | 0.5247 | 0.9204 | 0.9379 |
| MF | LT_95 | dag | 0.5498 | 0.6768 | 0.5851 | 0.5243 | 0.9217 | 0.9414 |
| BP | LT_30 | raw | 0.4322 | 0.8928 | 0.2751 | 0.2268 | 0.7747 | 1.0000 |
| BP | LT_30 | dag | 0.4358 | 0.8865 | 0.2803 | 0.2288 | 0.7788 | 1.0000 |
| BP | LT_95 | raw | 0.4557 | 0.8508 | 0.3012 | 0.2740 | 0.7803 | 0.9987 |
| BP | LT_95 | dag | 0.4582 | 0.8459 | 0.3057 | 0.2754 | 0.7839 | 0.9981 |
| CC | LT_30 | raw | 0.4883 | 0.8133 | 0.3792 | 0.3105 | 0.8356 | 0.9766 |
| CC | LT_30 | dag | 0.4909 | 0.8113 | 0.3832 | 0.3104 | 0.8382 | 0.9766 |
| CC | LT_95 | raw | 0.4941 | 0.7996 | 0.3635 | 0.3479 | 0.8396 | 0.9664 |
| CC | LT_95 | dag | 0.4955 | 0.7973 | 0.3665 | 0.3497 | 0.8418 | 0.9741 |

## 2. With DIAMOND ensemble (α=0.6, DAG-propagated)

| Branch | Split | Fmax | Smin | AUPRC_micro | AUROC_micro | hom-hits |
|---|---|---|---|---|---|---|
| MF | LT_30 | 0.5149 | 0.7491 | 0.4974 | 0.9075 | 839/1582 |
| MF | LT_95 | 0.6142 | 0.6120 | 0.6001 | 0.9262 | 2323/3123 |
| BP | LT_30 | 0.4603 | 0.8704 | 0.2835 | 0.7863 | 834/1582 |
| BP | LT_95 | 0.5069 | 0.7952 | 0.3409 | 0.8007 | 2273/3123 |
| CC | LT_30 | 0.5154 | 0.8019 | 0.3909 | 0.8426 | 623/1582 |
| CC | LT_95 | 0.5383 | 0.7598 | 0.4011 | 0.8527 | 1897/3123 |

## 3. DeepFRI-GCN baseline (Gligorijević et al. 2021)

| Branch | Split | Fmax | AUPRC_micro |
|---|---|---|---|
| MF | LT_30 | 0.545 | 0.443 |
| MF | LT_95 | 0.759 | 0.671 |
| BP | LT_30 | 0.282 | 0.169 |
| BP | LT_95 | 0.395 | 0.272 |
| CC | LT_30 | 0.434 | 0.308 |
| CC | LT_95 | 0.561 | 0.443 |

## 4. Validation Fmax_dag (best, final configs)

| Branch | best val Fmax_dag | test (dag) | gap | config |
|---|---|---|---|---|
| MF | 0.7603 | 0.5559 | 0.204 | wd=1e-2, dropout=0.2 (reg winner) |
| BP | 0.6726 | 0.4582 | 0.214 | baseline (wd=0, dropout=0.1) |
| CC | 0.6817 | 0.4955 | 0.186 | baseline (wd=0, dropout=0.1) |

> MF row uses the regularized model (adopted). BP/CC baseline kept (reg lowered their val).

## 5. Architecture sweep (MF, val Fmax_dag)

| classifier | d_hidden | go_emb | val Fmax_dag |
|---|---|---|---|
| both | 512 | combined | **0.7525** |
| label_attn | 512 | combined | 0.5987 |
| label_attn | 1024 | combined | 0.5950 |
| both | 1024 | combined | 0.5781 |
| any | any | text-only | 0.0209 (collapse) |

## 6. Regularization sweep (val Fmax_dag; test+DIAMOND)

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

- Model eval: `logs/{mf,bp,cc}_v3_esm3b.eval_{test,test_LT_30,test_LT_95}.log`
- Reg sweep: `logs/{mf,bp,cc}_reg_*.{log,eval_test_LT_95.log}`
- Arch sweep: `logs/sweep_mf_*.log`, `docs/sweep_results_mf/RANKING.md`
- Training traces: `logs/*_v3_esm3b.train.run.log`
