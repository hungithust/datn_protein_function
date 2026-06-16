# Phase V6 — SWISS-MODEL Data Expansion: Results

**Ngày:** 2026-06-17
**Spec:** [docs/superpowers/specs/2026-06-13-ampr-swissmodel-expansion-design.md](superpowers/specs/2026-06-13-ampr-swissmodel-expansion-design.md)
**Plan:** [docs/superpowers/plans/2026-06-13-ampr-swissmodel-expansion.md](superpowers/plans/2026-06-13-ampr-swissmodel-expansion.md)

> **Trạng thái:** MF + CC đã xong (pretrain→finetune, 3 seed, eval đủ bin).
> BP đang chạy lại (BCE pretrain). Baseline đối chứng 650M-PDB-30K **đang chờ chạy**
> (xem §4) — **kết luận cuối phụ thuộc số baseline này.**

---

## 1. Thiết lập

- **Chiến lược B:** pretrain 220K SWISS-MODEL → finetune 30K PDB.
- **Backbone:** ESM-2 **650M** (d_model 1280) — *đổi từ 3B* để chứa 220K embeddings.
- **Recipe finetune (sau khi sửa under-training):** 60 epoch, lr 3e-4, ASL+DAG, dropout 0.4,
  inference = DAG propagation → DIAMOND ensemble (α=0.6) → 3-seed ensemble.
- **Pretrain:** warm-continue, MF/CC dùng ASL; **BP dùng BCE+pos_weight** (ASL collapse
  trên long-tail 1943 term — val phẳng 0.2038, dead gradient).

### Val Fmax_dag (PDB valid, trung bình 3 seed)
| Nhánh | val Fmax_dag |
|---|---|
| MF | ~0.665 |
| CC | ~0.61 |
| BP | ~0.527 (s42, đang chạy lại) |

---

## 2. Test Fmax — AMPR-B (650M, 220K→30K), 3-seed ensemble

| Bin | MF (dag) | MF (+DIAMOND) | CC (dag) | CC (+DIAMOND) |
|---|---|---|---|---|
| LT_30 | 0.442 | 0.524 | 0.490 | 0.515 |
| LT_40 | 0.452 | 0.537 | 0.492 | 0.516 |
| LT_50 | 0.470 | 0.562 | 0.497 | 0.521 |
| LT_70 | 0.496 | 0.594 | 0.496 | 0.522 |
| **full (LT_95)** | **0.515** | **0.622** | **0.503** | **0.535** |

Full-test các metric khác: MF ens Smin=0.619 AUPRC=0.563; CC ens Smin=0.774 AUPRC=0.373.

---

## 3. So sánh với baseline (full test Fmax)

| Model | Train data | Backbone | MF | CC |
|---|---|---|---|---|
| DeepFRI-MERGED | 220K SWISS-MODEL + PDB | GCN | 0.626 | ~0.61 |
| AMPR-30K (cũ) | 29,902 PDB | ESM-2 **3B** | **0.649** | trail |
| **AMPR-B (đây)** | 220K → 30K | ESM-2 **650M** | 0.622 | 0.535 |
| AMPR 650M-PDB-30K baseline | 29,902 PDB | ESM-2 650M | *(chờ §4)* | *(chờ §4)* |

**Quan sát:**
- MF của AMPR-B (0.622) **thấp hơn** cả DeepFRI (0.626) và AMPR-30K-3B (0.649).
- CC (0.535) **tụt rõ** dưới DeepFRI (~0.61).
- Gap val→test của model thuần *rộng ra*: MF val 0.665 → test dag 0.515 (gap **0.15**),
  so với gap ~0.11 của AMPR-30K. → 220K SWISS-MODEL **không** tạo representation tổng
  quát hơn; điểm số chủ yếu do DIAMOND homology gánh (MF dag 0.515 → +DIAMOND 0.622).

---

## 4. Confound chưa gỡ → baseline đối chứng (BẮT BUỘC để kết luận)

Thí nghiệm đổi **đồng thời** backbone (3B→650M) **và** data (+220K). Không thể quy
việc MF tụt 0.649→0.622 cho data hay cho backbone. Cần biến đối chứng **650M-PDB-30K**
(cùng recipe finetune, **không** `--init-from`):

- Nếu baseline 650M ≈ 0.62 → drop là **do backbone**; data 220K **trung tính**.
  Kết luận: "ở 650M, data-scaling SWISS-MODEL không cải thiện; chênh so 3B là do capacity."
- Nếu baseline 650M < 0.62 → pretrain 220K **có giúp** ở cùng backbone → story dương cứu được.

Configs: `configs/{mf,cc,bp}_v6_pdb30base_s{42,123,2024}.yaml` (sinh bởi
`scripts/gen_v6_baseline_configs.py`). Lệnh chạy ở §6.

---

## 5. Verdict (sơ bộ, theo tiêu chí spec §1)

- **Tiêu chí chính (test ≥ AMPR-30K mọi bin):** ❌ chưa đạt (MF/CC dưới mốc 3B cũ).
- **Negative result hợp lệ (spec §7):** data-scaling SWISS-MODEL **không** thu hẹp gap
  val→test; nhiều khả năng do nhiễu homology-model + hạ backbone xuống 650M.
- **Điểm cần baseline §4 để phát biểu sạch:** liệu ở cùng 650M, 220K có trung tính hay có hại.
- BP: collapse ASL đã chữa bằng BCE (đóng góp phương pháp luận, xem [[ampr-phase3-mf-collapse-fix]]);
  số cuối bổ sung khi chạy xong.

---

## 6. Lệnh chạy baseline + eval (server)

```bash
# Train 650M-PDB-30K baseline (MF/CC, 3 seed) — KHÔNG --init-from
for cfg in mf cc; do
  for seed in 42 123 2024; do
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/${cfg}_v6_pdb30base_s${seed}.yaml --seed $seed \
      2>&1 | tee logs/${cfg}_v6_pdb30base_s${seed}.run.log
  done
done

# Eval + 3-seed ensemble (đủ bin)
for cfg in mf cc; do
  cks="checkpoints/${cfg}_v6_pdb30base_s42/best.pt checkpoints/${cfg}_v6_pdb30base_s123/best.pt checkpoints/${cfg}_v6_pdb30base_s2024/best.pt"
  for split in test test_LT_30 test_LT_40 test_LT_50 test_LT_70 test_LT_95; do
    CUDA_VISIBLE_DEVICES=3 python scripts/ensemble_eval.py \
      --config configs/${cfg}_v6_pdb30base_s42.yaml --checkpoints $cks --split $split \
      2>&1 | tee logs/ensemble_${cfg}_pdb30base_${split}.log
  done
done
grep -iE fmax logs/ensemble_*_pdb30base_*.log
```

Điền số vào §3 (dòng baseline) + chốt §5 sau khi có kết quả.
