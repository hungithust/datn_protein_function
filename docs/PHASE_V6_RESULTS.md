# Phase V6 — SWISS-MODEL Data Expansion: Results

**Ngày:** 2026-06-17
**Spec:** [docs/superpowers/specs/2026-06-13-ampr-swissmodel-expansion-design.md](superpowers/specs/2026-06-13-ampr-swissmodel-expansion-design.md)
**Plan:** [docs/superpowers/plans/2026-06-13-ampr-swissmodel-expansion.md](superpowers/plans/2026-06-13-ampr-swissmodel-expansion.md)

> **Trạng thái:** MF + CC đã xong (pretrain→finetune + baseline đối chứng, 3 seed, eval đủ bin).
> **Confound đã gỡ:** baseline 650M-PDB-30K cho kết luận sạch — **pretrain 220K SWISS-MODEL
> làm HẠI** (xem §3, §5). BP (BCE pretrain) bổ sung sau.

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
| AMPR-30K (cũ) | 29,902 PDB | ESM-2 **3B** | 0.649 | trail |
| **AMPR 650M-PDB-30K baseline** | 29,902 PDB | ESM-2 650M | **0.654** | **0.566** |
| **AMPR-B (220K→30K)** | 220K → 30K | ESM-2 650M | 0.622 | 0.535 |

**BP (full test Fmax):** AMPR-B (220K→30K, BCE-pretrain) dag **0.398** / +DIAMOND **0.521**
— đã *thấp hơn* DeepFRI (~0.54) và AMPR-30K-3B (0.539). Baseline BP 650M-PDB-30K **chờ chạy**
(configs `bp_v6_pdb30base_s*`) để xác nhận pretrain hại cho BP như MF/CC.

### Đối chứng sạch — pretrain SWISS-MODEL làm HẠI (cùng 650M, cùng recipe)

| Bin | MF baseline | MF AMPR-B | Δ | CC baseline | CC AMPR-B | Δ |
|---|---|---|---|---|---|---|
| LT_30 | 0.565 | 0.524 | −0.041 | 0.556 | 0.515 | −0.041 |
| LT_40 | 0.578 | 0.537 | −0.041 | 0.556 | 0.516 | −0.040 |
| LT_50 | 0.600 | 0.562 | −0.038 | 0.560 | 0.521 | −0.039 |
| LT_70 | 0.630 | 0.594 | −0.036 | 0.557 | 0.522 | −0.035 |
| **full** | **0.654** | **0.622** | **−0.032** | **0.566** | **0.535** | **−0.031** |
| **full (dag, model thuần)** | **0.644** | **0.515** | **−0.129** | **0.556** | **0.503** | **−0.053** |

**Quan sát:**
- Pretrain 220K SWISS-MODEL **thua baseline ở MỌI bin** (MF & CC) → **không trung tính, mà có hại.**
- Tổn hại tập trung ở **model thuần (dag)**: MF 0.644→0.515 (**−0.13**). DIAMOND homology
  che bớt nên Δ full chỉ −0.03, nhưng net vẫn âm.
- Gap val→test của AMPR-B *rộng ra*: MF val 0.665 → test dag 0.515 (gap **0.15**) vs baseline
  gap nhỏ hơn → pretrain đẩy model fit đặc trưng homology-model **không** chuyển sang PDB sạch.
- **Backbone không phải vấn đề:** baseline 650M (0.654) *vượt* DeepFRI (0.626) và ≈/hơn
  AMPR-30K-3B (0.649) nhờ recipe mới (60ep, dropout 0.4). Toàn bộ phần tụt của AMPR-B là **do pretrain**.

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

## 5. Verdict (theo tiêu chí spec §1) — đã có baseline, kết luận sạch

- **Giả thuyết (data-scaling 220K thu hẹp gap val→test) → BÁC BỎ.** Ở cùng backbone 650M
  và cùng recipe, pretrain SWISS-MODEL **làm giảm** test Fmax ở mọi bin (MF −0.03, CC −0.03
  full; model thuần MF −0.13). Không phải bão hoà — mà là **suy giảm chủ động**.
- **Cơ chế:** homology-model SWISS-MODEL nhiễu (contact map dựng từ model, không phải cấu
  trúc thực nghiệm). Pretrain đẩy biểu diễn fit cái nhiễu đó; finetune 30K không gỡ hết →
  gap val→test rộng ra (val 0.665 vs test dag 0.515).
- **Confound đã loại:** baseline 650M (MF 0.654) vượt DeepFRI (0.626) và ≈ 3B cũ (0.649) →
  việc hạ 3B→650M **không** gây hại; toàn bộ tụt là do pretrain.
- **Đóng góp luận văn (negative result mạnh, spec §7):** *"Pretrain quy mô lớn trên cấu
  trúc homology-model (SWISS-MODEL) làm suy giảm dự đoán chức năng trên cấu trúc thực nghiệm,
  ngay cả khi đã finetune — nhiễu cấu trúc của homology model có hại, không chỉ vô ích."*
  Đây là phản-ví dụ cho phương pháp "MERGED" của DeepFRI khi xét riêng modality cấu trúc.
- **Hệ quả hành động:** chốt **650M-PDB-30K làm model chính thức** (đã vượt DeepFRI & 3B cũ);
  bỏ nhánh pretrain SWISS-MODEL.
- BP: collapse ASL đã chữa bằng BCE (đóng góp phương pháp luận, xem
  [[ampr-phase3-mf-collapse-fix]]); số BP bổ sung khi chạy xong — kỳ vọng cùng xu hướng.

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
