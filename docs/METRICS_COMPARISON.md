# Bảng metric tổng hợp — AMPR vs baseline (PDBch benchmark)

**Mục đích:** một nguồn tra cứu chính xác, có dẫn nguồn, cho 3 metric phổ biến nhất của
mảng dự đoán chức năng protein dựa trên cấu trúc, qua tất cả phiên bản AMPR và các baseline
trong paper. Mọi số đo trên **cùng tập test PDBch** (3,416 chain, DeepFRI release), cùng 5
ngưỡng homology LT_30..LT_95, cùng định nghĩa Fmax/AUPR.

---

## 0. Lưu ý độ chính xác về metric (đọc trước)

Bộ metric **chuẩn** của subfield này (DeepFRI 2021, HEAL 2023, DeepGO) là:

| Metric | Mô tả | Hướng | So sánh chéo được? |
|---|---|---|---|
| **Fmax** | max F1 protein-centric trên mọi ngưỡng | cao tốt | ✅ mọi model |
| **AUPR (macro)** | diện tích Precision–Recall, trung bình theo **term** | cao tốt | ✅ mọi model |
| **Smin** | khoảng cách ngữ nghĩa theo Information Content | thấp tốt | ⚠️ xem dưới |

- **AUC (ROC-AUC) KHÔNG phải metric chuẩn ở đây.** DeepFRI và HEAL **không công bố AUC**
  trên PDBch — vì nhãn GO cực thưa nên ROC-AUC bị thổi phồng (~0.9) và không phân biệt được
  model. Cộng đồng dùng **AUPR thay cho AUC** chính vì lý do này. → Không có số AUC baseline
  để đối chiếu (xem §4 cho AUROC của riêng AMPR).
- **Smin của AMPR là biến thể chuẩn hoá [0,1]**, KHÁC Smin raw-IC của DeepFRI/HEAL → **không
  so trực tiếp** được. Vì vậy bảng chính dưới đây dùng **Fmax và macro-AUPR** (hai metric
  *vừa chuẩn vừa so chéo được*).

**Nguồn baseline:** HEAL supplementary, Bảng S3.1 (macro-AUPR) & S3.2 (Fmax) — Gu et al.,
*Bioinformatics* 39(7):btad410 (2023). File: `baselines/HEAL/supplementary-data.md`.
**Nguồn AMPR:** `docs/RESULTS_DATA.md`, `docs/PHASE_V6_RESULTS.md`, logs `ensemble_*`.

---

## 1. Fmax @ LT_95 (full test) — tất cả model

| Model | Train data | Backbone | MF | BP | CC |
|---|---|---|---|---|---|
| DeepGO | PDB+SM | 1D-CNN | 0.575 | 0.494 | 0.595 |
| **DeepFRI** | 220K SM + 30K PDB | GCN | 0.626 | 0.540 | 0.612 |
| HEAL-PDB | 30K PDB | Graph-Transformer | 0.691 | 0.566 | 0.654 |
| **HEAL (SOTA)** | PDB + AF2 | Graph-Transformer+CL | **0.749** | **0.594** | **0.687** |
| — AMPR — | | | | | |
| AMPR v3 model-only (DAG) | 30K PDB | ESM-2 3B | 0.550 | 0.458 | 0.496 |
| AMPR v3 single +DIAMOND | 30K PDB | ESM-2 3B | 0.614 | 0.507 | 0.538 |
| AMPR v3 **3-seed ens +D** | 30K PDB | ESM-2 3B | 0.649 | 0.533 | 0.560 |
| AMPR v5 **drop0.4 ens +D** | 30K PDB | ESM-2 3B | **0.650** | **0.539** | 0.554 |
| AMPR v6 650M baseline ens +D | 30K PDB | ESM-2 650M | 0.654 | 0.530 | 0.566 |
| AMPR v6-B (220K→30K) ens +D | 220K→30K | ESM-2 650M | 0.622 | 0.521 | 0.535 |

**Headline AMPR (best-per-branch):** MF **0.654** (v6-650M) / BP **0.539** (v5-drop04) /
CC **0.566** (v6-650M). → AMPR **vượt DeepFRI ở MF** (0.654>0.626), **hoà BP** (0.539≈0.540),
**dưới CC** (0.566<0.612). Tất cả dưới HEAL SOTA. v6-B (pretrain SWISS-MODEL) **kém** baseline
mọi nhánh (xem `PHASE_V6_RESULTS.md`).

---

## 2. macro-AUPR @ LT_95 (full test) — tất cả model

| Model | MF | BP | CC |
|---|---|---|---|
| DeepGO | 0.395 | 0.185 | 0.272 |
| **DeepFRI** | 0.504 | 0.268 | 0.285 |
| HEAL-PDB | 0.571 | 0.263 | 0.347 |
| **HEAL (SOTA)** | **0.698** | **0.345** | **0.468** |
| — AMPR — | | | |
| AMPR v3 model-only (DAG) | 0.524 | 0.275 | 0.350 |
| AMPR v6 650M baseline ens +D | **0.631** | 0.353 | **0.432** |
| AMPR v6-B (220K→30K) ens +D | 0.563 | 0.287 | 0.373 |

**Ghi chú AUPR:** AMPR **thắng DeepFRI ở macro-AUPR cả 3 nhánh** (MF 0.631/0.524 vs 0.504;
CC 0.432 vs 0.285), dù Fmax CC vẫn dưới — AMPR phân hạng term tốt hơn nhưng calibration
ngưỡng kém hơn ở CC. (v3 model-only AUPR từ RESULTS_DATA §2; v6 từ log `ensemble_*`.)

---

## 3. Bảng theo từng bin identity (LT_30..LT_95)

Số đầy đủ per-bin cho Fmax & AUPR: **`docs/RESULTS_DATA.md` §1, §1b, §4** (AMPR v3 + DeepFRI
+ HEAL) và **`docs/PHASE_V6_RESULTS.md` §3** (v6 650M baseline vs v6-B). Baseline gốc per-bin:
HEAL supp Bảng S3.1/S3.2 (`baselines/HEAL/supplementary-data.md`).

Ví dụ MF Fmax theo bin (AMPR v3 ens+D vs DeepFRI vs HEAL):

| Bin | AMPR v3 ens+D | DeepFRI | HEAL |
|---|---|---|---|
| LT_30 | 0.562 | 0.544 | 0.698 |
| LT_40 | 0.571 | 0.552 | 0.702 |
| LT_50 | 0.594 | 0.575 | 0.719 |
| LT_70 | 0.625 | 0.604 | 0.735 |
| LT_95 | 0.649 | 0.626 | 0.749 |

---

## 4. AUROC (chỉ AMPR — baseline không công bố)

AMPR có tính **AUROC_micro**; DeepFRI/HEAL/DeepGO **không báo cáo AUC** trên PDBch nên
**không có đối chứng**. Để tham khảo (AMPR v3 model-only DAG, LT_95; RESULTS_DATA §2):

| | MF | BP | CC |
|---|---|---|---|
| AMPR v3 AUROC_micro | 0.922 | 0.784 | 0.842 |

→ Đúng như cảnh báo §0: AUROC cao đều (~0.78–0.92) và **không phân biệt** được chất lượng
như Fmax/AUPR. **Không nên dùng AUC làm metric so sánh chính** trong luận văn; nếu cần nêu,
chỉ trình bày như chỉ số phụ của riêng AMPR, kèm giải thích vì sao field bỏ AUC.

---

## 5. Khuyến nghị trình bày trong luận văn

- Dùng **Fmax (chính) + macro-AUPR (phụ)** làm bộ metric so sánh — đúng chuẩn DeepFRI/HEAL,
  so chéo được, có nguồn.
- Smin: chỉ so **trong nội bộ AMPR** (biến thể chuẩn hoá), KHÔNG ghép bảng với Smin raw-IC
  của baseline; hoặc tính lại Smin raw-IC nếu muốn so trực tiếp.
- AUC: nêu một câu giải thích "vì sao field dùng AUPR thay AUC trên nhãn thưa" + bảng AUROC
  riêng của AMPR (§4) — thể hiện hiểu biết về metric, không dùng để claim hơn/kém baseline.

---

## Nguồn

- Gu et al., "Hierarchical Graph Transformer with Contrastive Learning for Protein Function
  Prediction", *Bioinformatics* 39(7):btad410 (2023) — baseline Fmax/AUPR/Smin, PDBch.
- Gligorijević et al., "Structure-based protein function prediction using graph convolutional
  networks", *Nature Communications* 12:3168 (2021) — DeepFRI.
- Kulmanov et al., DeepGO, *Bioinformatics* 34(4) (2018).
- AMPR nội bộ: `docs/RESULTS_DATA.md`, `docs/PHASE_V6_RESULTS.md`, `baselines/HEAL/supplementary-data.md`.
