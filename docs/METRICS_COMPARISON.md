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

- **AUC phụ thuộc LINEAGE benchmark — đây là điểm dễ sai nhất.** Có HAI dòng benchmark
  riêng biệt, **không trộn số được**:
  - **Lineage A — PDBch** (DeepFRI 2021, HEAL 2023): cấu trúc **thực nghiệm PDB**, train ~30K,
    test 3,416, term 489/1943/320. Metric **Fmax/AUPR/Smin — KHÔNG có AUC** (nhãn thưa → ROC-AUC
    bị thổi phồng, vô nghĩa). **AMPR thuộc lineage này** → §1–§4.
  - **Lineage B — human-AF/Struct2GO** (Struct2GO 2023, StructSeq2GO, GAT-GO): protein **người**
    + cấu trúc **AlphaFold2**, term 273/809/298, split 8:1:1. Metric **Fmax/AUC/AUPR — CÓ AUC**.
    → §6 (tham khảo, **không so trực tiếp với AMPR**).
  - Cùng một model (DeepFRI) cho **số khác nhau** giữa 2 lineage (MFO 0.542 ở B vs MF 0.626 ở A)
    → tuyệt đối không ghép chung bảng.
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

## 6. Lineage B — benchmark human-AF / Struct2GO (THAM KHẢO, không so với AMPR)

> **CẢNH BÁO:** Bảng này **KHÁC benchmark** với §1–§4. Protein **người** + cấu trúc
> **AlphaFold2** (≠ PDBch dùng cấu trúc thực nghiệm). Số GO term khác: **MFO 273, BPO 809,
> CCO 298** (PDBch: 489/1943/320). Dataset 20,395 protein, split 8:1:1. **KHÔNG ghép số này
> với AMPR/DeepFRI-PDBch.** Đưa vào đây chỉ để khảo sát literature + thấy lineage này dùng AUC.

**StructSeq2GO Table 3 — "Experimental results on human protein data"** (nguồn: bảng trong
paper *A unified graph-based approach…*, ScienceDirect S1476927125002701). Giá trị chép
nguyên từ paper:

| Model | BPO Fmax | BPO AUC | BPO AUPR | CCO Fmax | CCO AUC | CCO AUPR | MFO Fmax | MFO AUC | MFO AUPR |
|---|---|---|---|---|---|---|---|---|---|
| BLAST (1990) | 0.339 | 0.577 | 0.489 | 0.441 | 0.563 | 0.269 | 0.411 | 0.623 | 0.461 |
| DeepGO (2018) | 0.327 | 0.639 | 0.571 | 0.589 | 0.695 | 0.448 | 0.404 | 0.760 | 0.625 |
| DeepGOA (2019) | 0.385 | 0.698 | 0.622 | 0.629 | 0.757 | 0.500 | 0.477 | 0.820 | 0.710 |
| DeepFRI (2021) | 0.425 | 0.732 | 0.635 | 0.624 | 0.779 | 0.641 | 0.542 | 0.881 | 0.763 |
| GAT-GO (2022) | 0.462 | 0.586 | 0.512 | 0.647 | 0.831 | 0.681 | 0.633 | 0.912 | 0.776 |
| Struct2GO (2023) | 0.481 | 0.873 | 0.661 | 0.658 | 0.942 | 0.763 | **0.701** | **0.969** | **0.796** |
| StructSeq2GO | 0.485 | 0.764 | 0.688 | 0.681 | 0.939 | 0.763 | 0.663 | 0.891 | 0.702 |

**Lưu ý đọc bảng:** trên benchmark này, **Struct2GO (2023)** vẫn dẫn MFO Fmax (0.701) và AUC
(0.969); StructSeq2GO chỉ hơn ở BPO Fmax và một số cột AUPR. Đây là lý do AUC ở lineage B
**bão hoà cao** (0.88–0.97) → khó phân biệt model, củng cố lý do lineage A (PDBch) bỏ AUC.

**Nếu muốn AMPR so được với lineage B:** phải eval lại AMPR trên chính dataset human-AF của
Struct2GO (cấu trúc AF2 người, 273/809/298 term) — là một thí nghiệm riêng, ngoài phạm vi hiện tại.

---

## 7. Lineage A — paper MỚI hơn (post-HEAL, cùng PDBch → so được với AMPR)

Các phương pháp 2024–2025 vẫn đánh giá trên **đúng PDBch test set** (cấu trúc thực nghiệm,
Fmax/AUPR/Smin) — **ghép thẳng bảng với AMPR được**. Đây là các đối thủ cập nhật cho luận văn.

### 7.1 ProteinRPN (2024) — có số đã verify

Region-proposal trên protein graph. Nguồn: arXiv 2409.00610 (Sep 2024), Bảng 1, **cùng
dataset DeepFRI/PDBch**. Số chép nguyên:

| Model | MF Fmax | BP Fmax | CC Fmax | MF AUPR | BP AUPR | CC AUPR |
|---|---|---|---|---|---|---|
| DeepFRI | 0.625 | 0.540 | 0.613 | 0.495 | 0.261 | 0.274 |
| HEAL | 0.708 | 0.581 | 0.673 | 0.630 | 0.298 | 0.415 |
| **ProteinRPN** | **0.754** | **0.618** | **0.691** | **0.683** | **0.344** | **0.459** |

→ ProteinRPN là **SOTA mới trên PDBch MF** (0.754 > HEAL self-report 0.749).

> **⚠️ Cảnh báo nhất quán nguồn:** số HEAL/DeepFRI mà ProteinRPN *tự eval lại* **lệch** so với
> HEAL self-report (HEAL MF: 0.708 ở ProteinRPN vs **0.749** ở HEAL supp; AUPR 0.630 vs 0.698).
> → Khi dựng bảng luận văn, **ghi rõ mỗi số lấy từ paper nào**, và lý tưởng dùng **một nguồn
> nhất quán cho mỗi baseline** (ưu tiên HEAL supp cho DeepFRI/HEAL — §1/§2). Đừng trộn số
> reproduced với số self-reported trong cùng một cột.

### 7.2 Các paper Lineage-A khác (chưa lấy được số chính xác — cần đọc PDF)

| Paper | Nơi đăng | Ý tưởng | Trạng thái số |
|---|---|---|---|
| **DPFunc** | Nat. Commun. 16:70 (2024/25) | domain-guided + structure graph | paywall — số trong paper/supp, **chưa verify** |
| **GOBeacon** | 2024/25 (PMC12183117) | ensemble + contrastive | cần xác nhận đúng PDBch |
| **BeProf** | Brief. Bioinform. bbae050 (2024) | **bộ benchmark** (không phải 1 model) | nguồn tổng hợp số nhiều method |

**Khuyến nghị:** lấy số DPFunc từ Nature supplementary (Wang et al. 2024) và đối chiếu với
BeProf — hai nguồn này nhiều khả năng có bảng PDBch đầy đủ. Tôi có thể trích nếu bạn tải được
PDF (Nature/Brief.Bioinform. chặn fetch tự động).

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
- **Lineage B (human-AF, KHÁC benchmark):** Jiao et al., "Struct2GO", *Bioinformatics* 39(10):btad637
  (2023); "A unified graph-based approach… (StructSeq2GO)", *Comput. Biol. Chem.* (2025),
  ScienceDirect S1476927125002701; Lai & Xu, "GAT-GO" (2022); Zhou et al., DeepGOA (2019).
  Dataset: 20,395 human protein, AlphaFold2, term MFO 273 / BPO 809 / CCO 298.
- AMPR nội bộ: `docs/RESULTS_DATA.md`, `docs/PHASE_V6_RESULTS.md`, `baselines/HEAL/supplementary-data.md`.
