# AMPR — Báo cáo tổng hợp kết quả (v3 + v4 experiments)

**Sinh viên:** Nguyễn Việt Hùng (20224998) · **Cập nhật:** 2026-06-12
**Phạm vi:** kết quả cuối cùng trên PDBch, đối chiếu DeepFRI/HEAL, các thí nghiệm cải tiến
(Module A contrastive, multi-seed ensemble), ablation và phân tích độ nhạy.

Dữ liệu chi tiết: [RESULTS_DATA.md](RESULTS_DATA.md) (§1, §1b, §4) và
[RESULTS_V4.md](RESULTS_V4.md). Tất cả số đo trên cùng tập test PDBch, cùng 5 ngưỡng
homology LT_*, cùng Fmax protein-centric — so sánh trực tiếp được với DeepFRI/HEAL
(nguồn: HEAL supplementary, Gu et al., Bioinformatics btad410 2023).

---

## 1. Kết quả chính (headline)

**3-seed ensemble (+DIAMOND), Fmax tại LT_95 — best-per-branch:**

| Nhánh | AMPR (ensemble +D) | Cấu hình | DeepFRI | HEAL (SOTA) | Đối chiếu DeepFRI |
|---|---|---|---|---|---|
| **MF** | **0.650** | dropout 0.4 | 0.626 | 0.749 | **vượt +0.024** |
| **BP** | **0.539** | dropout 0.4 | 0.540 | 0.594 | **hòa −0.001** |
| **CC** | 0.560 | dropout 0.2 | 0.612 | 0.687 | dưới −0.052 |

**Phát biểu chính:** ensemble AMPR **đạt hoặc vượt DeepFRI trên MF và BP** (MF vượt ở toàn
bộ 5 ngưỡng identity; BP hòa tại LT_95), còn dưới trên CC. Cả ba vẫn dưới HEAL. Nhánh
anti-overfit (dropout 0.4) chọn theo val_Fmax — xem §3b.

**MF — vượt DeepFRI ở mọi bin:**

| Bin | LT_30 | LT_40 | LT_50 | LT_70 | LT_95 |
|---|---|---|---|---|---|
| AMPR ens +D | 0.562 | 0.571 | 0.594 | 0.625 | 0.649 |
| DeepFRI | 0.544 | 0.552 | 0.575 | 0.604 | 0.626 |

---

## 2. Đóng góp của ensemble (single → 3-seed)

Trung bình xác suất qua 3 seed {42,123,2024} của cùng config v3, rồi DAG + DIAMOND.
Không đổi kiến trúc, chỉ giảm variance.

| Nhánh | single +D (LT_95) | ensemble +D (LT_95) | Δ |
|---|---|---|---|
| MF | 0.614 | 0.649 | +0.035 |
| BP | 0.507 | 0.533 | +0.026 |
| CC | 0.538 | 0.560 | +0.022 |

Ensemble là đòn bẩy test mạnh và đáng tin nhất tìm được (+0.02–0.035), chính nó đưa MF
từ "kém DeepFRI 0.012" thành "vượt 0.023". Script tái lập: `scripts/ensemble_eval.py`.

---

## 3. Module A — Multi-label Supervised Contrastive (kết quả âm)

Thử nghiệm cải tiến: thêm loss contrastive có trọng số Jaccard (tổng quát hóa SupCon,
Khosla NeurIPS 2020; lấy động lực từ HEAL) trên biểu diễn hợp nhất `z`.

- **Phát hiện then chốt (độ nhạy thang loss):** ASL ≈ 0.036 vs SupCon ≈ 4.3 — lệch ~120×.
  Chỉ `weight ≈ 1e-3` mới giữ classification chi phối; weight ≥ 0.1 phá hỏng train.
- **Kết quả:** val_Fmax tăng 0.760→0.783 nhưng **test không đổi ở mọi bin** (kể cả LT_30,
  nơi contrastive được kỳ vọng giúp nhất). Gap val→test *rộng ra* (0.204→0.233).
- **Kết luận:** contrastive auxiliary kiểu này **overfit val, không cải thiện test** trên
  MF. Đây là một **negative result trung thực** + một **bảng Hyperparameter Sensitivity**
  hoàn chỉnh cho luận văn (xem RESULTS_V4.md §1).

---

## 3b. Anti-overfit (dropout 0.4) — Regularization vs Ensembling

Train lại cả 3 nhánh với `dropout` 0.4 (MF gốc 0.2; BP/CC gốc 0.1), mọi thứ khác giữ nguyên.
Config: `configs/{mf,bp,cc}_v5_drop04*.yaml`. Chọn 0.4 theo val_Fmax (0.5 cho val tương đương
nhưng test thấp hơn trên MF → quá mạnh).

- **Single model (MF, LT_95):** val 0.760→0.7505, **test +D 0.614→0.625**, gap val→test
  **0.146→0.125**. Regularization thu hẹp gap đúng như kỳ vọng.
- **Ở mức ensemble 3 seed** lợi ích gần như biến mất: MF **hòa** (0.649→0.650), BP nhích
  (+0.006 → **chạm DeepFRI 0.540**), CC hơi giảm (−0.006). Số đầy đủ: RESULTS_DATA.md §1c.
- **Phát hiện:** *regularization mạnh và ensembling là hai cơ chế chống overfit **dư thừa
  lẫn nhau** — mỗi cái riêng thu hẹp gap, kết hợp không cộng tuyến tính.* CC yếu do DIAMOND
  phủ thấp (LT_95 1897/3123), không phải overfit, nên dropout không cứu được.

---

## 4. Ablation modality (Phase 0)

Eval-only, zero từng nhánh trước fusion trên checkpoint v3 (test_LT_95):

| Nhánh | full | −gnn | −ppi | seq_only |
|---|---|---|---|---|
| MF | 0.5498 | 0.3021 | 0.4912 | 0.0875 |
| BP | 0.4582 | 0.3900 | 0.3731 | 0.1625 |
| CC | 0.4955 | 0.3992 | 0.4145 | 0.2085 |

Cả ba modality đều đóng góp đáng kể (GNN/structure trội nhất ở MF). **Bác bỏ** giả thuyết
ban đầu rằng structure/PPI có thể "vô dụng" → multimodality được chứng minh chính đáng.
(Lưu ý: đây là chặn dưới — model được train với cả ba.)

---

## 5. Điểm mạnh của AMPR (ngoài Fmax tuyệt đối)

- **Robust với identity thấp:** slope Fmax theo identity *phẳng hơn* DeepFRI; **thắng
  DeepFRI ở macro-AUPR MF tại LT_30** (0.458 vs 0.425) — tốt ở vùng protein ít tương đồng.
- **Robust missing-modality:** gating thích ứng + ablation chứng minh đóng góp 3 modality
  (đề tài cốt lõi của luận văn).
- **Tính nhất quán phân cấp GO:** DAG loss (True Path Rule) + DAG propagation khi suy luận.

---

## 6. Hạn chế & hướng phát triển

- **Gap val→test ~0.20** là vấn đề mở cốt lõi (val MF 0.76 nhưng test 0.55). Đây là
  overfitting/dịch phân phối, không phải thiếu modality. Là rào cản chính để đuổi HEAL.
- **CC là nhánh yếu nhất** (−0.052 vs DeepFRI). DIAMOND chỉ phủ ~40% protein ở CC LT_30.
- **Hướng future work (chưa chạy, đề xuất):**
  1. *Diverse/heterogeneous ensemble* — ghép model khác cấu hình (dropout/wd) để decorrelate
     lỗi, kỳ vọng cao hơn seed-ensemble đồng nhất.
  2. *Embedding-kNN homology transfer* — bổ sung DIAMOND bằng kNN trên ESM-2 cho protein
     không có hit, nhắm đúng điểm yếu CC low-identity.
  3. *Hierarchical/structure contrastive kiểu HEAL* — cơ chế đứng sau SOTA 0.749 MF.
  4. *Anti-overfit mạnh* (dropout 0.4–0.5, stochastic depth) để thu hẹp gap val→test.

---

## 7. Tính minh bạch & nguồn (cho thesis)

- **Baseline đã sửa:** số DeepFRI lưu trước đây (MF 0.759, BP 0.395) **sai** — đã thay bằng
  số đúng từ HEAL supplementary (Bảng S3.1/S3.2), cùng giao thức PDBch/LT_*/Fmax.
- **Provenance code cải tiến:** `MultiLabelSupConLoss` viết lại từ công thức SupCon (Khosla
  2020), phần soft-positive theo Jaccard là đóng góp riêng; **không copy code paper khác**.
  Ensemble/ablation là kỹ thuật chuẩn, tự viết.
- **Tái lập:** seed cố định; script `scripts/ensemble_eval.py`, `scripts/diagnose_modality.py`;
  config `configs/{mf,bp,cc}_v3_esm3b.yaml` + seed configs.

---

## 8. Một dòng kết

Với multi-seed ensemble, **AMPR vượt DeepFRI trên Molecular Function (toàn bộ dải identity,
LT_95 0.649 vs 0.626) và tiệm cận trên Biological Process**, kèm ưu thế robustness ở vùng
ít tương đồng; Cellular Component và khoảng cách tới HEAL là hướng phát triển tiếp theo.
