# AMPR-Large: Mở rộng dữ liệu SWISS-MODEL (pretrain → finetune)

**Ngày:** 2026-06-13
**Tác giả:** Nguyen Viet Hung (20224998) + Claude Code (brainstorming)
**Trạng thái:** Design — chờ duyệt → writing-plans

---

## 1. Bối cảnh & động lực

AMPR đã hoàn thiện về kiến trúc (ESM-2 3B + Transformer/attention-pool sequence, GNN trên
contact map thật, PPI có mask, cross-modal fusion, classifier `both`, loss ASL+DAG,
inference DAG-propagation + Diamond ensemble + 3-seed ensemble). Mọi tinh chỉnh kiến trúc
trong cùng paradigm (bigger / label-attention / text / SupCon contrastive) **đã chạm trần**:
val Fmax ~0.76 nhưng test MF ~0.65 — **khoảng cách val→test (tổng quát hoá trên protein
độ tương đồng thấp) là vấn đề cốt lõi còn lại**, không phải kiến trúc.

**Đòn bẩy được chọn:** mở rộng dữ liệu huấn luyện. AMPR hiện chỉ train trên **29,902 chain
PDB** (cluster representatives của nrPDB-GO_2019.06.18). DeepFRI cung cấp sẵn một tập lớn
hơn nhiều: **220,297 chain SWISS-MODEL** (homology models), cùng không gian nhãn
(489 MF / 1,943 BP / 320 CC) và **cùng test set 3,416 chain PDB**.

**Lập luận học thuật quyết định:** file weights baseline trong repo là
`DeepFRI-MERGED_MultiGraphConv_3x512_...` — tức con số DeepFRI mà AMPR đang so
(MF 0.626) **đã được train trên cả 220K SWISS-MODEL**. AMPR-30K hiện tại **đã thắng**
model train-trên-220K đó (MF 0.649 vs 0.626). Cho AMPR ăn cùng lượng dữ liệu → kỳ vọng
vượt rõ, và là phương pháp luận **trùng với chính DeepFRI** nên trích dẫn được, không bị
coi là so sánh không công bằng.

### Giả thuyết kiểm chứng
> Val→test gap của AMPR do thiếu dữ liệu đa dạng, không phải kiến trúc. Mở rộng train lên
> 220K SWISS-MODEL (theo đúng phương pháp DeepFRI) sẽ thu hẹp gap và đẩy test Fmax vượt
> DeepFRI-MERGED ở mọi bin identity.

### Tiêu chí thành công
- **Chính:** test Fmax (cả 3 nhánh, mọi bin LT_30..LT_95) ≥ AMPR-30K hiện tại; mục tiêu
  MF > 0.66, BP > 0.55, CC bắt kịp/vượt DeepFRI (CC hiện đang trail).
- **Phụ:** thu hẹp khoảng cách val→test so với baseline 30K.
- **Kết quả âm vẫn có giá trị:** nếu 220K không vượt ensemble hiện tại → báo cáo "data
  scaling bão hoà do nhiễu homology-model" (vẫn là đóng góp luận văn).

---

## 2. Quyết định thiết kế (đã chốt khi brainstorm)

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Đòn bẩy | Mở rộng dữ liệu train (không fine-tune PLM, không ra Swiss-Prot/AlphaFold) | Generalization lever số 1; né leakage & chi phí fine-tune |
| Nguồn dữ liệu | TFRecords DeepFRI (SWISS-MODEL-GO): cmap + seq + label, ~220,297 chain | Đã có sẵn cấu trúc + nhãn; cùng test set; không rò rỉ |
| Chiến lược | **B: pretrain 220K SWISS-MODEL → finetune 30K PDB** | Xử lý domain shift cấu trúc (test là PDB sạch); narrative transfer-learning |
| Backbone ESM | **ESM-2 650M (1280d)** cho cả 2 giai đoạn | Storage 220K ở 3B (~280GB) quá nặng; 650M ~140GB; 2 stage phải cùng số chiều |
| Kiến trúc | Giữ nguyên (chỉ đổi `seq.d_model` 2560→1280) | Cô lập sạch hiệu ứng data-scaling cho ablation |
| Test set | Giữ nguyên 3,416 PDB test + đủ bin | So sánh trực tiếp DeepFRI/HEAL |

---

## 3. Thành phần

### Thành phần 1 — Trích xuất dữ liệu từ TFRecords *(rủi ro cao nhất → GATE đầu tiên)*

**Đầu vào:** tar.gz TFRecords DeepFRI SWISS-MODEL-GO (~220,297 chain).
**Đầu ra:** với mỗi chain: `sequence` (str), `Cα contact map` nhị phân ngưỡng 10Å (đúng
format GNN đang ăn), `nhãn GO` mf/bp/cc (multi-hot), `chain_id`.

**GATE (việc đầu tiên, chặn mọi việc sau):**
1. Viết script in schema của 1–2 TFRecord example (feature keys + dtype + shape). DeepFRI
   thường dùng các key kiểu `L`, `seq`/`seq_1hot`, `ca_dist_matrix` hoặc `A_ca`,
   `mf_labels`/`bp_labels`/`cc_labels`, `prot_id` — **phải xác nhận trên file thật**.
2. Unit test parse 5 record: kiểm tra shape contact map (L×L), độ dài seq khớp `L`, nhãn
   có đúng độ dài 489/1943/320.
3. **Không viết pipeline tiếp khi schema chưa được xác nhận.**

**Căn nhãn (label alignment):** map index nhãn trong TFRecord ↔ thứ tự term trong
`data/pdbch/go_terms_{mf,bp,cc}.json` hiện có. Verify bằng cách so vài chain xuất hiện ở
cả hai nguồn; build bảng remap nếu thứ tự lệch.

**Split:**
- Test: giữ nguyên 3,416 PDB test (data/pdbch).
- Validation pretrain: SWISS-MODEL valid (DeepFRI cung cấp 24,478) hoặc held-out slice.
- Validation finetune: PDB valid 3,323 (data/pdbch).

### Thành phần 2 — Precompute embeddings (ESM-2 650M)

- Model: `facebook/esm2_t33_650M_UR50D`, frozen, `eval()`.
- Output: residue-level `(L, 1280)` fp16, HDF5 sharded (một dataset/chain hoặc shard theo
  block), dedup theo sequence để tránh tính trùng. Tổng ~140–160GB.
- **Precompute lại cả 30K PDB ở 650M** (rẻ) để 2 giai đoạn cùng số chiều 1280.
- Truncate/sliding-window cho protein > giới hạn token ESM-2 (1022 residue + BOS/EOS);
  giữ `max_seq_len: 1000` như config hiện tại.
- Contact map: trích từ TFRecord → append vào `cmap_all.h5` (cùng format GNN consume).
- **PPI:** chain SWISS-MODEL phần lớn không có STRING PPI → `ppi_mask=0`; adaptive gating
  đã xử lý missing modality. KHÔNG precompute PPI mới cho SWISS-MODEL.
- GO embedding SapBERT 896d: giữ nguyên (cấp label-space, không phụ thuộc protein).

### Thành phần 3 — Kiến trúc: GIỮ NGUYÊN

AMPR v3 y hệt hiện tại. Thay đổi DUY NHẤT: `model.seq.d_model` 2560 → 1280 (vì đổi ESM 3B
→ 650M). Mọi thứ khác (Transformer layers, GNN, fusion, classifier, loss) không đổi →
cô lập sạch hiệu ứng data-scaling. Config mới: `configs/{mf,bp,cc}_v6_swissmodel.yaml`.

### Thành phần 4 — Train hai giai đoạn (chiến lược B)

- **Stage 1 — pretrain (220K SWISS-MODEL):** train full AMPR, early-stop trên SWISS-MODEL
  valid; lưu checkpoint `checkpoints/{branch}_v6_pretrain/`.
- **Stage 2 — finetune (30K PDB):** load checkpoint Stage 1, train tiếp trên 30K PDB
  (cấu trúc sạch), LR thấp (~1e-4), early-stop trên PDB valid 3,323; lưu
  `checkpoints/{branch}_v6_finetune/`.
- Giữ ASL (γ⁻=4, γ⁺=0, clip=0.05) + DAG (λ=0.5) + dropout/weight_decay từ cấu hình tốt
  nhất hiện tại. Mỗi nhánh MF/BP/CC một model riêng.

### Thành phần 5 — Evaluation: GIỮ NGUYÊN pipeline

Test 3,416 + đủ bin LT_30..LT_95. Áp dụng DAG propagation → Diamond ensemble →
threshold calibration (trên val) → 3-seed ensemble (đều đã có).

**Bảng so sánh chính:**

| Model | Train data | MF | BP | CC |
|---|---|---|---|---|
| DeepFRI-MERGED (baseline) | 220K SWISS-MODEL + PDB | 0.626 | ~0.54 | ~0.61 |
| AMPR-PDB-30K (hiện tại) | 29,902 PDB | 0.649 | 0.539 | trail |
| AMPR pretrain-only-220K | 220K SWISS-MODEL | ? | ? | ? |
| **AMPR-B (pretrain→finetune)** | 220K → 30K | ? | ? | ? |

### Thành phần 6 — Thí nghiệm cho luận văn

- **Đường cong data-scaling:** train trên 30K / 60K / 120K / 220K → biểu đồ Fmax theo
  log(lượng dữ liệu). Figure trung tâm cho chương kết quả.
- Hiệu ứng theo nhánh: kỳ vọng BP hưởng lợi nhất (thêm positive cho term hiếm long-tail).
- (Tuỳ chọn rẻ) chạy A (union 250K một lượt = "MERGED-parity") cạnh B để so chiến lược.

### Thành phần 7 — Rủi ro & tiêu chí dừng

| Rủi ro | Mitigate |
|---|---|
| Schema TFRecord lệch dự đoán | GATE inspect + unit test ở Thành phần 1 trước khi code tiếp |
| Index nhãn lệch go_terms.json | Verify + remap, test trên chain xuất hiện 2 nguồn |
| Cấu trúc SWISS-MODEL nhiễu hại GNN | Chính là lý do có Stage 2 finetune trên PDB sạch |
| Storage/thời gian precompute | Đã hạ ESM xuống 650M; dedup theo sequence |
| 220K không vượt ensemble hiện tại | Báo cáo kết quả âm "scaling bão hoà do nhiễu model" |

---

## 4. Nguồn học thuật

- **DeepFRI** — Gligorijević V. et al. "Structure-based protein function prediction using
  graph convolutional networks." *Nature Communications* 12:3168 (2021).
  → nguồn 220K SWISS-MODEL, contact-map GNN, weights "MERGED" = PDB+SWISS-MODEL.
- **HEAL** — Gu Z. et al. *Bioinformatics* 39(7):btad410 (2023).
  → định nghĩa split PDBch + bin identity LT_30..LT_95.
- **ESM-2** — Lin Z. et al. "Evolutionary-scale prediction of atomic-level protein
  structure with a language model." *Science* 379:1123–1130 (2023). → backbone 650M.
- **ESM-1b / transfer** — Rives A. et al. *PNAS* 118(15):e2016239118 (2021).
  → tinh thần pretrain→finetune trên representation protein.
- **Asymmetric Loss** — Ridnik T., Ben-Baruch E. et al. *ICCV 2021* (arXiv:2009.14119).
  → loss long-tail multi-label (đã dùng, giữ nguyên).
- **SWISS-MODEL** — Waterhouse A. et al. *Nucleic Acids Research* 46:W296–W303 (2018).
  → nguồn homology models của tập 220K.

---

## 5. Phạm vi (scope)

**Trong phạm vi:** parse TFRecords; precompute ESM-2 650M; pretrain→finetune 2 giai đoạn;
evaluation đầy đủ + bảng so sánh; data-scaling curve.

**Ngoài phạm vi (YAGNI):** fine-tune PLM (LoRA); dữ liệu Swiss-Prot/AlphaFold ngoài
DeepFRI; thay đổi kiến trúc module; chiến lược C (quality weighting). Chiến lược A (union)
là tuỳ chọn rẻ, không bắt buộc.
