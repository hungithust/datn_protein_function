# AMPR v4 — Breakthrough Improvement Design (Spec)

**Ngày:** 2026-06-11 · **Tác giả:** Nguyen Viet Hung (20224998) + Claude Code
**Trạng thái:** Design approved → chờ implementation plan
**Builds on:** [ARCHITECTURE.md](../../ARCHITECTURE.md) (v3), [DESIGN_RATIONALE.md](../../DESIGN_RATIONALE.md),
[RESULTS_DATA.md](../../RESULTS_DATA.md), spec `2026-06-01-ampr-h200-migration-design.md`

---

## 0. Bối cảnh & vấn đề cốt lõi

GVHD chưa hài lòng vì AMPR v3 vẫn dưới SOTA hiện đại (2023–2025) khi so ở bin headline
**LT_95** (bin cao nhất, dùng để so với các paper khác trong thesis):

| Branch | AMPR+D (LT_95) | val Fmax | HEAL | DPFunc | TAWFN | MAEF-GO | GOBoost |
|---|---|---|---|---|---|---|---|
| MF | 0.614 | 0.760 | 0.747 | 0.731 | 0.762 | 0.787 | 0.787 |
| BP | 0.507 | 0.673 | 0.595 | 0.606 | 0.628 | 0.652 | 0.659 |
| CC | 0.538 | 0.682 | 0.687 | 0.689 | 0.693 | 0.720 | 0.745 |

**Chẩn đoán (chính):** Vấn đề **không** phải "test khó intrinsic" như [DESIGN_RATIONALE §8](../../DESIGN_RATIONALE.md)
lập luận — vì SOTA đạt 0.74–0.79 *trên chính test set đó*. val MF=0.76 ≈ SOTA nhưng test=0.556 →
**gap val→test ~0.20 chính là generalization gap**: model overfit phân phối high-similarity, không
generalize xuống low-identity như SOTA. Đây là chỗ cần đột phá.

**Nguyên nhân gốc (giả thuyết testable = chất liệu thesis):**
1. Backbone ESM-2 3B **frozen hoàn toàn** + chỉ train head ~60M → representation generic, không định hình
   theo chức năng. SOTA dùng contrastive/fine-tune/domain-prior để generalize.
2. Nhánh structure (GNN contact-map) **chưa từng được ablate** — không rõ có đóng góp thực không.
3. Không có cơ chế localize active-site/domain → MF (phụ thuộc residue-level) yếu nhất so với SOTA.

**Lỗi cần sửa:** baseline DeepFRI ghi MF LT_95 = **0.759** trong [RESULTS_DATA.md](../../RESULTS_DATA.md)
nhưng paper gốc (Gligorijević 2021) báo cáo MF Fmax ≈ **0.625–0.631** (con số 0.759 trùng dải HEAL/TAWFN
→ nhiều khả năng lấy nhầm dòng). Cần verify lại provenance trước khi viết thesis.

---

## 1. Phạm vi & nguyên tắc

- **4 cải tiến**, mỗi cái là **module bật/tắt độc lập qua config** trên pipeline v3 hiện có → ra bảng
  ablation sạch, attribute được từng đóng góp. Baseline v3 giữ nguyên để regression.
- Ưu tiên theo **impact / setup-cost** vì quỹ GPU còn ~24h (8×H200). Module không cần precompute mới
  được ưu tiên; module precompute nặng → Future work.
- **Trung thực:** mỗi module phải cho đóng góp **đo được**; kết quả âm vẫn ghi nhận (hợp lệ thesis).

### Bảng ưu tiên (24h budget)

| Module | Precompute mới | Train cost | Impact | Quyết định |
|---|---|---|---|---|
| D0.2 Sửa baseline DeepFRI | 0 GPU | 0 | chính trực | **Làm ngay** |
| D0.1/D0.3 Diagnostic ablation | eval-only masking | ~chục phút | cao | **Làm** (eval-only) |
| A — Multi-label SupCon | **Không** | = baseline | **cao** (đánh gap) | **Ưu tiên 1** |
| C — Term-conditioned pooling | Không | = baseline; BP nặng | cao + interpretability | **Ưu tiên 2** (MF+CC) |
| B — Domain-guided (DPFunc) | SIFTS+InterPro (nặng) | + train | cao (MF) | **Future work** |

---

## 2. Phase 0 — Diagnostic & sửa baseline (gần như 0 GPU)

### D0.1 — Modality ablation (eval-only masking)
Trên checkpoint v3 đã train sẵn, **zero-out từng modality tại inference** (không retrain) và đo Fmax mỗi
nhánh × bin: `full` / `−structure` / `−ppi` / `seq-only`. Trả lời câu hỏi *"nhánh structure đóng góp thực
không?"* — bảng ablation thesis đang thiếu.
> **Caveat rigor:** eval-only masking là cận dưới (model không được train để thiếu modality). Nếu còn GPU,
> retrain ablation chính xác hơn — ghi rõ phương pháp nào dùng trong thesis.

### D0.2 — Sửa số baseline DeepFRI
Verify MF/BP/CC LT_* của DeepFRI từ paper gốc (hoặc HEAL Table S3.2, ghi rõ nguồn nào). Cập nhật
[RESULTS_DATA.md §1/§4](../../RESULTS_DATA.md) + report. **0 GPU.**

### D0.3 — Log gate/attention weight theo bin
Eval-only: log trọng số fusion (attention/gate) per modality theo identity bin. Kiểm tra giả thuyết PPI
weight blow-up ở LT_30 cold-start (khuếch đại noise khi thiếu PPI).

---

## 3. Module A — Multi-label Supervised Contrastive (①) · ƯU TIÊN 1

**Mục tiêu:** đánh thẳng vào generalization gap — align hình học không gian `z` với độ tương đồng chức năng.
**Nguồn:** cơ chế contrastive low-identity của HEAL (Gu et al., *Bioinformatics* btad410, 2023); biến thể
supervised-contrastive multi-label (Khosla et al. SupCon, NeurIPS 2020).

### Thiết kế
- Thêm projection head `g: z(512) → 128`, L2-normalize → `p_i`.
- Positive mềm theo nhãn: `w_ij = Jaccard(labels_i, labels_j)` (tùy chọn threshold `jaccard_thr`).
- Weighted SupCon loss:
  ```
  L_CL = - Σ_i (1/Σ_{j≠i} w_ij) Σ_{j≠i} w_ij · log( exp(sim(p_i,p_j)/τ) / Σ_{k≠i} exp(sim(p_i,p_k)/τ) )
  ```
  `sim` = cosine, `τ = 0.1`.
- Tổng loss: `L = L_ASL + λ_dag·L_DAG + λ_cl·L_CL`. Sweep `λ_cl ∈ {0.1, 0.5, 1.0}`.
- Cần batch đủ lớn để có negative phong phú (H200 cho phép batch lớn).

### Interface / file
- File mới `ampr/training/contrastive.py` — class `MultiLabelSupConLoss(temp, jaccard_thr)`.
- Sửa `ampr/training/loss.py` (AMPRLoss tổng hợp thêm `λ_cl·L_CL`), `trainer_v3.py` (forward projection head).
- Config: `contrastive: {enabled: bool, weight: λ_cl, temp: 0.1, jaccard_thr: 0.0}`.

### Rủi ro & mitigation
- **ASL × CL có lịch sử collapse** (xem [DESIGN_RATIONALE §6](../../DESIGN_RATIONALE.md)). Monitor probs/grad
  mỗi epoch; nếu collapse → fallback BCE+pos_weight + CL.
- Projection head chỉ dùng khi train (discard khi inference) — không đổi logits path.

---

## 4. Module C — Term-conditioned pooling (💡) · ƯU TIÊN 2 (MF+CC trước)

**Mục tiêu:** localize active-site cho MF + sinh **bản đồ residue→function** (interpretability figure).
**Nguồn cảm hứng:** GO-as-query của TransFew (Boadu & Cheng, *Bioinformatics Advances* vbae120, 2024) +
localization của ProteinRPN (Mitra et al., arXiv:2409.00610, 2024). Cách vận hành trên frozen ESM-2 residue
field + blend với head "both" là phần riêng của dự án.

### Thiết kế
- **Thay** attention-pool ở nhánh seq bằng cross-attention term-conditioned:
  - Query = GO-term embedding `e_c` (từ `go_emb`, project → 512), `c = 1..C`.
  - Key/Value = residue ESM-2 features (project → 512), `L` residue.
  - Cross-attention → vector per-term `v_c (C, 512)` cho mỗi protein.
  - `logit_tc[c] = v_c · e_c` (hoặc qua scoring vector chung).
  - Attention map `(C, L)` → lưu để vẽ figure residue→term.
- Nhánh struct + ppi **giữ nguyên** → fuse thành `z` global → head "both" → `logit_both`.
- **Blend:** `logits = 0.5 · logit_both + 0.5 · logit_tc`.

### Interface / file
- File mới `ampr/models/term_conditioned_pool.py` — `TermConditionedPooling(d=512, n_heads)`.
- Sửa `ampr/models/ampr_v3.py` (rẽ nhánh pooling), config `seq_pool: {type: attention | term_conditioned}`.

### Rủi ro & mitigation
- **Compute/memory BP** (C=1943 × L residue × batch) nặng → **làm MF (489) + CC (320) trước**; BP để sau
  với mitigation (chunk term, shared multi-head low-rank).
- Blend 0.5/0.5 là điểm khởi đầu; có thể học hệ số blend nếu cần (ghi nhận, không bắt buộc).

---

## 5. Module B — Domain-guided structure (②) · FUTURE WORK

**Mục tiêu:** prior domain để attention tập trung vùng chức năng → vực MF. **Nguồn:** DPFunc (Wang et al.,
*Nat Commun* 16:70, 2025; code github.com/CSUBioGroup/DPFunc).

### Thiết kế (sẵn sàng chạy khi có lại GPU)
- **Precompute (đường nhẹ, đã chốt):** SIFTS map PDB chain → UniProt → tải **InterPro annotation
  precomputed** → per-residue domain membership → bảng embedding theo InterPro id. Lưu
  `data/domains/domain_feats.h5`. *Không* tải 50GB / *không* chạy full InterProScan.
  - Fallback nếu mapping thiếu: `hmmscan` với Pfam-A (~1.5GB) trực tiếp trên FASTA (~vài giờ, 192 core).
- **Kiến trúc:** inject domain embedding vào node feature GNN + bias attention pooling (CAM-style: residue
  trong domain chức năng được boost).
- Protein không map được → domain feature = 0 (graceful, giống PPI mask). **Báo cáo coverage.**
- File `scripts/precompute_domains.py`; config `domain: {enabled, source, emb_dim}`.

**Lý do defer:** pipeline precompute (network fetch SIFTS/InterPro, mapping, build feature) ăn nhiều
wall-clock trước khi train được → không hợp budget 24h. Thiết kế giữ nguyên để chạy sau.

---

## 6. Ablation matrix & config

Mỗi cell = 1 run/nhánh. Toggle độc lập:

```
v3 baseline · +A(SupCon) · +C(term-cond) · +A+C(v4)        × {MF, CC}     [trong 24h]
v3 baseline · +A(SupCon)                                    × {BP}         [trong 24h, BP chỉ A]
+B(domain), +all                                            × {MF,BP,CC}   [future work]
```

Config-driven theo đúng pattern v3 (`configs/{mf,bp,cc}_v4_*.yaml`), chỉ thêm các block
`contrastive`, `seq_pool`, `domain`.

---

## 7. Success criteria (trung thực, hướng thesis)

- **Primary:** thu hẹp gap val→test; nâng **LT_95 Fmax** so với AMPR+D hiện tại. Target thực tế:
  MF +0.05–0.10, BP +0.03–0.06, CC +0.05–0.10. **Stretch:** chạm HEAL ở ≥1 nhánh.
- Mỗi module cho đóng góp **đo được** (âm vẫn ghi nhận).
- **Deliverable mới:** figure residue→function (Module C) — điểm cộng khi bảo vệ.
- Bảng modality ablation (Phase 0) + baseline DeepFRI đã sửa.

---

## 8. Testing

- **Unit:** shape + gradient flow từng module mới (`MultiLabelSupConLoss`, `TermConditionedPooling`) trên
  synthetic batch; verify `.grad` không None.
- **Integration:** train tiny-subset → loss giảm, **không collapse** (canh ASL×CL, probs≠0).
- **Regression:** config baseline v3 vẫn reproduce val Fmax 0.7525 (MF).
- Chạy pytest qua PowerShell + Anaconda python (`python -m pytest tests/`), theo [CLAUDE.md](../../../CLAUDE.md).

---

## 9. Execution order (24h, 8×H200 song song)

1. **H0 (0 GPU):** sửa số DeepFRI (D0.2); viết `contrastive.py` + unit test; chạy eval-only modality
   masking + gate logging (D0.1/D0.3) trên checkpoint v3 sẵn có.
2. **H0–8:** Module A — launch sweep `λ_cl ∈ {0.1,0.5,1.0} × {MF,BP,CC}` (≤9 run/8 GPU; MF/CC xong nhanh,
   BP lâu nhất). Eval mọi bin + DIAMOND ensemble. Chọn `λ_cl` tốt nhất theo val.
3. **H8–24:** Module C term-conditioned **MF+CC** (code + train + xuất attention map figure). Nếu còn thời
   gian: +A+C combined cho MF+CC.
4. **Module B** → Future work (đã có thiết kế §5, chạy khi có lại GPU).

---

## 10. Nguồn tham khảo

- HEAL — Gu et al., *Bioinformatics* btad410, 2023 (contrastive, low-identity generalization).
- SupCon — Khosla et al., NeurIPS 2020 (supervised contrastive).
- TransFew — Boadu & Cheng, *Bioinformatics Advances* vbae120, 2024 (GO-as-query cross-attention).
- ProteinRPN — Mitra et al., arXiv:2409.00610, 2024 (functional-residue localization).
- DPFunc — Wang et al., *Nat Commun* 16:70, 2025 (domain-guided structure).
- GOBoost — Zhang et al., *Bioinformatics* btaf267, 2025 (long-tail; tham khảo Future work).
- PhiGnet — *Nat Commun* s41467-024-50955-0, 2024 (evolutionary couplings; tham khảo Future work).
- DeepFRI — Gligorijević et al., *Nat Commun* 2021 (baseline cần verify số).
