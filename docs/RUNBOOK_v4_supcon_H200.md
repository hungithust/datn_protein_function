# Runbook — AMPR v4 Module A (SupCon) on 8×H200

**Mục đích:** chạy phần **thực thi GPU** của plan
[2026-06-11-ampr-v4-phase0-and-contrastive.md](superpowers/plans/2026-06-11-ampr-v4-phase0-and-contrastive.md)
trên server hackathon (8×H200). Phần viết code (Task 1–8) đã được làm local & merge — runbook này
giả định repo trên server đã có `ampr/training/contrastive.py`, hook `return_z/ablate`, và các config
`configs/{mf,cc,bp}_v4_supcon.yaml`.

**Tiền đề:**
- Workdir `/raid/team/datn` (alias `/workspace`), NGC image `nvcr.io/nvidia/pytorch:24.10-py3`.
- Embeddings/labels/cmap đã có tại `data/` (ESM-2 3B residue h5, go_emb_*_v2, cmap_all.h5, ppi_deepgo, splits, labels, dag).
- Checkpoint v3 baseline có sẵn (cho bước ablation): `checkpoints/{mf,bp,cc}_v3_esm3b/best.pt`.
- Quỹ GPU ~24h. MF/CC nhanh nhất, BP là long-pole.

---

## 0. Setup (mỗi phiên)

```bash
cd /raid/team/datn
git pull
tmux new -s v4a            # mỗi job dài chạy trong tmux
python -c "import ampr.training.contrastive; print('contrastive OK')"
python -m pytest tests/test_contrastive.py tests/test_ampr_v3_return_z.py \
                 tests/test_trainer_v3_contrastive_smoke.py -q   # sanity trên server
```

---

## 1. Phase 0 — Modality ablation (eval-only, ~chục phút)

Chạy trên checkpoint v3 baseline, không train. Mỗi nhánh, bin headline LT_95:

```bash
for b in mf bp cc; do
  python scripts/diagnose_modality.py \
    --config configs/${b}_v3_esm3b.yaml \
    --checkpoint checkpoints/${b}_v3_esm3b/best.pt \
    --split test_LT_95
done
```

Kết quả: `[ABLATE] full/-gnn/-ppi/seq_only Fmax=...` + file
`results/${b}_v3_esm3b_predictions.ablate_test_LT_95.json`.
→ **Đọc:** nếu `full ≈ seq_only` thì nhánh structure/PPI không đóng góp (xác nhận giả thuyết spec §0).
Đây là bảng ablation cho thesis.

---

## 2. Tạo biến thể λ_cl (weight sweep)

Plan dùng `weight ∈ {0.1, 0.5, 1.0}`. Sinh config con từ MF (CC/BP chỉ chạy weight=0.5 để tiết kiệm GPU):

```bash
cd /raid/team/datn
for w in 01 05 10; do
  case $w in 01) val=0.1;; 05) val=0.5;; 10) val=1.0;; esac
  sed "s/^    weight: .*/    weight: ${val}/; \
       s#mf_v4_supcon/#mf_v4_supcon_w${w}/#g; \
       s#mf_v4_supcon_train#mf_v4_supcon_w${w}_train#g; \
       s#mf_v4_supcon_predictions#mf_v4_supcon_w${w}_predictions#g" \
       configs/mf_v4_supcon.yaml > configs/mf_v4_supcon_w${w}.yaml
done
python -c "import yaml;[yaml.safe_load(open(f'configs/mf_v4_supcon_w{w}.yaml')) for w in ('01','05','10')];print('OK')"
```

(CC/BP: dùng thẳng `configs/cc_v4_supcon.yaml`, `configs/bp_v4_supcon.yaml` với weight mặc định 0.5.)

---

## 3. Dry-run trước khi launch thật (bắt buộc)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/mf_v4_supcon_w05.yaml --dry_run
```
Kỳ vọng: log `[V3] Contrastive (Module A): SupCon ... weight=0.5`, 1 epoch xong,
`[DIAG] ... cl=<khác 0>`, không crash.

---

## 4. Check VRAM rồi launch — 1 run / 1 GPU

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```
Quy tắc: chỉ nhồi job thứ 2 lên 1 GPU nếu VRAM trống >> footprint dry-run; **không** ghép 2 job nặng (BP).

```bash
# 5 run chính (MF sweep 3 + CC + BP), mỗi GPU một run:
CUDA_VISIBLE_DEVICES=0 nohup python main.py --config configs/mf_v4_supcon_w01.yaml > logs/run_mf_w01.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --config configs/mf_v4_supcon_w05.yaml > logs/run_mf_w05.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python main.py --config configs/mf_v4_supcon_w10.yaml > logs/run_mf_w10.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python main.py --config configs/cc_v4_supcon.yaml     > logs/run_cc_w05.out 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python main.py --config configs/bp_v4_supcon.yaml     > logs/run_bp_w05.out 2>&1 &
# GPU 5-7 rảnh: queue thêm CC/BP weight khác nếu muốn, hoặc để dành.
```

---

## 5. Giám sát collapse guard (mỗi epoch)

```bash
grep -E "cross_protein_std|val_Fmax_dag" logs/mf_v4_supcon_w05_train.log | tail -20
```
- **OK:** `cross_protein_std > 1e-4` và `val_Fmax_dag` tăng dần.
- **Collapse:** `cross_protein_std → 0`, `val_Fmax` kẹt ~0.02 → kill run đó, thêm vào config
  `loss_type: bce` + `pos_weight_cap: 50` (giữ contrastive) rồi relaunch:
  ```bash
  # trong configs/<branch>_v4_supcon_*.yaml: đổi loss_type: asl -> bce, thêm pos_weight_cap: 50
  ```

---

## 6. Eval mọi bin + DIAMOND (sau khi train xong, chọn best val_Fmax_dag)

```bash
BEST=mf_v4_supcon_w05    # thay bằng config có best val mỗi nhánh
for s in test_LT_30 test_LT_40 test_LT_50 test_LT_70 test_LT_95; do
  python main.py --config configs/${BEST}.yaml --eval-only \
    --checkpoint checkpoints/${BEST}/best.pt --test-split $s
done
grep -E "\[V3-EVAL\]\[(dag|ens)\]" logs/${BEST}_*.log
```
So **LT_95 Fmax (+DIAMOND)** với baseline v3: **MF 0.614 / BP 0.507 / CC 0.538**.

---

## 7. Ghi kết quả

Tổng hợp per-branch: best λ_cl, val Fmax, gap val→test, LT_95(+D) vs v3 → thêm vào
`docs/RESULTS_V4.md`, commit:

```bash
git add docs/RESULTS_V4.md
git commit -m "docs(results): Module A (SupCon) sweep results vs v3 baseline"
git push
```

---

## Thứ tự thời gian gợi ý (24h)

| Giờ | Việc |
|---|---|
| 0.0–0.5 | §0 setup + §1 modality ablation (eval-only) |
| 0.5–1.0 | §2 tạo config + §3 dry-run + §4 launch |
| 1.0–8.0 | train MF/CC xong sớm; BP chạy tiếp; §5 giám sát |
| 8.0–10 | §6 eval mọi bin + DIAMOND cho MF/CC; BP khi xong |
| 10–24 | dự phòng cho BP / relaunch nếu collapse / §7 ghi kết quả |
