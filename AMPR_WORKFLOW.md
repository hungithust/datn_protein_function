# AMPR Workflow Guide

Step-by-step execution for the **V3 pipeline** (`AMPRModelV3`, entry `main.py → _run_v3 / _eval_v3`).
Maps each task to its actual script + artifact. Runs on the 8×H200 server (or Kaggle for legacy runs).

> The old ProteinBERT/ProstT5/Node2Vec + TFRecord workflow is **deprecated**. Modalities are now
> **ESM-2 (frozen) + contact-map GCN + DeepGO PPI**, fused by cross-modal adaptive gating, with a
> combined SapBERT+GO-graph term embedding. Migration story: [docs/HANH_TRINH_THU_NGHIEM.md](docs/HANH_TRINH_THU_NGHIEM.md).

---

## Phase 0: Server setup + input gate

```bash
bash scripts/server_setup.sh        # NGC pytorch image deps, dirs (/raid/team/datn)
bash scripts/pull_kaggle_data.sh    # FASTA, DeepFRI annotations, splits, go-basic.obo
python scripts/verify_inputs.py     # GATE — confirm every artifact exists & is index-aligned
```

**Rule:** if `verify_inputs.py` reports a missing/stale artifact, **regenerate it** — never shim.
Only FASTA is raw input on the server; `cmap_all.h5` and `ppi_deepgo.npy` are reused from Kaggle.

**Note (node-07):** GPUs 0-1 are usually occupied (vLLM). Use `CUDA_VISIBLE_DEVICES=2..7`.

---

## Phase 1: Build artifacts (one-time)

All embeddings are precomputed once; training only touches the small fusion+classifier.

### 1.1 Labels, splits, DAG

```bash
python scripts/build_labels_from_annot.py     # → data/pdbch/labels_{mf,bp,cc}.npy
python scripts/build_splits_from_deepfri.py   # → data/pdbch/splits.json (+ LT_* test bins)
python scripts/build_dag_from_obo.py          # → data/pdbch/dag_matrix_{mf,bp,cc}.npy
```

`protein_order.json` fixes the row order shared by every `.npy`/`.h5` — the parity invariant
checked by `verify_label_parity.py`.

### 1.2 Sequence embeddings (ESM-2 residue-level, sharded)

```bash
bash scripts/launch_esm3b_precompute.sh       # → data/embeddings/esm2_3b_residue.h5  (2560d)
# or precompute_esm2_residue.py for 650M       → data/embeddings/esm2_650m_residue.h5 (1280d)
```

**Backbone choice:** 650M (MF/CC official) or 3B (BP). Frozen — no fine-tuning.
Storage: 3B for 30K PDB is heavy but fine; 220K SWISS-MODEL at 3B is ~280GB → use 650M if scaling data.

### 1.3 PPI (DeepGO)

```bash
python scripts/build_ppi_from_deepgo.py
# → data/embeddings/ppi_deepgo.npy (256d) + ppi_deepgo_mask.npy (availability)
```

The mask lets the fusion gate **mask off PPI** for proteins without interaction data.

### 1.4 GO term embedding (combined — REQUIRED)

```bash
python scripts/precompute_go_text.py          # SapBERT on GO definitions (768d)
python scripts/precompute_go_graph.py         # GO-graph structural embedding
python scripts/build_go_combined.py           # → data/embeddings/go_emb_{mf,bp,cc}_v2.npy (896d)
```

⚠️ **Use the combined `_v2` (896d), not text-only** — text-only GO embeddings cause a
dead-gradient collapse under ASL (Fmax stuck at 0.0209). See journey §1 and §3.

### 1.5 Contact maps + DIAMOND homology baseline

```bash
python scripts/precompute_cmap_test.py        # → data/contact_maps/cmap_all.h5 (10Å threshold)
bash scripts/run_diamond.sh                   # → data/diamond/diamond_results_{mf,bp,cc}.tsv
```

DIAMOND transfers GO terms from the most similar training protein; ensembled with the model
at inference (`diamond_alpha: 0.6`).

---

## Phase 2: Training (3 seeds × 3 branches for the ensemble)

```bash
for seed in 42 123 2024; do
  python main.py --config configs/mf_v3_esm3b.yaml --seed $seed
done
# repeat with bp_v3_esm3b.yaml and cc_v3_esm3b.yaml
```

### Config anatomy (`configs/mf_v3_esm3b.yaml`)

```yaml
branch: MF
n_terms: 489
data:
  protein_order: data/pdbch/protein_order.json
  splits:        data/pdbch/splits.json
  labels:        data/pdbch/labels_mf.npy
  dag_matrix:    data/pdbch/dag_matrix_mf.npy
  esm2_h5:       data/embeddings/esm2_3b_residue.h5     # or esm2_650m_residue.h5
  ppi_emb:       data/embeddings/ppi_deepgo.npy
  ppi_mask:      data/embeddings/ppi_deepgo_mask.npy
  cmap_h5:       data/contact_maps/cmap_all.h5
  go_emb:        data/embeddings/go_emb_mf_v2.npy        # combined 896d
  diamond_tsv:   data/diamond/diamond_results_mf.tsv
model:
  version: v3
  structure_modality: gnn
  seq:    {d_model: 2560, n_transformer_layers: 2, n_heads: 8, dropout: 0.2}
  gnn:    {node_dim: 256, n_layers: 3, cmap_threshold: 10.0}
  ppi:    {in_dim: 256, hidden: 512}
  fusion: {d_model: 512, n_layers: 2, n_heads: 8}
  classifier: both        # linear + GO-emb head
  d_hidden: 512
training:
  epochs: 50
  batch_size: 64          # 2560d residue ~2x memory; 64 safe on H200
  lr: 1.0e-3
  lr_scheduler: plateau   # halve LR when val Fmax_dag plateaus
  weight_decay: 1.0e-2    # AdamW
  loss_type: asl          # ASL with combined GO emb (BP uses bce + pos_weight)
  asl_gamma_neg: 4
  lambda_dag: 0.5
  seed: 42
inference:
  use_dag_propagation: true
  use_diamond_ensemble: true
  diamond_alpha: 0.6
```

**Loss choice per branch:**
- MF/CC: `loss_type: asl` with the combined GO embedding (no collapse).
- **BP (1943 terms, longest tail): `loss_type: bce` + per-class `pos_weight`** — ASL
  dead-gradients on the long tail at *every* stage (train and pretrain).

### Logging to expect

```
[EPOCH 01/50] lr=1.0e-3 | loss=0.342 (cls=0.289 dag=0.053)
              α_seq=0.61 α_struct=0.25 α_ppi=0.14 | val Fmax_dag=0.41
...
[BEST] epoch 29 → saved checkpoints/mf_v3_esm3b/best.pt  (val Fmax_dag=0.760)
```

If Fmax is pinned at **0.0209** with probs ≈ 0 → dead-gradient collapse (switch to BCE+pos_weight,
verify GO emb is the combined `_v2`, check grad_clip).

---

## Phase 3: Evaluation (ensemble + stratified by identity)

### 3.1 Single model

```bash
python main.py --config configs/mf_v3_esm3b.yaml --eval-only \
  --checkpoint checkpoints/mf_v3_esm3b/best.pt --test-split test_LT_95
```

### 3.2 3-seed ensemble + DIAMOND (headline)

```bash
cks="checkpoints/mf_v3_esm3b_s42/best.pt checkpoints/mf_v3_esm3b_s123/best.pt checkpoints/mf_v3_esm3b_s2024/best.pt"
for split in test test_LT_30 test_LT_40 test_LT_50 test_LT_70 test_LT_95; do
  python scripts/ensemble_eval.py --config configs/mf_v3_esm3b.yaml \
    --checkpoints $cks --split $split
done
grep -iE fmax logs/ensemble_mf_*.log
```

`ensemble_eval.py` = average probs across seeds → DAG-propagate → DIAMOND ensemble (α=0.6) →
Fmax/Smin/AUPRC per bin. Add `--tune-alpha` to tune the DIAMOND weight on valid and apply to test.

### 3.3 Diagnostics

```bash
python scripts/diagnose_modality.py --config <cfg> --checkpoint <best.pt>   # modality ablation
python scripts/evaluate_stratified.py ...                                   # per-LT-bin breakdown
```

---

## Phase 4: Results & reporting

### Headline (3-seed ensemble + DIAMOND, full test LT_95, Fmax)

| Branch | AMPR | DeepFRI | HEAL (SOTA) | Verdict |
|---|---|---|---|---|
| MF | **0.654** | 0.626 | 0.749 | beats DeepFRI (all bins) |
| BP | **0.539** | 0.540 | 0.594 | ≈ ties DeepFRI |
| CC | **0.566** | 0.612 | 0.687 | trails (low DIAMOND coverage) |

- Official model = **PDB-30K, no SWISS-MODEL pretrain** (pretrain hurt — see [docs/PHASE_V6_RESULTS.md](docs/PHASE_V6_RESULTS.md)).
- Baselines sourced from **HEAL supplementary Tables S3.1/S3.2** (same PDBch test, same LT_* bins).
- Full per-bin appendix: [docs/RESULTS_DATA.md](docs/RESULTS_DATA.md).

> ⚠️ DeepFRI numbers must come from the corrected source. The old
> `results/deepfri_baseline.json` was wrong and produced a false "BP/CC beat DeepFRI" claim.

### Figures

`python scripts/plot_v6_figures.py` — identity curve (Fmax vs LT bin), modality contribution,
val→test gap, baseline comparison.

---

## Troubleshooting

| Issue | Cause / Fix |
|---|---|
| Fmax stuck at 0.0209, probs≈0 | Dead-gradient collapse → BCE+pos_weight; ensure GO emb is combined `_v2`; grad_clip 5 |
| `verify_inputs.py` / parity fails | Artifact missing or row-order mismatch → regenerate against `protein_order.json` |
| OOM at 2560d (3B) | `batch_size: 64` (default); or drop to 650M backbone |
| GPUs 0-1 unavailable | node-07: `CUDA_VISIBLE_DEVICES=2..7` |
| Contrastive (Module A) tanks val | loss-scale mismatch — start `weight ≈ 1e-3`, retune per branch (negative result anyway) |
| val ↑ but test flat/worse | The intrinsic problem is the val→test gap (~0.20), not val; chase generalization, not val Fmax |

---

## Code Review Checklist (before thesis submission)

- [ ] All 3 branches (MF/BP/CC) × 3 seeds train without collapse
- [ ] Loss decreases; α weights are non-zero and shift during training
- [ ] `verify_label_parity.py` passes (row order consistent across all artifacts)
- [ ] Ensemble eval run on **all** LT bins (LT_30..LT_95), not just full test
- [ ] DeepFRI/HEAL baselines cite HEAL supp. (not the old wrong JSON)
- [ ] Single-model vs ensemble numbers reported separately
- [ ] Negative results (SupCon, SWISS-MODEL pretrain) documented with confound controls
- [ ] No hardcoded paths; configs drive everything

---

**Status:** V3 pipeline complete. MF/BP/CC trained + evaluated; official model = PDB-30K.
Remaining work tracked in [docs/HANH_TRINH_THU_NGHIEM.md](docs/HANH_TRINH_THU_NGHIEM.md).
