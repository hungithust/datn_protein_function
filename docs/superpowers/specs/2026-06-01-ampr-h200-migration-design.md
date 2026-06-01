# AMPR — Migration to 8×H200 + Architecture Upgrade

**Date:** 2026-06-01
**Author:** Nguyen Viet Hung (with Claude Code)
**Status:** Design — awaiting user review
**Supersedes infra targets in:** CLAUDE.md (Colab/Kaggle), `notebooks/kaggle_phase3_*`

---

## 1. Goal

Move AMPR training off Kaggle/Colab/vast.ai onto a dedicated **NVIDIA Open Hackathon 2026** server
(8× H200 141 GB, 192 cores, ~2 TB RAM, 28 TB NVMe at `/raid`, root + Docker + JupyterLab + SSH).
Two objectives, run as **parallel tracks** on the 8 GPUs:

1. **Track 1 — Reproduce baseline fast.** Pull the existing precomputed embeddings from Kaggle and
   retrain MF/BP/CC v3 on H200 *today*, to confirm the new environment and re-establish numbers.
2. **Track 2 — Upgrade the architecture.** Re-precompute the strongest feasible embeddings and add
   real architectural improvements, then sweep capacity across the 8 GPUs.

The tracks share `/raid` but not GPUs, so neither blocks the other.

### Guiding principle (user directive, 2026-06-01)
**Prefer regenerate over force-fit.** If any cached artifact (an embedding HDF5, a `.npy`, a precomputed
matrix) doesn't match the new hardware/stack or the upgraded dims, **re-run it from source** rather than
writing shims to make the stale file work. Code and config are freely changeable to fit the H200 stack —
do not preserve old shapes/paths for their own sake. The only hard reuse constraints are the artifacts we
*cannot* regenerate from available raw inputs (cmap, PPI — no PDB/DeepGO source).

## 2. Constraints & givens

- **Raw inputs available:** only **FASTA** (`nrPDB-GO_2019.06.18_sequences.fasta`). No PDB, no DeepGO
  `graph_new_embeddings.pkl`, no SIFTS. Therefore:
  - Contact maps (`cmap_all.h5`) and PPI (`ppi_deepgo.npy` + mask) are **reused as-is** from Kaggle.
  - Every *new* embedding must be derivable from sequence alone (ESM-2, ProstT5) or from artifacts we
    already have (GO text/graph from `go-basic.obo` + `dag_matrix_*`).
- **Data transfer:** `kaggle.json` available → pull datasets directly on-server (internet, no VPN).
- **Execution:** SSH + `tmux`, one job per GPU (`CUDA_VISIBLE_DEVICES=i`). JupyterLab only for inspection.
- **Image:** `nvcr.io/nvidia/pytorch:24.10-py3` (torch CUDA 13, Python 3.12, TE/Apex/Triton). Extra deps
  via `pip` into the running container; `/raid` persists across container restart.
- **Dataset size:** train 29 902 / valid 3 323 / test 3 416 proteins; MF 489 terms (BP 1943, CC 320).
  Small relative to model capacity — see §6 overfitting discipline.
- **Do NOT change** driver, CUDA, Docker data-root, or the system JupyterLab container (hackathon rules).

## 3. Migration architecture

```
Kaggle datasets ──(kaggle API, on-server)──> /raid/team/datn/data/
                                                   │
   GPUs 0–2: TRACK 1 (baseline)            GPUs 3–7: TRACK 2 (upgrade)
   tmux MF/BP/CC on existing emb           ESM2-3B + ProstT5 precompute (FASTA)
        │                                  GO encoder + GO-graph precompute
   checkpoints/{mf,bp,cc}_v3/best.pt       retrain v3 (new dims) + capacity sweep
```

## 4. Phases

### Phase 0 — Bootstrap server (~15 min, one-time)
- SSH in; `tmux new -s setup`. Clone repo to `/raid/team/datn` (persists).
- `pip install transformers==4.41.2 obonet biopython h5py pyyaml tqdm sentence-transformers` into the
  NGC image. **Verify** the v3 GNN path does not hard-require the DGL cu121 wheel (the encoder appears
  custom, not DGL — confirm in `ampr/models/gnn_encoder.py` before relying on it).
- Place `kaggle.json` at `~/.kaggle/kaggle.json`, `chmod 600`.
- **Deliverable:** `scripts/server_setup.sh` (idempotent).

### Phase 1 — Data transfer + verification (~30–60 min)
- `kaggle datasets download` the required datasets straight into `/raid/team/datn/data/`, unzip in place.
  Exact slugs to confirm at run time; from `notebooks/kaggle_phase3_train_mf.ipynb`:
  `hungnguyenviet04/ampr-phase3-embeddings`, `…/ampr-phase3-embeddings-2`, `…/cmap-all`,
  `…/ampr-pdbch-phase0`.
- Place files at config-expected paths (`data/embeddings/esm2_residue.h5`,
  `data/contact_maps/cmap_all.h5`, `data/pdbch/*`, `data/embeddings/ppi_deepgo*.npy`,
  `data/embeddings/go_emb_*.npy`).
- Run the existing **VERIFY 1/3–3/3** alignment checks (from the Kaggle notebook) as a standalone script
  *before any training* — protein/term alignment, HDF5 coverage, DAG orientation.
- **Deliverable:** `scripts/verify_inputs.py` (extracted from notebook cells).

### Phase 2 — Track 1: baseline on H200 (start right after Phase 1)
- One `tmux` session per branch: `CUDA_VISIBLE_DEVICES={0,1,2} python main.py --config configs/{mf,bp,cc}_v3.yaml`.
- Raise `batch_size` (16 → 128+) and `num_workers` (2 → 16+) for H200; log to file.
- **Deliverable:** reproduced MF/BP/CC Fmax on new infra, same day; `scripts/launch_baseline.sh`.

### Phase 3 — Track 2: re-precompute strongest embeddings (parallel, GPUs 3–7)
All precompute jobs shard across free GPUs (split FASTA / term list into N chunks, one tmux pane per GPU).

- **3a. ESM2-3B sequence** — `facebook/esm2_t36_3B_UR50D` (2560d) → `esm2_3b_residue.h5`. Drop-in:
  `seq.d_model: 2560` (drives both `seq_encoder` and `gnn_node_init`, [ampr.py:179-188](../../../ampr/models/ampr.py#L179-L188)).
  Storage est. ≈ 60–95 GB gzip.
- **3b. ProstT5 structure** *(stretch — see §7)* — sequence→3Di→1024d → `prostt5.npy`. Enables a 4th
  modality; requires fusion surgery (§5).
- **3c. GO text encoder** — replace BioBERT with **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`,
  entity-aware, ideal for ontology terms) or **PubMedBERT-large**. Re-encode GO name+definition →
  `go_emb_{mf,bp,cc}_v2.npy`; update `go_emb_dim`.
- **3d. GO-graph embedding** — Node2Vec/GCN over the GO DAG (`go-basic.obo` + `dag_matrix_*`) → topology
  vector per term; **concat** with 3c text embedding to form the final GO label matrix.

### Phase 4 — Track 2: architecture upgrades + capacity sweep
New model surface (all gated by config; backbones stay frozen):

- **4a. Label-attention head** — replace the bilinear `z·proj(go_emb)ᵀ`
  ([ampr.py:255](../../../ampr/models/ampr.py#L255)) with cross-attention where the protein representation
  attends over the GO label-embedding matrix (TALE/ATGO/DeepGOZero pattern). New `classifier: 'label_attn'`
  option; keep `linear`/`biobert`/`both` for ablation.
- **4b. Wider fusion** — `CrossModalFusion` 2→4 layers; 4 tokens if ProstT5 enabled.
- **4c. Scale-up knobs** — `d_hidden 512→1024`, `seq_n_layers 2→4-6`, `ffn_mult 2→4`, `gnn 3→6`, all config.
- **4d. Regularization to match capacity** — weight decay, stochastic depth, label smoothing (modality
  dropout already present).
- **4e. Capacity sweep** — run combinations across 8 GPUs in parallel, select by **val Fmax (DAG-prop)**:
  - GO encoder: {SapBERT, PubMedBERT-large} × {text-only, text+graph}
  - head: {bilinear `both`, label-attn}
  - capacity: {d_hidden 512, 1024}
  Winner promoted to the final per-branch config.

## 5. Code changes (in the implementation plan, not now)

| File | Change |
|---|---|
| `scripts/precompute_esm2_residue.py` | add `--model` + `--shard i/N` for multi-GPU sharding |
| `scripts/precompute_prostt5.py` *(new, stretch)* | FASTA → 3Di → 1024d npy |
| `scripts/precompute_go_text.py` *(new)* | SapBERT/PubMedBERT GO encoder (generalize `encode_biobert_go`) |
| `scripts/precompute_go_graph.py` *(new)* | Node2Vec/GCN over GO DAG → term-topology emb |
| `ampr/models/cross_modal_fusion.py` | optional 4th token; deeper stack |
| `ampr/models/label_attention.py` *(new)* | cross-attention head |
| `ampr/models/ampr.py` (`AMPRModelV3`) | wire `classifier='label_attn'`, ProstT5 branch (stretch) |
| `ampr/data/dataset.py` (`AMPRDatasetV3`) | load ProstT5 (stretch) + concatenated GO emb |
| `configs/*_v3_esm3b.yaml`, `configs/*_v3_sweep_*.yaml` | new dims + sweep grid |
| `scripts/server_setup.sh`, `scripts/launch_baseline.sh`, `scripts/launch_sweep.sh`, `scripts/verify_inputs.py` | orchestration |
| `docs/RUNBOOK_H200.md` | operator runbook |

## 6. Overfitting discipline (why "phình to" is bounded)

29 902 frozen-embedding training samples is small. H200 removes the *memory* limit, not the *statistical*
one. Therefore: backbones stay frozen; capacity goes into **better inputs (ESM2-3B, stronger GO encoder)**
and **a smarter head (label-attention)**, with only moderate fusion/`d_hidden` scaling plus matching
regularization. "How big?" is answered by the **measured sweep (4e)**, not by a single large guess. A config
is kept only if it beats the baseline on val Fmax (DAG-prop).

## 7. Out of scope / stretch

- **ProstT5 4th modality (3b/4b-4-token)** — *stretch.* Adds fusion surgery; include only if the sweep shows
  the 3-modality upgrade still has headroom. Decision deferred to after Phase 4e.
- **ESM2-15B / ESM-C ablation** — not in this round (ESM2-3B only, per decision).
- **ESM-IF / any PDB-derived modality** — impossible from FASTA; excluded.
- **Multi-GPU DDP for a single model** — unnecessary; the model is small and parallelism is across
  jobs/branches, not within one model.
- **Retraining the pretrained backbones** — excluded; they stay frozen.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Kaggle dataset slugs/paths drift | Confirm live in Phase 1; `verify_inputs.py` gates training |
| DGL cu121 wheel vs CUDA-13 image | Confirm GNN path is DGL-free in Phase 0 before training |
| `transformers==4.41.2` vs ESM2-3B / SapBERT load | Smoke-load each model in Phase 0; bump pin only if needed |
| GO encoder dim change breaks alignment | `go_emb_dim` driven by config; re-run VERIFY after re-encode |
| Capacity sweep overfits | Selection strictly by val Fmax (DAG-prop), not train loss |

## 9. Deliverables summary

1. `scripts/server_setup.sh`, `scripts/verify_inputs.py`
2. `scripts/launch_baseline.sh`, `scripts/launch_sweep.sh`
3. precompute scripts: ESM2-3B (sharded), GO-text, GO-graph (+ ProstT5 stretch)
4. model: `label_attention.py`, fusion/`AMPRModelV3`/dataset edits
5. configs: `*_v3_esm3b.yaml` + sweep grid
6. `docs/RUNBOOK_H200.md`
