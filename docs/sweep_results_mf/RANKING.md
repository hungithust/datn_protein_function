# MF Architecture/Capacity Sweep — Results

**Date:** 2026-06-03 · **Hardware:** 8×H200 (NGC pytorch:24.10, `--ipc=host`) · **Epochs:** 50

Grid (8 cells): `classifier{both, label_attn}` × `d_hidden{512, 1024}` × `go_emb{comb v2 (896d), text-only (768d)}`.

## Ranking (best val Fmax_dag)

| Rank | Config | classifier | d_hidden | go_emb | best val_Fmax_dag |
|------|--------|-----------|----------|--------|-------------------|
| 1 | **both_h512_comb** | both | 512 | comb (v2, 896d) | **0.7525** |
| 2 | la_h512_comb | label_attn | 512 | comb | 0.5987 |
| 3 | la_h1024_comb | label_attn | 1024 | comb | 0.595 |
| 4 | both_h1024_comb | both | 1024 | comb | 0.5781 |
| 5-8 | *_text (all 4) | both/label_attn | 512/1024 | text-only (768d) | 0.0209 (collapsed) |

**Winner:** `both_h512_comb` — Fmax_dag **0.7525** (vs prior BCE baseline 0.678).

## Conclusions

1. **Winning combo is the existing base config** (`classifier=both`, `d_hidden=512`, `go_emb=go_emb_mf_v2.npy`).
   `configs/{mf,bp,cc}_v3_esm3b.yaml` already match → no config change needed to promote.
2. **Scaling up hurts:** `d_hidden=1024` is worse than 512 in every pairing.
3. **`label_attn` head hurts:** the new multi-head bilinear head underperforms the `both` dot-product head.
4. **Combined go_emb is essential:** all 4 text-only (768d) cells suffered total dead-gradient
   collapse (probs=0.0000, loss stuck at 0.1606, Fmax 0.0209). The text+graph combined `_v2` (896d)
   embedding is required.

## Winner training trace (excerpt)

```
[V3] Epoch 31/50: loss=0.0023 val_Fmax_dag=0.7270  (saved best)
[V3] Epoch 33/50: loss=0.0022 val_Fmax_dag=0.7350  (saved best)
[V3] Epoch 50/50: loss=0.0016 val_Fmax_dag=0.7525  (saved best)
[V3] Training complete. Best val Fmax (DAG): 0.7525
```

Late-epoch val Fmax oscillates (~0.66–0.75) near convergence at very low loss — normal; the
save-best-checkpoint mechanism captures the peak. A LR scheduler (cosine / ReduceLROnPlateau) is a
suggested follow-up to smooth the final run.

> Raw per-cell logs (`sweep_mf_*.log`) live on the server at `/raid/team/datn/logs/`. This file is
> the committed summary for report writing.
