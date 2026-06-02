#!/usr/bin/env bash
# scripts/launch_esm3b_precompute.sh — ESM2-3B residue embeddings across GPUs 3-7.
set -euo pipefail
cd /raid/team/datn
FASTA=data/pdbch/nrPDB-GO_2019.06.18_sequences.fasta
ORDER=data/pdbch/protein_order.json
OUT=data/embeddings/esm2_3b_residue.h5   # one shared file (resumable, skips existing keys)
MODEL=facebook/esm2_t36_3B_UR50D
N=5
for i in 0 1 2 3 4; do
  gpu=$((i + 3))
  sess="esm3b_$i"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpu python scripts/precompute_esm2_residue.py \
       --fasta $FASTA --protein_order $ORDER --out $OUT \
       --model $MODEL --batch 8 --max_len 1022 --shard $i --nshards $N \
       2>&1 | tee logs/esm3b_shard$i.log"
  echo "[ESM3B] shard $i on GPU $gpu"
done
echo "[ESM3B] watch: tail -f logs/esm3b_shard*.log ; verify count after: python - <<'PY'
import h5py; print(len(h5py.File('$OUT').keys()))
PY"
