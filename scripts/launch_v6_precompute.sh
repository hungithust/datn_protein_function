#!/usr/bin/env bash
# scripts/launch_v6_precompute.sh — ESM-2 650M residue embeddings (PDB + SWISS-MODEL).
# PDB (~37K, 1 GPU) + SWISS-MODEL (~245K, sharded across GPUs). Each shard own HDF5; merge after.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
MODEL=facebook/esm2_t33_650M_UR50D
mkdir -p logs data/embeddings

# --- PDB on GPU 0 ---
tmux kill-session -t esm650_pdb 2>/dev/null || true
tmux new-session -d -s esm650_pdb \
  "CUDA_VISIBLE_DEVICES=0 python scripts/precompute_esm2_residue.py \
     --fasta data/pdbch/sequences.fasta --protein_order data/pdbch/protein_order.json \
     --out data/embeddings/esm2_650m_pdb.h5 --model $MODEL --batch 16 --max_len 1022 \
     2>&1 | tee logs/esm650_pdb.log"

# --- SWISS-MODEL sharded across GPUs 1-7 ---
N=7
for i in $(seq 0 6); do
  gpu=$((i + 1))
  tmux kill-session -t "esm650_sm_$i" 2>/dev/null || true
  tmux new-session -d -s "esm650_sm_$i" \
    "CUDA_VISIBLE_DEVICES=$gpu python scripts/precompute_esm2_residue.py \
       --fasta data/swissmodel_art/sequences_sm.fasta \
       --protein_order data/swissmodel_art/protein_order_sm.json \
       --out data/embeddings/esm2_650m_sm_shard$i.h5 \
       --model $MODEL --batch 16 --max_len 1022 --shard $i --nshards $N \
       2>&1 | tee logs/esm650_sm_shard$i.log"
  echo "[ESM650] SM shard $i -> GPU $gpu"
done
echo "[ESM650] watch: tail -f logs/esm650_*.log"
echo "[ESM650] after all sessions end, merge SM shards:"
echo "  python scripts/merge_residue_h5.py --shards data/embeddings/esm2_650m_sm_shard*.h5 --out data/embeddings/esm2_650m_sm.h5"
