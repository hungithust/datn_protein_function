#!/usr/bin/env bash
# scripts/launch_v6_train.sh — stage1 pretrain (220K SM) then stage2 finetune (30K PDB, 3 seeds).
# Runs branches in parallel across GPUs; finetune seeds sequentially per branch.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

run_branch() {  # $1=short  $2=gpu
  local short="$1" gpu="$2"
  # Stage 1: pretrain (seed 42)
  CUDA_VISIBLE_DEVICES="$gpu" python main.py --config "configs/${short}_v6sm_pretrain.yaml" \
    2>&1 | tee "logs/${short}_v6sm_pretrain.run.log"
  local pre="checkpoints/${short}_v6sm_pretrain/best.pt"
  # Stage 2: finetune from the pretrained checkpoint, 3 seeds
  for seed in 42 123 2024; do
    CUDA_VISIBLE_DEVICES="$gpu" python main.py \
      --config "configs/${short}_v6sm_finetune_s${seed}.yaml" \
      --seed "$seed" --init-from "$pre" \
      2>&1 | tee "logs/${short}_v6sm_finetune_s${seed}.run.log"
  done
}

tmux kill-session -t v6_mf 2>/dev/null || true; tmux new-session -d -s v6_mf "$(declare -f run_branch); run_branch mf 0"
tmux kill-session -t v6_bp 2>/dev/null || true; tmux new-session -d -s v6_bp "$(declare -f run_branch); run_branch bp 1"
tmux kill-session -t v6_cc 2>/dev/null || true; tmux new-session -d -s v6_cc "$(declare -f run_branch); run_branch cc 2"
echo "[v6] launched mf/bp/cc on GPUs 0/1/2 — watch logs/*_v6sm_*.run.log"
