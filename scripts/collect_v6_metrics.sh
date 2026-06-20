#!/usr/bin/env bash
# scripts/collect_v6_metrics.sh — full multi-metric × 5-bin collection for the v6 650M recipe.
# Runs 3-seed ensemble eval for BOTH recipes (650M-PDB-30K baseline + AMPR-B 220K->30K),
# all 3 branches, all 5 identity bins. ensemble_eval writes the FULL compute_all_metrics dict
# (fmax, smin, smin_raw, auprc_micro, auprc_macro, micro/macro AUROC, coverage) for the
# raw / dag / ensemble(+DIAMOND) blocks into results/*.ensemble_<split>.json.
# Usage: bash scripts/collect_v6_metrics.sh [GPU]   (default GPU 3; use 2-7 on node-07)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU="${1:-3}"
BINS=(test_LT_30 test_LT_40 test_LT_50 test_LT_70 test_LT_95)   # test == LT_95
mkdir -p logs

run() {  # $1 = config/checkpoint prefix (without _s<seed>)
  local pre="$1"
  local cks="checkpoints/${pre}_s42/best.pt checkpoints/${pre}_s123/best.pt checkpoints/${pre}_s2024/best.pt"
  for split in "${BINS[@]}"; do
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/ensemble_eval.py \
      --config "configs/${pre}_s42.yaml" --checkpoints $cks --split "$split" \
      2>&1 | tee "logs/collect_${pre}_${split}.log"
  done
}

for cfg in mf bp cc; do
  run "${cfg}_v6_pdb30base"     # 650M-PDB-30K baseline (no pretrain) — official model
  run "${cfg}_v6sm_finetune"    # AMPR-B (220K SWISS-MODEL -> 30K PDB)
done
echo "[collect] done — parse with: python scripts/extract_metrics_table.py"
