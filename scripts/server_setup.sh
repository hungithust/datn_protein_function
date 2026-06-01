#!/usr/bin/env bash
# scripts/server_setup.sh — one-time H200 environment bootstrap.
# Idempotent: safe to re-run. Run from inside the JupyterLab/NGC container.
set -euo pipefail

REPO_DIR=/raid/team/datn
REPO_URL=https://github.com/hungithust/datn_protein_function

echo "[SETUP] repo -> $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "[SETUP] python extras (into NGC image)"
pip install -q \
  transformers==4.41.2 obonet biopython h5py pyyaml tqdm sentence-transformers

echo "[SETUP] kaggle cli + credentials"
pip install -q kaggle
mkdir -p "$HOME/.kaggle"
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "[SETUP][WARN] place kaggle.json at $HOME/.kaggle/kaggle.json then re-run"
else
  chmod 600 "$HOME/.kaggle/kaggle.json"
fi

echo "[SETUP] GPU sanity"
python - <<'PY'
import torch
print("[SETUP] cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
assert torch.cuda.device_count() == 8, "expected 8 H200"
print("[SETUP] gpu0:", torch.cuda.get_device_name(0))
PY

echo "[SETUP] DGL-free check for v3 GNN path"
python - <<'PY'
import ast, pathlib
src = pathlib.Path("ampr/models/gnn_encoder.py").read_text()
assert "import dgl" not in src and "from dgl" not in src, \
    "gnn_encoder imports DGL — cu121 wheel will not match CUDA-13 image; refactor or install matching wheel"
print("[SETUP] gnn_encoder is DGL-free OK")
PY

echo "[SETUP] done"
