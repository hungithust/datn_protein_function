#!/usr/bin/env bash
# scripts/server_setup.sh — one-time H200 environment bootstrap.
# Idempotent: safe to re-run. MUST run INSIDE the jupyterlab NGC container:
#     docker exec -it jupyterlab bash
#     bash /raid/team/datn/scripts/server_setup.sh
# We isolate our extra deps in a venv created with --system-site-packages so the
# container's CUDA-enabled PyTorch is inherited (NEVER reinstall torch) and root
# site-packages stays clean. (/raid/team is the persistent NVMe path.)
set -euo pipefail

REPO_DIR=/raid/team/datn
REPO_URL=https://github.com/hungithust/datn_protein_function
VENV="$REPO_DIR/.venv"
export KAGGLE_CONFIG_DIR="$REPO_DIR/.kaggle"

# --- guard: must be inside the container -------------------------------------
if ! command -v python >/dev/null 2>&1; then
  echo "[SETUP][ERR] 'python' not found — not inside the jupyterlab container."
  echo "[SETUP][ERR] run:  docker exec -it jupyterlab bash"
  exit 1
fi

echo "[SETUP] repo -> $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "[SETUP] venv (inherits container torch via --system-site-packages) -> $VENV"
if [ ! -d "$VENV" ]; then
  python -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -c "import torch; print('[SETUP] inherited torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[SETUP] python extras into venv (NOT torch)"
pip install -q \
  transformers==4.41.2 obonet biopython h5py pyyaml tqdm sentence-transformers kaggle

echo "[SETUP] kaggle credentials -> $KAGGLE_CONFIG_DIR"
mkdir -p "$KAGGLE_CONFIG_DIR"
if [ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
  echo "[SETUP][WARN] place kaggle.json at $KAGGLE_CONFIG_DIR/kaggle.json then re-run"
else
  chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json"
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
import pathlib
src = pathlib.Path("ampr/models/gnn_encoder.py").read_text()
assert "import dgl" not in src and "from dgl" not in src, \
    "gnn_encoder imports DGL — cu121 wheel will not match the NGC CUDA image; refactor or install matching wheel"
print("[SETUP] gnn_encoder is DGL-free OK")
PY

echo "[SETUP] done — activate later with:  . $VENV/bin/activate"
