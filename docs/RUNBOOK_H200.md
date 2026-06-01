# RUNBOOK — AMPR on 8×H200 (Hackathon, node-07)

## Environment model
- All work happens **inside the `jupyterlab` NGC container** (`nvcr.io/nvidia/pytorch:24.10-py3`),
  not on the host. Enter it with: `docker exec -it jupyterlab bash`
- Inside the container `/workspace` == host `/raid/team`. The repo lives at **`/workspace/datn`**.
- Python deps are isolated in a venv at `/workspace/datn/.venv` created with
  `--system-site-packages` so the container's CUDA-enabled PyTorch is inherited (never reinstall torch).
- Kaggle creds live at `/workspace/datn/.kaggle/kaggle.json` (persistent on /raid);
  scripts export `KAGGLE_CONFIG_DIR` to it.

## Access
- JupyterLab: http://<IP>:8888/lab  (password from VTS) — inspection only
- SSH: `ssh root@<IP>` → `docker exec -it jupyterlab bash` — all training via tmux

## Cold start (one-time, inside the container)
1. `docker exec -it jupyterlab bash`
2. Ensure repo at `/workspace/datn` (server_setup.sh clones it if missing)
3. Put `kaggle.json` at `/workspace/datn/.kaggle/kaggle.json`
4. `bash /workspace/datn/scripts/server_setup.sh`  → creates venv, installs extras, GPU + DGL checks
5. `bash /workspace/datn/scripts/pull_kaggle_data.sh`
6. `. /workspace/datn/.venv/bin/activate`
   `python scripts/verify_inputs.py --config configs/mf_v3.yaml`  (repeat bp/cc) → ALL PASS

## Train baseline (Track 1)
- `bash /workspace/datn/scripts/launch_baseline.sh`  → GPUs 0–2, sessions train_{mf,bp,cc}
  (each tmux session activates the venv itself)
- Monitor: `tmux attach -t train_mf` (detach Ctrl-b d) ; `watch -n1 nvidia-smi`
- Logs: `logs/{mf,bp,cc}_v3_h200.log` ; checkpoints: `checkpoints/{mf,bp,cc}_v3/best.pt`

## GPU map
- 0–2: baseline (Track 1).  3–7: free for upgrade precompute/sweep (Plans 2–3).

## Gotchas
- Jobs survive browser close (tmux). They survive container restart only if started under tmux that
  itself survives — prefer re-launching after a restart; `/workspace` (=/raid) data + venv persist regardless.
- OOM: lower `batch_size`. Never edit driver/CUDA/Docker-root (hackathon rule).
- `python: command not found` → you're on the host; run `docker exec -it jupyterlab bash` first.
- Stale/misaligned artifact → regenerate it (Plan 2), do not shim.
