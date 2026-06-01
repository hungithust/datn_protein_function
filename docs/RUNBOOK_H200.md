# RUNBOOK — AMPR on 8×H200 (Hackathon)

## Access
- JupyterLab: http://<IP>:8888/lab  (password from VTS) — inspection only
- SSH: `ssh <user>@<IP>` — all training via tmux

## Cold start (one-time)
1. `ssh <user>@<IP>` ; ensure repo at `/raid/team/datn` or run `scripts/server_setup.sh`
2. Upload `kaggle.json` to `~/.kaggle/`, then `bash scripts/server_setup.sh`
3. `bash scripts/pull_kaggle_data.sh`
4. `python scripts/verify_inputs.py --config configs/mf_v3.yaml`  (repeat bp/cc) → ALL PASS

## Train baseline (Track 1)
- `bash scripts/launch_baseline.sh`  → GPUs 0–2, sessions train_{mf,bp,cc}
- Monitor: `tmux attach -t train_mf` (detach Ctrl-b d) ; `watch -n1 nvidia-smi`
- Logs: `logs/{mf,bp,cc}_v3_h200.log` ; checkpoints: `checkpoints/{mf,bp,cc}_v3/best.pt`

## GPU map
- 0–2: baseline (Track 1).  3–7: free for upgrade precompute/sweep (Plans 2–3).

## Gotchas
- Jobs survive browser close (tmux). They survive container restart only if started under tmux that
  itself survives — prefer re-launching after a restart; `/raid` data persists regardless.
- OOM: lower `batch_size`. Never edit driver/CUDA/Docker-root (hackathon rule).
- Stale/misaligned artifact → regenerate it (Plan 2), do not shim.
