#!/usr/bin/env python
"""Collect sweep results: best val_Fmax_dag per cell -> ranked table + winner.

Usage:
  python scripts/collect_sweep.py --logs_glob 'logs/sweep_mf_*.log'
"""
import argparse
import glob
import re

_FMAX = re.compile(r'val_Fmax_dag=([0-9.]+)')


def best_fmax_from_log(text: str):
    vals = [float(m.group(1)) for m in _FMAX.finditer(text)]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_glob', required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(args.logs_glob)):
        with open(path) as f:
            best = best_fmax_from_log(f.read())
        rows.append((path, best))
    rows.sort(key=lambda r: (r[1] is not None, r[1] or -1), reverse=True)
    print(f"{'config log':<48} best_val_Fmax_dag")
    for path, best in rows:
        print(f"{path:<48} {best if best is not None else 'n/a'}")
    if rows and rows[0][1] is not None:
        print(f"\n[WINNER] {rows[0][0]}  Fmax_dag={rows[0][1]:.4f}")


if __name__ == '__main__':
    main()
