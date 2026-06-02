#!/usr/bin/env python
"""Merge per-shard residue-embedding HDF5 files into one (skips existing keys).

Usage:
  python scripts/merge_residue_h5.py \
    --shards data/embeddings/esm2_3b_shard0.h5 data/embeddings/esm2_3b_shard1.h5 ... \
    --out data/embeddings/esm2_3b_residue.h5
"""
import argparse
from pathlib import Path

import h5py


def merge_h5(shard_paths, out_path):
    """Copy every dataset from each shard into out_path; skip keys already present."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with h5py.File(out_path, 'a') as dst:
        for sp in shard_paths:
            with h5py.File(sp, 'r') as src:
                n = 0
                for key in src.keys():
                    if key in dst:
                        continue
                    src.copy(src[key], dst, name=key)
                    n += 1
                total += n
                print(f"[MERGE] {sp}: +{n} (src had {len(src.keys())})")
        print(f"[MERGE] DONE — {len(dst.keys())} total keys in {out_path} (+{total} new)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    merge_h5(args.shards, args.out)


if __name__ == '__main__':
    main()
