#!/usr/bin/env python
"""Verify TFRecord label vectors use the SAME term order as our labels_*.npy.

Strategy: find PDB chains present in BOTH a PDB_GO TFRecord and our
data/pdbch/labels_{ont}.npy (via protein_order.json), and assert the binary
vectors are identical. If they match, SWISS-MODEL labels (same DeepFRI release)
can be consumed directly with no remap.
"""
import argparse
import glob as _glob
import json
from pathlib import Path

import numpy as np

from ampr.data.tfrecord_loader import iter_tfrecord


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-glob", required=True, help="PDB_GO_*.tfrecords glob (from PDB-GO.tar.gz)")
    ap.add_argument("--data-dir", default="data/pdbch")
    ap.add_argument("--dist-key", default="cb_dist_matrix")
    ap.add_argument("--max-check", type=int, default=200)
    args = ap.parse_args()

    dd = Path(args.data_dir)
    order = json.loads((dd / "protein_order.json").read_text())
    idx = {p: i for i, p in enumerate(order)}
    labs = {ont: np.load(dd / f"labels_{ont}.npy") for ont in ("mf", "bp", "cc")}

    checked = 0
    for tfp in sorted(_glob.glob(args.pdb_glob)):
        for rec in iter_tfrecord(Path(tfp), dist_key=args.dist_key):
            pid = rec["prot_id"]
            if pid not in idx:
                continue
            row = idx[pid]
            for ont in ("mf", "bp", "cc"):
                ours = (labs[ont][row] > 0.5).astype(np.int64)
                theirs = rec["labels"][ont].astype(np.int64)
                assert ours.shape == theirs.shape, f"{pid}/{ont}: shape {ours.shape} vs {theirs.shape}"
                if not np.array_equal(ours, theirs):
                    raise SystemExit(f"MISMATCH {pid}/{ont}: term order differs — remap needed")
            checked += 1
            if checked >= args.max_check:
                print(f"[parity] OK — {checked} chains match across mf/bp/cc; no remap needed")
                return
    print(f"[parity] OK — {checked} chains checked (all match)")


if __name__ == "__main__":
    main()
