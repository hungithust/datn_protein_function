#!/usr/bin/env python
"""Extract SWISS-MODEL TFRecords into AMPRDatasetV3 artifacts.

Outputs (in --out dir):
    protein_order_sm.json   list[str]            row order shared by all npy below
    splits_sm.json          {"train": [...], "valid": [...]}
    labels_{mf,bp,cc}_sm.npy  (N, C) float32     binary GO labels
    ppi_zero_sm.npy         (N, 256) float32     all zeros (SWISS-MODEL lacks PPI)
    ppi_mask_sm.npy         (N,) bool            all False
    sequences_sm.fasta      FASTA                for ESM-2 precompute

Contact maps are NOT produced here — run scripts/tfrecord_to_h5.py separately.
"""
import argparse
import glob as _glob
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ampr.data.tfrecord_loader import iter_tfrecord

logging.basicConfig(level=logging.INFO, format="[sm_artifacts] %(message)s")
log = logging.getLogger(__name__)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYBOUXZ-."


def _onehot_to_seq(M: np.ndarray) -> str:
    return "".join(ALPHABET[i] for i in M.argmax(axis=-1))


def build_artifacts(train_glob, valid_glob, out_dir, n_terms, dist_key="cb_dist_matrix"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    order, split_of, seqs = [], {}, {}
    labels = {ont: [] for ont in ("mf", "bp", "cc")}
    seen = set()

    for split, pattern in (("train", train_glob), ("valid", valid_glob)):
        files = sorted(_glob.glob(pattern))
        log.info("%s: %d TFRecord files", split, len(files))
        for tfp in files:
            for rec in iter_tfrecord(Path(tfp), dist_key=dist_key):
                pid = rec["prot_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                order.append(pid)
                split_of[pid] = split
                seqs[pid] = _onehot_to_seq(rec["seq_1hot"])
                for ont in ("mf", "bp", "cc"):
                    v = rec["labels"][ont].astype(np.float32)
                    assert v.shape[0] == n_terms[ont], f"{pid}/{ont}: {v.shape[0]} != {n_terms[ont]}"
                    labels[ont].append(v)
                if len(order) % 5000 == 0:
                    log.info("  processed %d chains", len(order))

    (out / "protein_order_sm.json").write_text(json.dumps(order))
    (out / "splits_sm.json").write_text(json.dumps({
        "train": [p for p in order if split_of[p] == "train"],
        "valid": [p for p in order if split_of[p] == "valid"],
    }))
    for ont in ("mf", "bp", "cc"):
        np.save(out / f"labels_{ont}_sm.npy", np.stack(labels[ont]).astype(np.float32))
    N = len(order)
    np.save(out / "ppi_zero_sm.npy", np.zeros((N, 256), dtype=np.float32))
    np.save(out / "ppi_mask_sm.npy", np.zeros((N,), dtype=bool))
    with open(out / "sequences_sm.fasta", "w") as f:
        for pid in order:
            f.write(f">{pid}\n{seqs[pid]}\n")
    log.info("done — %d chains (%d train / %d valid)", N,
             sum(v == "train" for v in split_of.values()),
             sum(v == "valid" for v in split_of.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-glob", required=True)
    ap.add_argument("--valid-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-key", default="cb_dist_matrix")
    args = ap.parse_args()
    build_artifacts(args.train_glob, args.valid_glob, args.out,
                    n_terms={"mf": 489, "bp": 1943, "cc": 320}, dist_key=args.dist_key)


if __name__ == "__main__":
    main()
