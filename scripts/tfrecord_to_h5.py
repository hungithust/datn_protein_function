#!/usr/bin/env python
"""Convert DeepFRI PDB_GO TFRecords to a single HDF5 per split.

Output schema (matches Phase 1's contact_maps_test.h5):
    h5_file[prot_id] = (L, L) float32 distance matrix
    h5_file[prot_id].attrs["sequence"] = sequence string (decoded from seq_1hot)
"""

import argparse
import glob as _glob
import logging
from pathlib import Path

import h5py
import numpy as np

from ampr.data.tfrecord_loader import iter_tfrecord

logging.basicConfig(level=logging.INFO, format="[tfr2h5] %(message)s")
log = logging.getLogger(__name__)

# DeepFRI 26-character alphabet for seq_1hot (from preprocessing/PDB2TFRecord.py)
ALPHABET = "ACDEFGHIKLMNPQRSTVWYBOUXZ-."


def onehot_to_sequence(M: np.ndarray) -> str:
    idx = M.argmax(axis=-1)
    return "".join(ALPHABET[i] for i in idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True,
                        help="e.g. /kaggle/input/pdb-go-tfrecords/PDB_GO_train_*.tfrecords")
    parser.add_argument("--out", type=Path, required=True,
                        help="output .h5 path")
    args = parser.parse_args()

    files = sorted(_glob.glob(args.input_glob))
    files = [Path(p) for p in files]
    log.info("found %d TFRecord files", len(files))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with h5py.File(args.out, "w") as h5:
        for tfp in files:
            log.info("processing %s", tfp)
            for rec in iter_tfrecord(tfp):
                pid = rec["prot_id"]
                if pid in h5:
                    continue
                ds = h5.create_dataset(pid, data=rec["cmap"],
                                       compression="gzip", compression_opts=4)
                ds.attrs["sequence"] = onehot_to_sequence(rec["seq_1hot"])
                n_done += 1
                if n_done % 500 == 0:
                    log.info("  saved %d", n_done)
    log.info("done — %d cmaps in %s", n_done, args.out)


if __name__ == "__main__":
    main()
