#!/usr/bin/env python
"""Print feature keys, dtypes and lengths for the first N TFRecord examples."""
import argparse
import glob as _glob

import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. data/swissmodel/SWISS-MODEL_GO_train_*.tfrecords")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    files = sorted(_glob.glob(args.glob))
    print(f"[inspect] {len(files)} files match; reading {args.n} records from {files[0]}")
    ds = tf.data.TFRecordDataset(files[0])
    for k, raw in enumerate(ds.take(args.n)):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        print(f"--- record {k} ---")
        for key, feat in sorted(ex.features.feature.items()):
            kind = feat.WhichOneof("kind")
            vals = getattr(feat, kind).value
            print(f"  {key:18s} {kind:11s} len={len(vals)}")


if __name__ == "__main__":
    main()
