"""
Seq2TFRecord: Convert FASTA sequences + GO annotations to TFRecord format.

Lightweight alternative to DeepFRI's PDB2TFRecord.py — no distance matrices,
only sequence one-hot + GO labels. For running on Colab/Kaggle with minimal storage.

Usage:
    python seq2tfrecord.py \
        --annot data/nrPDB-GO_annot.tsv \
        --fasta data/nrPDB-GO_sequences.fasta \
        --split data/nrPDB-GO_train.txt \
        --out_prefix data/tfrecords/GO_train \
        --num_shards 10 \
        --num_threads 2
"""

import argparse
import gzip
import multiprocessing
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm


AMINO_ACIDS = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
               'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']


def seq2onehot(seq):
    """Convert amino acid sequence to one-hot encoding (L, 26)."""
    charset = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    onehot = np.zeros((len(seq), len(AMINO_ACIDS)), dtype=np.int8)
    for i, aa in enumerate(seq.upper()):
        idx = charset.get(aa, 0)
        onehot[i, idx] = 1
    return onehot


def read_fasta(fasta_file):
    """Read FASTA file (gzip or plain text). Returns {prot_id: sequence}."""
    sequences = {}

    if str(fasta_file).endswith('.gz'):
        f = gzip.open(fasta_file, 'rt')
    else:
        f = open(fasta_file, 'r')

    with f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)

    return sequences


def load_list(fname):
    """Read protein IDs from file (one per line)."""
    with open(fname, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_GO_annot(annot_file):
    """
    Load GO annotations from nrPDB-GO_annot.tsv.

    Returns:
        prot2annot: {prot_id: {'mf': [0,1,0,...], 'bp': [...], 'cc': [...]}}
        goterms: {'mf': [GO:001, GO:002, ...], ...}
        gonames: {'mf': [name1, name2, ...], ...}
    """
    prot2annot = {}
    goterms = {}
    gonames = {}

    with open(annot_file, 'r') as f:
        lines = [line.rstrip() for line in f]

    idx = 0
    ontologies = ['mf', 'bp', 'cc']
    ont_full_names = {'mf': 'molecular_function', 'bp': 'biological_process', 'cc': 'cellular_component'}

    for ont in ontologies:
        while idx < len(lines) and not lines[idx].startswith(f"### GO-terms ({ont_full_names[ont]})"):
            idx += 1

        if idx >= len(lines):
            continue
        idx += 1

        goterms[ont] = lines[idx].split('\t')
        idx += 1

        while idx < len(lines) and not lines[idx].startswith(f"### GO-names ({ont_full_names[ont]})"):
            idx += 1
        idx += 1

        gonames[ont] = lines[idx].split('\t')
        idx += 1

    while idx < len(lines) and not lines[idx].startswith("### PDB-chain"):
        idx += 1
    idx += 1

    for line in lines[idx:]:
        if not line or line.startswith("#"):
            continue

        parts = line.split('\t')
        if len(parts) < 4:
            continue

        prot_id = parts[0]
        prot2annot[prot_id] = {}

        for ont_idx, ont in enumerate(ontologies):
            go_str = parts[ont_idx + 1]
            annot_vec = np.zeros(len(goterms[ont]), dtype=np.int64)

            if go_str and go_str != '-':
                go_ids = go_str.split(',')
                for go_id in go_ids:
                    go_id = go_id.strip()
                    if go_id in goterms[ont]:
                        annot_vec[goterms[ont].index(go_id)] = 1

            prot2annot[prot_id][ont] = annot_vec

    return prot2annot, goterms, gonames


class SeqTFRecordWriter:
    """Write sequence one-hot + GO labels to TFRecord format (parallel)."""

    def __init__(self, prot_list, prot2annot, goterms, sequences, tfr_prefix, num_shards=10):
        self.prot_list = prot_list
        self.prot2annot = prot2annot
        self.goterms = goterms
        self.sequences = sequences
        self.tfr_prefix = tfr_prefix
        self.num_shards = num_shards

        shard_size = len(prot_list) // num_shards
        self.shard_ranges = [
            (i * shard_size, (i + 1) * shard_size if i < num_shards - 1 else len(prot_list))
            for i in range(num_shards)
        ]
        self.skipped = 0

    def _serialize_example(self, prot_id):
        """Create TF Example proto for one protein. Returns (example, success)."""
        if prot_id not in self.sequences:
            return None, False

        seq = self.sequences[prot_id]
        if len(seq) < 10:
            return None, False

        onehot = seq2onehot(seq).astype(np.float32)

        feature = {
            'prot_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[prot_id.encode('utf-8')])),
            'seq_1hot': tf.train.Feature(float_list=tf.train.FloatList(value=onehot.flatten())),
            'L': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(seq)])),
            'mf_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=self.prot2annot[prot_id]['mf'])),
            'bp_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=self.prot2annot[prot_id]['bp'])),
            'cc_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=self.prot2annot[prot_id]['cc'])),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto, True

    def _write_shard(self, shard_idx):
        """Write one shard to TFRecord file."""
        start, end = self.shard_ranges[shard_idx]
        shard_proteins = self.prot_list[start:end]

        output_path = f"{self.tfr_prefix}_{shard_idx:02d}-of-{self.num_shards:02d}.tfrecords"

        written = 0
        skipped = 0

        with tf.io.TFRecordWriter(output_path) as writer:
            for prot_id in shard_proteins:
                example, success = self._serialize_example(prot_id)
                if success:
                    writer.write(example.SerializeToString())
                    written += 1
                else:
                    skipped += 1

        return written, skipped, output_path

    def write_all(self, num_threads=4):
        """Write all shards in parallel."""
        print(f"\n[WRITER] Writing {len(self.prot_list)} proteins to {self.num_shards} shards")
        print(f"[WRITER] Using {num_threads} threads\n")

        try:
            with multiprocessing.Pool(processes=num_threads) as pool:
                results = pool.map(self._write_shard, range(self.num_shards))

            total_written = sum(r[0] for r in results)
            total_skipped = sum(r[1] for r in results)

            print("\n[DONE] TFRecord shards written:")
            for written, skipped, path in results:
                print(f"  {Path(path).name}: {written} proteins")

            print(f"\n[STATS] Total: {total_written} written, {total_skipped} skipped")

        except Exception as e:
            print(f"[ERROR] Multiprocessing failed: {e}")
            print("[FALLBACK] Writing sequentially...\n")

            total_written = 0
            total_skipped = 0

            for shard_idx in tqdm(range(self.num_shards), desc="Shards"):
                written, skipped, path = self._write_shard(shard_idx)
                total_written += written
                total_skipped += skipped

            print(f"[DONE] Total: {total_written} written, {total_skipped} skipped")


def main():
    parser = argparse.ArgumentParser(description='Convert sequences + GO annotations to TFRecord')
    parser.add_argument('--annot', type=str, required=True, help='Path to nrPDB-GO_annot.tsv')
    parser.add_argument('--fasta', type=str, required=True, help='Path to nrPDB-GO_sequences.fasta')
    parser.add_argument('--split', type=str, required=True, help='Path to train/valid/test protein ID list')
    parser.add_argument('--out_prefix', type=str, required=True, help='Output TFRecord prefix')
    parser.add_argument('--num_shards', type=int, default=10, help='Number of TFRecord shards')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of parallel threads')

    args = parser.parse_args()

    print("[START] Seq2TFRecord converter\n")

    print(f"[LOAD] Reading annotations from {args.annot}")
    prot2annot, goterms, gonames = load_GO_annot(args.annot)
    print(f"[ANNOT] Loaded {len(prot2annot)} proteins")
    for ont in ['mf', 'bp', 'cc']:
        print(f"        {ont.upper()}: {len(goterms[ont])} GO terms")

    print(f"\n[LOAD] Reading sequences from {args.fasta}")
    sequences = read_fasta(args.fasta)
    print(f"[FASTA] Loaded {len(sequences)} sequences")

    print(f"\n[LOAD] Reading split from {args.split}")
    prot_list = load_list(args.split)
    print(f"[SPLIT] {len(prot_list)} proteins in split")

    writer = SeqTFRecordWriter(prot_list, prot2annot, goterms, sequences, args.out_prefix, args.num_shards)
    writer.write_all(num_threads=args.num_threads)

    print(f"\n[SUCCESS] Output prefix: {args.out_prefix}")
    print(f"[SUCCESS] Output files: {args.out_prefix}_00-of-{args.num_shards:02d}.tfrecords ... "
          f"{args.out_prefix}_{args.num_shards-1:02d}-of-{args.num_shards:02d}.tfrecords")


if __name__ == '__main__':
    main()
