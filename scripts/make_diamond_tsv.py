#!/usr/bin/env python
"""Build DIAMOND homology TSV for the v3 ensemble (DeepGOPlus-style).

Makes a DIAMOND DB from TRAIN sequences, blastp-searches the query split
(default `test`, which is a superset of all LT_* bins) against it, and writes
`query<TAB>subject<TAB>pident` to data/diamond/diamond_results_{branch}.tsv.

The hit table is sequence-only (branch-independent), so one search is copied to
all branch filenames; each branch later transfers its own labels.

Usage:
  python scripts/make_diamond_tsv.py \
    --fasta data/pdbch/nrPDB-GO_2019.06.18_sequences.fasta \
    --splits data/pdbch/splits.json
Requires the `diamond` binary (apt-get install -y diamond-aligner).
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def read_fasta(path):
    seqs, pid, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if pid is not None:
                    seqs[pid] = ''.join(buf)
                pid = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if pid is not None:
        seqs[pid] = ''.join(buf)
    return seqs


def write_fasta(path, ids, seqs):
    n = 0
    with open(path, 'w') as f:
        for p in ids:
            s = seqs.get(p)
            if s:
                f.write(f">{p}\n{s}\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--out_dir', default='data/diamond')
    ap.add_argument('--train_split', default='train')
    ap.add_argument('--query_splits', nargs='+', default=['valid', 'test'],
                    help='splits to use as queries (valid needed for alpha tuning)')
    ap.add_argument('--branches', nargs='+', default=['mf', 'bp', 'cc'])
    ap.add_argument('--diamond_bin', default='diamond')
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()

    seqs = read_fasta(args.fasta)
    splits = json.load(open(args.splits))
    train_ids = splits[args.train_split]
    query_ids, seen = [], set()
    for s in args.query_splits:
        for p in splits.get(s, []):
            if p not in seen:
                seen.add(p)
                query_ids.append(p)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_fa, query_fa = tmp / 'train.fasta', tmp / 'query.fasta'
        nt = write_fasta(train_fa, train_ids, seqs)
        nq = write_fasta(query_fa, query_ids, seqs)
        print(f"[DIAMOND] train={nt} seqs, query={nq} seqs")

        db = tmp / 'train_db'
        subprocess.run([args.diamond_bin, 'makedb', '--in', str(train_fa),
                        '-d', str(db), '--threads', str(args.threads)], check=True)
        merged = out_dir / 'diamond_results.tsv'
        subprocess.run([args.diamond_bin, 'blastp', '-q', str(query_fa),
                        '-d', str(db), '-o', str(merged), '--outfmt', '6',
                        'qseqid', 'sseqid', 'pident', '--max-target-seqs', '250',
                        '--threads', str(args.threads), '--quiet'], check=True)

    n_lines = sum(1 for _ in open(merged))
    print(f"[DIAMOND] wrote {merged} ({n_lines} hits)")
    for b in args.branches:
        dst = out_dir / f'diamond_results_{b}.tsv'
        shutil.copyfile(merged, dst)
        print(f"[DIAMOND] copied -> {dst}")


if __name__ == '__main__':
    main()
