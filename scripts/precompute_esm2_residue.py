"""Precompute frozen ESM-2 650M per-residue embeddings → HDF5.

Each protein stored as separate dataset f[protein_id] = (L, 1280) float32.
Resumable: existing keys are skipped.

Usage (Colab/local):
  python scripts/precompute_esm2_residue.py \
    --fasta data/pdbch/nrPDB-GO_2019.06.18_sequences.fasta \
    --protein_order data/pdbch/protein_order.json \
    --out data/embeddings/esm2_residue.h5 \
    --batch 4 --max_len 1022
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np

logger = logging.getLogger('precompute_esm2')


def select_shard(ordered_ids, shard: int, nshards: int):
    """Deterministic contiguous partition: shard i of nshards over ordered_ids."""
    if nshards <= 1:
        return list(ordered_ids)
    n = len(ordered_ids)
    lo = (n * shard) // nshards
    hi = (n * (shard + 1)) // nshards
    return list(ordered_ids[lo:hi])


def write_residue_h5(iterator: Iterable[Tuple[str, np.ndarray]], out_path: str) -> None:
    """Append (protein_id, residue_emb) pairs to HDF5; skip existing keys."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'a') as f:
        for pid, arr in iterator:
            if pid in f:
                continue
            f.create_dataset(pid, data=arr.astype(np.float32),
                             compression='gzip', compression_opts=4)


def _load_fasta(path: str) -> dict:
    seqs = {}
    cur_id, cur_seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if cur_id is not None:
                    seqs[cur_id] = ''.join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seqs[cur_id] = ''.join(cur_seq)
    return seqs


def _esm2_iterator(seqs: dict, ordered_ids: list, existing: set,
                   batch: int, max_len: int):
    """Yield (pid, residue_emb) by running ESM-2 in batches."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"[ESM2] device={device}")
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()

    todo = [p for p in ordered_ids if p in seqs and p not in existing]
    logger.info(f"[ESM2] {len(todo)} proteins to embed ({len(existing)} already done)")
    n_truncated = 0

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        seqs_chunk = [seqs[p][:max_len] for p in chunk]
        n_truncated += sum(1 for p in chunk if len(seqs[p]) > max_len)
        enc = tok(seqs_chunk, return_tensors='pt', padding=True,
                  truncation=True, max_length=max_len + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hs = out.last_hidden_state.cpu().numpy()  # (B, L+2, 1280)
        for j, pid in enumerate(chunk):
            L = min(len(seqs[pid]), max_len)
            yield pid, hs[j, 1:1 + L, :]
        if (i // batch) % 50 == 0:
            logger.info(f"[ESM2] {i + len(chunk)}/{len(todo)}")
    logger.info(f"[ESM2] truncated {n_truncated} sequences (>{max_len} aa)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--protein_order', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max_len', type=int, default=1022)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    seqs = _load_fasta(args.fasta)
    ordered = json.loads(Path(args.protein_order).read_text())
    if isinstance(ordered, dict):
        ordered = [k for k, _ in sorted(ordered.items(), key=lambda kv: kv[1])]

    existing = set()
    if Path(args.out).exists():
        with h5py.File(args.out, 'r') as f:
            existing = set(f.keys())

    write_residue_h5(_esm2_iterator(seqs, ordered, existing, args.batch, args.max_len),
                     args.out)
    with h5py.File(args.out, 'r') as f:
        logger.info(f"[ESM2] DONE — {len(f.keys())} proteins in {args.out}")


if __name__ == '__main__':
    main()
