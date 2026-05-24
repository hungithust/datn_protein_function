#!/bin/bash
set -euo pipefail
python -c "
from Bio import SeqIO
import json
order = set()
with open('data/pdbch/splits.json') as f:
    order = set(json.load(f)['test'])
out = open('data/pdbch/test_sequences.fasta', 'w')
for r in SeqIO.parse('data/pdbch/sequences.fasta', 'fasta'):
    if r.id in order:
        out.write(f'>{r.id}\n{str(r.seq)}\n')
out.close()
"
diamond blastp \
  --db results/baselines/diamond/train_db \
  --query data/pdbch/test_sequences.fasta \
  --out results/baselines/diamond/search_results.tsv \
  --outfmt 6 qseqid sseqid pident length qlen slen \
  --max-target-seqs 50 \
  --evalue 1e-3 \
  --more-sensitive
