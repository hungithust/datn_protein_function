#!/bin/bash
set -euo pipefail
# Build Diamond DB from training sequences only.
python -c "
from Bio import SeqIO
import json
order = set()
with open('data/pdbch/splits.json') as f:
    order = set(json.load(f)['train'])
out = open('data/pdbch/train_sequences.fasta', 'w')
for r in SeqIO.parse('data/pdbch/sequences.fasta', 'fasta'):
    if r.id in order:
        out.write(f'>{r.id}\n{str(r.seq)}\n')
out.close()
"
mkdir -p results/baselines/diamond
diamond makedb \
  --in data/pdbch/train_sequences.fasta \
  --db results/baselines/diamond/train_db
