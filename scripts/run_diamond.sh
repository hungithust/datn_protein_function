#!/usr/bin/env bash
# Build Diamond DB from train sequences, search valid+test against it.
# Usage: bash scripts/run_diamond.sh <branch>   e.g. mf
set -euo pipefail
BRANCH="${1:-mf}"
DATA=data/pdbch
OUT=data/diamond
mkdir -p "$OUT"

# Step 1: extract train/valid/test FASTAs from full sequences using splits.json
python - <<PY
import json
from pathlib import Path
seqs = {}
cur = None
for line in Path('${DATA}/nrPDB-GO_2019.06.18_sequences.fasta').read_text().splitlines():
    if line.startswith('>'):
        cur = line[1:].split()[0]
        seqs[cur] = []
    else:
        seqs[cur].append(line)
splits = json.loads(Path('${DATA}/splits.json').read_text())
for k in ('train', 'valid', 'test'):
    with open(f'${OUT}/{k}.fasta', 'w') as fh:
        for p in splits[k]:
            if p in seqs:
                fh.write(f">{p}\n{''.join(seqs[p])}\n")
PY

diamond makedb --in ${OUT}/train.fasta --db ${OUT}/train_db
for split in valid test; do
  diamond blastp \
    --db ${OUT}/train_db \
    --query ${OUT}/${split}.fasta \
    --out ${OUT}/diamond_${BRANCH}_${split}.tsv \
    --outfmt 6 qseqid sseqid pident length qlen slen \
    --max-target-seqs 50 --evalue 1e-3 --more-sensitive
done
