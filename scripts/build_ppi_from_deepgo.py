"""Build PPI embedding matrix from DeepGO graph_new_embeddings.pkl.

Mapping chain:
  PDB-chain (e.g. 1A0R-A) → UniProt accession (SIFTS) → DeepGO embedding (256d)

SIFTS mapping source: ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/pdb_chain_uniprot.csv.gz
or pre-parsed JSON {PDB-chain: UniProt}.

Outputs:
  ppi_deepgo.npy       (N, 256) float32
  ppi_deepgo_mask.npy  (N,)     bool
  ppi_coverage.json    {"total": N, "covered": k, "rate": k/N}
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger('build_ppi')


def build_ppi_matrix(pkl_path: str, sifts_map: dict, protein_order: list,
                     dim: int = 256) -> tuple:
    df = pd.read_pickle(pkl_path)
    acc_to_emb = {a: np.asarray(e, dtype=np.float32) for a, e in
                  zip(df['accessions'], df['embeddings'])}
    N = len(protein_order)
    emb = np.zeros((N, dim), dtype=np.float32)
    mask = np.zeros(N, dtype=bool)
    for i, pid in enumerate(protein_order):
        acc = sifts_map.get(pid)
        if acc and acc in acc_to_emb:
            vec = acc_to_emb[acc]
            if vec.shape[0] == dim:
                emb[i] = vec
                mask[i] = True
    return emb, mask


def _load_sifts(path: str) -> dict:
    """Accept JSON {pdbchain: uniprot} or CSV/TSV with columns PDB,CHAIN,SP_PRIMARY."""
    p = Path(path)
    if p.suffix == '.json':
        return json.loads(p.read_text())
    # Otherwise parse SIFTS CSV (skip header lines starting with #)
    df = pd.read_csv(p, comment='#')
    cols = {c.lower(): c for c in df.columns}
    pdb_c = cols.get('pdb')
    ch_c = cols.get('chain')
    up_c = cols.get('sp_primary') or cols.get('uniprot')
    if not (pdb_c and ch_c and up_c):
        raise ValueError(f"SIFTS CSV missing columns; got {df.columns.tolist()}")
    out = {}
    for _, row in df.iterrows():
        key = f"{str(row[pdb_c]).upper()}-{str(row[ch_c])}"
        out.setdefault(key, str(row[up_c]))  # first mapping wins
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--sifts', required=True, help='JSON {pdb-chain: uniprot} or SIFTS CSV')
    ap.add_argument('--protein_order', required=True)
    ap.add_argument('--out_emb', required=True)
    ap.add_argument('--out_mask', required=True)
    ap.add_argument('--out_coverage', required=True)
    ap.add_argument('--dim', type=int, default=256)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    sifts = _load_sifts(args.sifts)
    logger.info(f"[PPI] SIFTS entries: {len(sifts)}")

    order = json.loads(Path(args.protein_order).read_text())
    if isinstance(order, dict):
        order = [k for k, _ in sorted(order.items(), key=lambda kv: kv[1])]

    emb, mask = build_ppi_matrix(args.pkl, sifts, order, args.dim)
    Path(args.out_emb).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_emb, emb)
    np.save(args.out_mask, mask)
    cov = {'total': int(mask.size), 'covered': int(mask.sum()),
           'rate': float(mask.mean())}
    Path(args.out_coverage).write_text(json.dumps(cov, indent=2))
    logger.info(f"[PPI] coverage {cov['covered']}/{cov['total']} = {cov['rate']:.3f}")


if __name__ == '__main__':
    main()
