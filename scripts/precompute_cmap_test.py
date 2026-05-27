"""Compute contact maps for test split and merge with train+valid into cmap_all.h5.

Reuses ampr.data.contact_map_h5.ContactMapStore conventions.
Downloads PDB if missing, computes Ca-Ca contact (A), writes HDF5.

Usage (Colab):
  python scripts/precompute_cmap_test.py \
    --split data/pdbch/splits.json --split_key test \
    --protein_order data/pdbch/protein_order.json \
    --train_valid_h5 data/contact_maps/cmap_train_valid.h5 \
    --out data/contact_maps/cmap_all.h5 \
    --pdb_dir data/pdb_cache/
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger('cmap_test')


def _download_pdb(pdb_id: str, out_dir: Path) -> Path:
    import urllib.request
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{pdb_id.lower()}.pdb"
    if p.exists():
        return p
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, p)
    return p


def _compute_cmap(pdb_path: Path, chain: str, threshold: float = 10.0) -> np.ndarray:
    from Bio.PDB import PDBParser, is_aa
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('s', str(pdb_path))
    model = next(iter(s))
    if chain not in [c.id for c in model]:
        raise ValueError(f"chain {chain} not found in {pdb_path}")
    coords = []
    for r in model[chain]:
        if not is_aa(r, standard=True):
            continue
        if 'CA' not in r:
            continue
        coords.append(r['CA'].coord)
    ca = np.asarray(coords, dtype=np.float32)
    if len(ca) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    d = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    return d.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--split_key', default='test')
    ap.add_argument('--protein_order', required=True)
    ap.add_argument('--train_valid_h5', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pdb_dir', required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    splits = json.loads(Path(args.split).read_text())
    test_ids = splits[args.split_key]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.train_valid_h5, 'r') as src, h5py.File(args.out, 'w') as dst:
        # Copy train+valid
        for k in src.keys():
            src.copy(k, dst)
        logger.info(f"[CMAP] copied {len(src.keys())} train/valid entries")

        n_ok, n_fail = 0, 0
        for pid in test_ids:
            if pid in dst:
                continue
            try:
                pdb_id, chain = pid.split('-')
                p = _download_pdb(pdb_id, Path(args.pdb_dir))
                d = _compute_cmap(p, chain)
                dst.create_dataset(pid, data=d, compression='gzip', compression_opts=4)
                n_ok += 1
            except Exception as e:
                logger.warning(f"[CMAP] {pid} FAIL: {e}")
                n_fail += 1
            if (n_ok + n_fail) % 100 == 0:
                logger.info(f"[CMAP] test: {n_ok} ok, {n_fail} fail")
    logger.info(f"[CMAP] DONE — added {n_ok} test cmaps, {n_fail} failed")


if __name__ == '__main__':
    main()
