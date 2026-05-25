#!/usr/bin/env python
"""Compute C-alpha distance matrices from mmCIF structures."""

import argparse
import gzip
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser

logging.basicConfig(level=logging.INFO, format="[cmap] %(message)s")
log = logging.getLogger(__name__)

# Standard 3-letter → 1-letter amino acid map
AA3 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H",
    "ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q",
    "ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}


def _parse_structure(cif_path: Path, auth_chains: bool):
    parser = MMCIFParser(QUIET=True, auth_chains=auth_chains)
    opener = gzip.open if str(cif_path).endswith(".gz") else open
    with opener(cif_path, "rt") as f:
        return parser.get_structure(cif_path.stem, f)


def compute_cmap(cif_path: Path, chain_id: str) -> tuple[np.ndarray, str]:
    """Return (distance_matrix (L,L) float32, sequence (str length L)).

    Tries auth_chains=True first (author chain IDs, matching DeepFRI splits),
    falls back to auth_chains=False (label chain IDs) when not found.
    """
    structure = _parse_structure(cif_path, auth_chains=True)
    model = next(structure.get_models())
    available = [c.id for c in model]
    if chain_id not in available:
        # Fallback: label chains (some deposited structures differ)
        structure = _parse_structure(cif_path, auth_chains=False)
        model = next(structure.get_models())
        available = [c.id for c in model]
        if chain_id not in available:
            raise KeyError(f"chain {chain_id} not in {cif_path.name} "
                           f"(auth chains: {available})")
    chain = model[chain_id]

    coords = []
    seq = []
    for res in chain:
        if res.id[0] != " ":  # heteroatom
            continue
        if "CA" not in res:
            continue
        coords.append(res["CA"].get_coord())
        seq.append(AA3.get(res.get_resname(), "X"))

    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)  # ensure (L, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    cmap = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)
    return cmap, "".join(seq)


def save_cmap_h5(h5_path: Path, prot_id: str, cmap: np.ndarray, sequence: str) -> None:
    with h5py.File(h5_path, "a") as f:
        if prot_id in f:
            del f[prot_id]
        ds = f.create_dataset(prot_id, data=cmap, compression="gzip", compression_opts=4)
        ds.attrs["sequence"] = sequence


def _worker(args_tuple):
    """Top-level function so ProcessPoolExecutor can pickle it."""
    cid, pdb_dir = args_tuple
    pdb_id, chain = cid.split("-")
    cif_path = pdb_dir / f"{pdb_id.upper()}.cif.gz"
    if not cif_path.exists():
        return cid, None, None, "missing_file"
    try:
        cmap, seq = compute_cmap(cif_path, chain)
        return cid, cmap, seq, None
    except Exception as e:
        return cid, None, None, str(e)


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/pdbch/splits.json"))
    parser.add_argument("--split-key", default="test")
    parser.add_argument("--pdb-dir", type=Path, default=Path("data/pdbch/test_pdb_files"))
    parser.add_argument("--out", type=Path, default=Path("data/pdbch/contact_maps_test.h5"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="parallel worker processes (default: min(8, cpu_count))")
    args = parser.parse_args()

    with open(args.splits, "r", encoding="utf-8") as f:
        chain_ids = json.load(f)[args.split_key]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with h5py.File(args.out, "r") as f:
            done = set(f.keys())
        log.info("resuming — %d cmaps already in %s", len(done), args.out)

    todo = [(cid, args.pdb_dir) for cid in chain_ids if cid not in done]
    log.info("processing %d chains with %d workers", len(todo), args.workers)

    failed = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in todo}
        for future in as_completed(futures):
            cid, cmap, seq, err = future.result()
            if err:
                if err != "missing_file":
                    log.warning("%s: %s", cid, err)
                failed.append(cid)
            else:
                save_cmap_h5(args.out, cid, cmap, seq)
                n_done += 1
            total = n_done + len(failed)
            if total % 200 == 0:
                log.info("progress %d/%d (failed: %d)", total, len(todo), len(failed))

    (args.out.parent / "test_pdb_files" / "_cmap_failed.txt").write_text(
        "\n".join(failed), encoding="utf-8")
    log.info("done — %d cmaps saved, %d failed", n_done, len(failed))


if __name__ == "__main__":
    main()
