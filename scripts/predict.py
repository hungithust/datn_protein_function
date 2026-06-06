#!/usr/bin/env python
"""AMPR inference for external users — predict GO terms for new proteins.

Full-modality: per protein you provide a sequence (FASTA) and a structure (PDB,
one file per protein at <pdb_dir>/<id>.pdb). The script computes ESM-2 3B residue
embeddings + a Cα contact map, runs the trained AMPR model, applies True-Path-Rule
DAG propagation, and (optionally) blends with a DIAMOND homology ensemble.

PPI is not available for novel proteins, so the PPI modality is masked off (the
model's adaptive gating handles the missing modality).

Example:
  python scripts/predict.py --branch mf \
    --fasta my_proteins.fasta --pdb_dir my_pdbs/ \
    --out predictions_mf.tsv --threshold 0.3

Outputs a TSV: protein_id, go_term, score (sorted desc, filtered by threshold/topk).
Requires: torch, transformers, h5py, biopython, pyyaml. ESM-2 3B needs a GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ampr.models.ampr import AMPRModelV3
from ampr.evaluation.dag_inference import propagate_scores_upward


def load_fasta(path):
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


def cmap_from_pdb(pdb_path, threshold=None):
    """Cα-Cα distance matrix (Å) over standard residues of the first chain."""
    from Bio.PDB import PDBParser, is_aa
    s = PDBParser(QUIET=True).get_structure('s', str(pdb_path))
    chain = next(iter(next(iter(s))))
    ca = [r['CA'].coord for r in chain
          if is_aa(r, standard=True) and 'CA' in r]
    ca = np.asarray(ca, dtype=np.float32)
    if len(ca) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.linalg.norm(ca[:, None] - ca[None, :], axis=-1).astype(np.float32)


def esm2_residue(seqs, ids, model_name, max_len, device):
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
    out = {}
    for pid in ids:
        enc = tok([seqs[pid][:max_len]], return_tensors='pt', truncation=True,
                  max_length=max_len + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc).last_hidden_state[0].cpu().numpy()
        L = min(len(seqs[pid]), max_len)
        out[pid] = hs[1:1 + L]
    return out


def build_model(cfg, go_emb_dim, device):
    m = cfg['model']
    seq, gnn = m.get('seq', {}), m.get('gnn', {})
    ppi, fusion = m.get('ppi', {}), m.get('fusion', {})
    return AMPRModelV3(
        n_terms=cfg['n_terms'], seq_dim=seq.get('d_model', 2560),
        seq_n_heads=seq.get('n_heads', 8), seq_n_layers=seq.get('n_transformer_layers', 2),
        gnn_node_dim=gnn.get('node_dim', 256), gnn_n_layers=gnn.get('n_layers', 3),
        ppi_dim=ppi.get('in_dim', 256), d_hidden=m.get('d_hidden', 512),
        fusion_n_heads=fusion.get('n_heads', 8), fusion_n_layers=fusion.get('n_layers', 2),
        classifier=m.get('classifier', 'both'), go_emb_dim=go_emb_dim,
        cmap_threshold=gnn.get('cmap_threshold', 10.0), dropout=seq.get('dropout', 0.1),
    ).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', required=True, choices=['mf', 'bp', 'cc'])
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--pdb_dir', required=True, help='dir with <protein_id>.pdb per protein')
    ap.add_argument('--config', default=None, help='default configs/<branch>_v3_esm3b.yaml')
    ap.add_argument('--checkpoint', default=None, help='default checkpoints/<branch>_v3_esm3b/best.pt')
    ap.add_argument('--go_terms', default=None, help='default data/pdbch/go_terms_<branch>.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--topk', type=int, default=0, help='if >0, keep top-k terms per protein instead of threshold')
    ap.add_argument('--esm_model', default='facebook/esm2_t36_3B_UR50D')
    ap.add_argument('--max_len', type=int, default=1000)
    args = ap.parse_args()

    b = args.branch
    cfg = yaml.safe_load(open(args.config or f'configs/{b}_v3_esm3b.yaml'))
    ckpt_path = args.checkpoint or f'checkpoints/{b}_v3_esm3b/best.pt'
    go_terms = json.load(open(args.go_terms or f'data/pdbch/go_terms_{b}.json'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    go_emb = np.load(cfg['data']['go_emb']).astype(np.float32)
    dag = np.load(cfg['data']['dag_matrix']).astype(np.float32)
    go_emb_t = torch.from_numpy(go_emb).to(device)

    model = build_model(cfg, go_emb.shape[1], device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"[PREDICT] {b}: loaded {ckpt_path} (val_fmax_dag={ckpt.get('fmax_dag')})")

    seqs = load_fasta(args.fasta)
    ids = list(seqs)
    emb = esm2_residue(seqs, ids, args.esm_model, args.max_len, device)

    rows = []
    for pid in ids:
        pdb = Path(args.pdb_dir) / f"{pid}.pdb"
        if not pdb.exists():
            print(f"[PREDICT] WARN no PDB for {pid} ({pdb}); skipping")
            continue
        res = emb[pid]
        cm = cmap_from_pdb(pdb)
        L = min(res.shape[0], cm.shape[0], args.max_len)
        if L == 0:
            print(f"[PREDICT] WARN empty structure/seq for {pid}; skipping")
            continue
        batch = {
            'x_seq_residue': torch.from_numpy(res[:L]).unsqueeze(0).to(device),
            'seq_mask': torch.ones(1, L, dtype=torch.bool, device=device),
            'cmap': torch.from_numpy(cm[:L, :L]).unsqueeze(0).to(device),
            'x_ppi': torch.zeros(1, cfg['model'].get('ppi', {}).get('in_dim', 256), device=device),
            'ppi_mask': torch.tensor([False], device=device),
        }
        with torch.no_grad():
            probs = torch.sigmoid(model(batch, go_emb=go_emb_t)).cpu().numpy()
        probs = propagate_scores_upward(probs, dag)[0]
        if args.topk > 0:
            idx = np.argsort(-probs)[:args.topk]
        else:
            idx = np.where(probs >= args.threshold)[0]
        for i in sorted(idx, key=lambda k: -probs[k]):
            rows.append((pid, go_terms[i], float(probs[i])))
        print(f"[PREDICT] {pid}: L={L}, {len(idx)} GO terms")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write("protein_id\tgo_term\tscore\n")
        for pid, go, sc in rows:
            f.write(f"{pid}\t{go}\t{sc:.4f}\n")
    print(f"[PREDICT] wrote {len(rows)} predictions -> {args.out}")


if __name__ == '__main__':
    main()
