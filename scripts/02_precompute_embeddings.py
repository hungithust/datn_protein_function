#!/usr/bin/env python
"""Precompute embeddings for AMPR (PDBch dataset).

Outputs (rows aligned to data/pdbch/protein_order.json):
    data/embeddings/seq_embeddings.npy     (N, 1024)   ProteinBERT (Rostlab/prot_bert)
    data/embeddings/struct_embeddings.npy  (N, 1024)   zeros (placeholder; GNN mode skips it)
    data/embeddings/ppi_embeddings.npy     (N, 256)    zeros (placeholder; rebuild on Colab via notebook)
    data/embeddings/go_emb_mf.npy          (489, 768)  BioBERT (dmis-lab/biobert-v1.1)
    data/embeddings/go_emb_bp.npy          (1943, 768)
    data/embeddings/go_emb_cc.npy          (320, 768)

Usage:
    python scripts/02_precompute_embeddings.py --data-dir data/pdbch --out-dir data/embeddings

Flags:
    --skip-seq      do not run ProteinBERT (assumes seq_embeddings.npy exists)
    --skip-go       do not run BioBERT for GO terms
    --batch-size 8  ProteinBERT batch (T4: 8 safe at max_len 1000)
    --max-len 1000  truncate sequences
    --ppi-dim 256   dim of placeholder ppi_embeddings.npy (configs use 256)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="[embed] %(message)s")
log = logging.getLogger(__name__)


def read_fasta(path):
    seqs = {}
    cur_id, cur = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur)
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
        if cur_id:
            seqs[cur_id] = "".join(cur)
    return seqs


def encode_proteinbert(sequences, out_path, batch_size=8, max_len=1000):
    """Encode sequences with Rostlab/prot_bert. mean-pool → (1024,)."""
    from transformers import BertModel, BertTokenizer
    import re

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"ProteinBERT device={device}")

    tok = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert").to(device).eval()

    N = len(sequences)
    out = np.zeros((N, 1024), dtype=np.float32)

    for i in range(0, N, batch_size):
        batch_seqs = sequences[i:i + batch_size]
        # ProtBERT convention: spaces between residues; rare AAs → X
        batch_text = [re.sub(r"[UZOB]", "X", " ".join(list(s[:max_len]))) for s in batch_seqs]
        enc = tok.batch_encode_plus(batch_text, padding=True, truncation=True,
                                    max_length=max_len + 2, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state  # (B, L+2, 1024)
        # mean-pool over non-pad, non-special tokens (= attention_mask minus [CLS]/[SEP])
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0  # strip [CLS]
        # strip last non-pad token = [SEP]
        last_idx = mask.sum(dim=1).long() - 1 + 1  # last position before pad
        for b in range(mask.size(0)):
            sep_pos = enc["attention_mask"][b].sum().item() - 1
            mask[b, sep_pos] = 0
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        out[i:i + pooled.size(0)] = pooled.cpu().numpy()

        if (i // batch_size) % 50 == 0:
            log.info(f"  ProteinBERT batch {i // batch_size}/{(N + batch_size - 1) // batch_size}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    log.info(f"saved {out_path} shape={out.shape}")


def encode_biobert_go(go_term_ids, obo_path, out_path):
    """Encode GO terms (name + definition) with BioBERT. mean-pool → (768,)."""
    import obonet
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"BioBERT device={device}; loading OBO from {obo_path}")
    graph = obonet.read_obo(str(obo_path))

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device).eval()

    texts = []
    for gid in go_term_ids:
        node = graph.nodes.get(gid, {})
        name = node.get("name", gid)
        defn = node.get("def", "").split('"')
        defn = defn[1] if len(defn) > 1 else ""
        texts.append(f"{name}. {defn}".strip())

    out = np.zeros((len(go_term_ids), 768), dtype=np.float32)
    bs = 16
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state  # (B, L, 768)
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        out[i:i + pooled.size(0)] = pooled.cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    log.info(f"saved {out_path} shape={out.shape}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/pdbch"))
    p.add_argument("--out-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1000)
    p.add_argument("--ppi-dim", type=int, default=256)
    p.add_argument("--skip-seq", action="store_true")
    p.add_argument("--skip-go", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    protein_order = json.load(open(args.data_dir / "protein_order.json"))
    N = len(protein_order)
    log.info(f"protein_order: N={N}")

    # ── ProteinBERT (seq) ─────────────────────────────────────────────────
    if not args.skip_seq:
        log.info("[stage] ProteinBERT (seq_embeddings.npy)")
        fasta = read_fasta(args.data_dir / "sequences.fasta")
        missing = [p for p in protein_order if p not in fasta]
        if missing:
            log.warning(f"{len(missing)} proteins in protein_order missing from FASTA "
                        f"(first 3: {missing[:3]}) — using empty sequence (zeros)")
        seqs = [fasta.get(pid, "") for pid in protein_order]
        encode_proteinbert(seqs, args.out_dir / "seq_embeddings.npy",
                           batch_size=args.batch_size, max_len=args.max_len)
    else:
        log.info("[skip] seq_embeddings.npy")

    # ── struct_embeddings.npy: zeros (GNN mode bypasses; ProstT5 not used) ──
    struct_path = args.out_dir / "struct_embeddings.npy"
    if not struct_path.exists():
        np.save(struct_path, np.zeros((N, 1024), dtype=np.float32))
        log.info(f"[zeros] saved {struct_path} shape=({N}, 1024)")

    # ── ppi_embeddings.npy: zeros (replace later via colab_run.ipynb) ───────
    ppi_path = args.out_dir / "ppi_embeddings.npy"
    if not ppi_path.exists():
        np.save(ppi_path, np.zeros((N, args.ppi_dim), dtype=np.float32))
        log.info(f"[zeros] saved {ppi_path} shape=({N}, {args.ppi_dim}) "
                 "— rebuild via colab_run.ipynb Bước 5.4 (Node2Vec on DeepGO PPI graph)")

    # ── BioBERT (GO terms) ────────────────────────────────────────────────
    if not args.skip_go:
        obo = args.data_dir / "go-basic.obo"
        if not obo.exists():
            log.error(f"missing {obo} — required for GO term names/definitions")
            sys.exit(1)
        for branch in ("mf", "bp", "cc"):
            log.info(f"[stage] BioBERT (go_emb_{branch}.npy)")
            go_ids = json.load(open(args.data_dir / f"go_terms_{branch}.json"))
            encode_biobert_go(go_ids, obo, args.out_dir / f"go_emb_{branch}.npy")
    else:
        log.info("[skip] go_emb_*.npy")

    log.info("done.")


if __name__ == "__main__":
    main()
