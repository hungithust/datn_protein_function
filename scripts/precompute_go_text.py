#!/usr/bin/env python
"""Encode GO terms (name + definition) with a configurable HF text encoder.

Generalizes the old BioBERT-only encoder. Default = SapBERT (entity-aware).
Output dim is inferred from the model (768 for SapBERT/PubMedBERT-base,
1024 for *-large), and main.py reads go_emb_dim from the saved array shape.

Usage:
  python scripts/precompute_go_text.py \
    --go_terms data/pdbch/go_terms_mf.json \
    --obo data/pdbch/go-basic.obo \
    --out data/embeddings/go_text_mf.npy \
    --model cambridgeltl/SapBERT-from-PubMedBERT-fulltext
"""
import argparse, json
from pathlib import Path
import numpy as np


def build_go_texts(go_term_ids, graph_nodes):
    """Pure: GO ids + node dict -> 'name. definition.' strings."""
    texts = []
    for gid in go_term_ids:
        node = graph_nodes.get(gid, {})
        name = node.get("name", gid)
        raw = node.get("def", "")
        defn = raw.split('"')[1] if '"' in raw else ""
        texts.append(f"{name}. {defn}".strip())
    return texts


def encode(texts, model_name, batch=16, max_len=128):
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    dim = model.config.hidden_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=max_len, return_tensors='pt').to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out[i:i + pooled.size(0)] = pooled.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--go_terms', required=True)
    ap.add_argument('--obo', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    args = ap.parse_args()
    import obonet
    graph = obonet.read_obo(args.obo)
    go_ids = json.loads(Path(args.go_terms).read_text())
    texts = build_go_texts(go_ids, dict(graph.nodes(data=True)))
    emb = encode(texts, args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, emb)
    print(f"[GO-TEXT] saved {args.out} shape={emb.shape} model={args.model}")


if __name__ == '__main__':
    main()
