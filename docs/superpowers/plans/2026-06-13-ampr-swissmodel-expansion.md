# AMPR-Large: SWISS-MODEL Data Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train AMPR on DeepFRI's 220K SWISS-MODEL set (pretrain) then finetune on 30K PDB, using ESM-2 650M, to close the val→test generalization gap.

**Architecture:** Reuse the existing AMPR v3 pipeline unchanged (ESM residue → Transformer + attention-pool, GNN on contact maps, masked PPI, cross-modal fusion, ASL+DAG loss). Add only: (a) data-extraction from SWISS-MODEL TFRecords into the same artifact formats the v3 dataset already consumes, (b) a `--init-from` checkpoint flag enabling stage-2 finetune. Backbone drops 3B→650M (`seq.d_model` 2560→1280) so both stages share embedding dim.

**Tech Stack:** Python 3.12, PyTorch 2.3, TensorFlow 2.16 (TFRecord read only), h5py, numpy, PyYAML. Heavy runs on the 8×H200 server; unit tests on the Anaconda env (`python -m pytest`).

**Spec:** [docs/superpowers/specs/2026-06-13-ampr-swissmodel-expansion-design.md](../specs/2026-06-13-ampr-swissmodel-expansion-design.md)

**Data download:** `https://users.flatironinstitute.org/~renfrew/DeepFRI_data/SWISS-MODEL-GO.tar.gz` (also `PDB-GO.tar.gz` for the parity check).

---

## File Structure

**New files:**
- `scripts/inspect_tfrecord_schema.py` — print feature keys/shapes of N records (GATE diagnostic).
- `scripts/verify_label_parity.py` — cross-check TFRecord label vectors vs our `labels_*.npy` on shared PDB chains (GATE).
- `scripts/build_swissmodel_artifacts.py` — TFRecords → `sequences_sm.fasta`, `protein_order_sm.json`, `labels_{mf,bp,cc}_sm.npy`, `splits_sm.json`, `ppi_zero_sm.npy`, `ppi_mask_sm.npy`.
- `scripts/gen_v6_configs.py` — emit 3 pretrain + 9 finetune YAML configs from a template.
- `scripts/launch_v6_precompute.sh` — ESM-2 650M precompute for PDB + SWISS-MODEL across GPUs.
- `scripts/launch_v6_train.sh` — orchestrate pretrain→finetune (3 seeds/branch).
- `tests/test_build_swissmodel_artifacts.py` — unit tests for the extractor (synthetic TFRecord).
- `tests/test_load_checkpoint_weights.py` — unit test for the init-from helper.
- `tests/test_gen_v6_configs.py` — unit test the config generator.

**Modified files:**
- `ampr/training/trainer.py` — add `load_checkpoint_weights(model, path, map_location)`.
- `main.py` — add `--init-from` CLI arg; call helper in `_run_v3` before the epoch loop.

**Reused as-is:** `ampr/data/tfrecord_loader.py`, `scripts/tfrecord_to_h5.py` (cmaps), `scripts/precompute_esm2_residue.py` (default model is already 650M), `scripts/merge_residue_h5.py`, `ampr/data/dataset.py` (`AMPRDatasetV3`), `scripts/evaluate_stratified.py`, `scripts/ensemble_eval.py`.

---

## Task 0 (GATE): Confirm SWISS-MODEL TFRecord schema + label-order parity

No further coding proceeds until both checks pass. These run on the machine that holds the tar.gz files (server).

**Files:**
- Create: `scripts/inspect_tfrecord_schema.py`
- Create: `scripts/verify_label_parity.py`

- [ ] **Step 1: Write the schema inspector**

```python
#!/usr/bin/env python
"""Print feature keys, dtypes and lengths for the first N TFRecord examples."""
import argparse
import glob as _glob

import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. data/swissmodel/SWISS-MODEL_GO_train_*.tfrecords")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    files = sorted(_glob.glob(args.glob))
    print(f"[inspect] {len(files)} files match; reading {args.n} records from {files[0]}")
    ds = tf.data.TFRecordDataset(files[0])
    for k, raw in enumerate(ds.take(args.n)):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        print(f"--- record {k} ---")
        for key, feat in sorted(ex.features.feature.items()):
            kind = feat.WhichOneof("kind")
            vals = getattr(feat, kind).value
            print(f"  {key:18s} {kind:11s} len={len(vals)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the inspector on the SWISS-MODEL tar.gz contents**

Extract first: `mkdir -p data/swissmodel && tar -xzf SWISS-MODEL-GO.tar.gz -C data/swissmodel`
Run: `python scripts/inspect_tfrecord_schema.py --glob "data/swissmodel/**/*train*.tfrecords" --n 2`
Expected: keys include `L`, `prot_id`, `seq_1hot` (len = L*26), `ca_dist_matrix` and/or `cb_dist_matrix` (len = L*L), `mf_labels` (len 489), `bp_labels` (len 1943), `cc_labels` (len 320).
**Decision gate:** if keys match `ampr/data/tfrecord_loader.iter_tfrecord`, continue. If a key name differs (e.g. only `ca_dist_matrix`), note the actual name — `iter_tfrecord(path, dist_key=...)` already accepts it; record which `dist_key` to use downstream. If label lengths differ from 489/1943/320, STOP and escalate (label-space mismatch invalidates the whole approach).

- [ ] **Step 3: Write the label-parity verifier**

```python
#!/usr/bin/env python
"""Verify TFRecord label vectors use the SAME term order as our labels_*.npy.

Strategy: find PDB chains present in BOTH a PDB_GO TFRecord and our
data/pdbch/labels_{ont}.npy (via protein_order.json), and assert the binary
vectors are identical. If they match, SWISS-MODEL labels (same DeepFRI release)
can be consumed directly with no remap.
"""
import argparse
import glob as _glob
import json
from pathlib import Path

import numpy as np

from ampr.data.tfrecord_loader import iter_tfrecord


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-glob", required=True, help="PDB_GO_*.tfrecords glob (from PDB-GO.tar.gz)")
    ap.add_argument("--data-dir", default="data/pdbch")
    ap.add_argument("--dist-key", default="cb_dist_matrix")
    ap.add_argument("--max-check", type=int, default=200)
    args = ap.parse_args()

    dd = Path(args.data_dir)
    order = json.loads((dd / "protein_order.json").read_text())
    idx = {p: i for i, p in enumerate(order)}
    labs = {ont: np.load(dd / f"labels_{ont}.npy") for ont in ("mf", "bp", "cc")}

    checked = 0
    for tfp in sorted(_glob.glob(args.pdb_glob)):
        for rec in iter_tfrecord(Path(tfp), dist_key=args.dist_key):
            pid = rec["prot_id"]
            if pid not in idx:
                continue
            row = idx[pid]
            for ont in ("mf", "bp", "cc"):
                ours = (labs[ont][row] > 0.5).astype(np.int64)
                theirs = rec["labels"][ont].astype(np.int64)
                assert ours.shape == theirs.shape, f"{pid}/{ont}: shape {ours.shape} vs {theirs.shape}"
                if not np.array_equal(ours, theirs):
                    raise SystemExit(f"MISMATCH {pid}/{ont}: term order differs — remap needed")
            checked += 1
            if checked >= args.max_check:
                print(f"[parity] OK — {checked} chains match across mf/bp/cc; no remap needed")
                return
    print(f"[parity] OK — {checked} chains checked (all match)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the parity verifier**

Extract PDB set: `mkdir -p data/pdbgo && tar -xzf PDB-GO.tar.gz -C data/pdbgo`
Run: `python scripts/verify_label_parity.py --pdb-glob "data/pdbgo/**/*.tfrecords" --dist-key cb_dist_matrix`
Expected: `[parity] OK — 200 chains match across mf/bp/cc; no remap needed`
**Decision gate:** on MISMATCH, stop and add a remap task (build DeepFRI term-order → our term-order permutation from their goterms file) before continuing. On OK, the SWISS-MODEL labels are consumable directly.

- [ ] **Step 5: Commit**

```bash
git add scripts/inspect_tfrecord_schema.py scripts/verify_label_parity.py
git commit -m "feat(scripts): TFRecord schema inspector + label-order parity gate"
```

---

## Task 1: SWISS-MODEL artifact extractor

Converts SWISS-MODEL TFRecords into the npy/json/fasta artifacts that `AMPRDatasetV3` already reads. Contact maps are produced separately by the existing `tfrecord_to_h5.py` (Task 4).

**Files:**
- Create: `scripts/build_swissmodel_artifacts.py`
- Test: `tests/test_build_swissmodel_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for scripts/build_swissmodel_artifacts.py using a synthetic TFRecord."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write(path, pid, L, mf, bp, cc):
    seq_1hot = np.eye(26, dtype=np.float32)[[0] * L]  # all 'A'
    cmap = np.zeros((L, L), dtype=np.float32)
    def _f(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    def _i(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    def _b(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    feat = {
        "L": _i([L]), "prot_id": _b(pid.encode()),
        "seq_1hot": _f(seq_1hot.reshape(-1).tolist()),
        "ca_dist_matrix": _f(cmap.reshape(-1).tolist()),
        "cb_dist_matrix": _f(cmap.reshape(-1).tolist()),
        "mf_labels": _i(mf), "bp_labels": _i(bp), "cc_labels": _i(cc),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feat))
    with tf.io.TFRecordWriter(str(path)) as w:
        w.write(ex.SerializeToString())


def test_build_artifacts(tmp_path):
    from scripts.build_swissmodel_artifacts import build_artifacts
    train = tmp_path / "SM_train_0.tfrecords"
    valid = tmp_path / "SM_valid_0.tfrecords"
    _write(train, "P1-A", 3, [1, 0] + [0] * 487, [0] * 1943, [1] + [0] * 319)
    _write(valid, "P2-A", 4, [0, 1] + [0] * 487, [1] + [0] * 1942, [0] * 320)
    out = tmp_path / "out"

    build_artifacts(str(train.parent / "SM_train_*.tfrecords"),
                    str(valid.parent / "SM_valid_*.tfrecords"),
                    str(out), n_terms={"mf": 489, "bp": 1943, "cc": 320})

    order = json.loads((out / "protein_order_sm.json").read_text())
    assert order == ["P1-A", "P2-A"]
    splits = json.loads((out / "splits_sm.json").read_text())
    assert splits["train"] == ["P1-A"] and splits["valid"] == ["P2-A"]
    mf = np.load(out / "labels_mf_sm.npy")
    assert mf.shape == (2, 489) and mf[0, 0] == 1.0 and mf[1, 1] == 1.0
    mask = np.load(out / "ppi_mask_sm.npy")
    ppi = np.load(out / "ppi_zero_sm.npy")
    assert mask.shape == (2,) and mask.sum() == 0
    assert ppi.shape == (2, 256) and ppi.sum() == 0
    fasta = (out / "sequences_sm.fasta").read_text()
    assert ">P1-A" in fasta and "AAA" in fasta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_swissmodel_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: cannot import name 'build_artifacts'`.

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python
"""Extract SWISS-MODEL TFRecords into AMPRDatasetV3 artifacts.

Outputs (in --out dir):
    protein_order_sm.json   list[str]            row order shared by all npy below
    splits_sm.json          {"train": [...], "valid": [...]}
    labels_{mf,bp,cc}_sm.npy  (N, C) float32     binary GO labels
    ppi_zero_sm.npy         (N, 256) float32     all zeros (SWISS-MODEL lacks PPI)
    ppi_mask_sm.npy         (N,) bool            all False
    sequences_sm.fasta      FASTA                for ESM-2 precompute

Contact maps are NOT produced here — run scripts/tfrecord_to_h5.py separately.
"""
import argparse
import glob as _glob
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ampr.data.tfrecord_loader import iter_tfrecord

logging.basicConfig(level=logging.INFO, format="[sm_artifacts] %(message)s")
log = logging.getLogger(__name__)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYBOUXZ-."


def _onehot_to_seq(M: np.ndarray) -> str:
    return "".join(ALPHABET[i] for i in M.argmax(axis=-1))


def build_artifacts(train_glob, valid_glob, out_dir, n_terms, dist_key="cb_dist_matrix"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    order, split_of, seqs = [], {}, {}
    labels = {ont: [] for ont in ("mf", "bp", "cc")}
    seen = set()

    for split, pattern in (("train", train_glob), ("valid", valid_glob)):
        files = sorted(_glob.glob(pattern))
        log.info("%s: %d TFRecord files", split, len(files))
        for tfp in files:
            for rec in iter_tfrecord(Path(tfp), dist_key=dist_key):
                pid = rec["prot_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                order.append(pid)
                split_of[pid] = split
                seqs[pid] = _onehot_to_seq(rec["seq_1hot"])
                for ont in ("mf", "bp", "cc"):
                    v = rec["labels"][ont].astype(np.float32)
                    assert v.shape[0] == n_terms[ont], f"{pid}/{ont}: {v.shape[0]} != {n_terms[ont]}"
                    labels[ont].append(v)
                if len(order) % 5000 == 0:
                    log.info("  processed %d chains", len(order))

    (out / "protein_order_sm.json").write_text(json.dumps(order))
    (out / "splits_sm.json").write_text(json.dumps({
        "train": [p for p in order if split_of[p] == "train"],
        "valid": [p for p in order if split_of[p] == "valid"],
    }))
    for ont in ("mf", "bp", "cc"):
        np.save(out / f"labels_{ont}_sm.npy", np.stack(labels[ont]).astype(np.float32))
    N = len(order)
    np.save(out / "ppi_zero_sm.npy", np.zeros((N, 256), dtype=np.float32))
    np.save(out / "ppi_mask_sm.npy", np.zeros((N,), dtype=bool))
    with open(out / "sequences_sm.fasta", "w") as f:
        for pid in order:
            f.write(f">{pid}\n{seqs[pid]}\n")
    log.info("done — %d chains (%d train / %d valid)", N,
             sum(v == "train" for v in split_of.values()),
             sum(v == "valid" for v in split_of.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-glob", required=True)
    ap.add_argument("--valid-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-key", default="cb_dist_matrix")
    args = ap.parse_args()
    build_artifacts(args.train_glob, args.valid_glob, args.out,
                    n_terms={"mf": 489, "bp": 1943, "cc": 320}, dist_key=args.dist_key)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_swissmodel_artifacts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_swissmodel_artifacts.py tests/test_build_swissmodel_artifacts.py
git commit -m "feat(scripts): SWISS-MODEL TFRecord -> AMPRDatasetV3 artifacts"
```

---

## Task 2: `--init-from` checkpoint flag (enables stage-2 finetune)

**Files:**
- Modify: `ampr/training/trainer.py` (add helper near top-level, after imports)
- Modify: `main.py:418-432` (argparse) and `main.py:148-152` (inside `_run_v3`, after `model = model.to(device)`)
- Test: `tests/test_load_checkpoint_weights.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test the init-from checkpoint loader helper."""
import torch
import torch.nn as nn

from ampr.training.trainer import load_checkpoint_weights


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_loads_model_state(tmp_path):
    src = Tiny()
    with torch.no_grad():
        src.fc.weight.fill_(1.234)
    ckpt = tmp_path / "best.pt"
    torch.save({"epoch": 7, "model": src.state_dict()}, ckpt)

    dst = Tiny()
    info = load_checkpoint_weights(dst, str(ckpt), map_location="cpu")

    assert torch.allclose(dst.fc.weight, torch.full_like(dst.fc.weight, 1.234))
    assert info["epoch"] == 7
    assert info["missing"] == [] and info["unexpected"] == []


def test_accepts_bare_state_dict(tmp_path):
    src = Tiny()
    ckpt = tmp_path / "bare.pt"
    torch.save(src.state_dict(), ckpt)
    dst = Tiny()
    info = load_checkpoint_weights(dst, str(ckpt), map_location="cpu")
    assert info["epoch"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_load_checkpoint_weights.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_checkpoint_weights'`.

- [ ] **Step 3: Add the helper to `ampr/training/trainer.py`**

Add at module top level (after the existing imports):

```python
def load_checkpoint_weights(model, path, map_location="cpu"):
    """Load weights from a checkpoint into `model` for stage-2 finetune.

    Accepts either a dict with a 'model' key (our save format) or a bare
    state_dict. Returns {'epoch', 'missing', 'unexpected'} for logging.
    """
    import torch
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return {"epoch": epoch, "missing": list(missing), "unexpected": list(unexpected)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_load_checkpoint_weights.py -v`
Expected: PASS.

- [ ] **Step 5: Wire `--init-from` into `main.py`**

In `main()` argparse block (after the `--dry_run` argument near `main.py:428`):

```python
    parser.add_argument('--init-from', type=str, default=None,
                        help='v3: load model weights from this checkpoint before training (stage-2 finetune)')
```

In `_run_v3`, immediately after `model = model.to(device)` (around `main.py:148`):

```python
    if getattr(args, 'init_from', None):
        from ampr.training.trainer import load_checkpoint_weights
        info = load_checkpoint_weights(model, args.init_from, map_location=device)
        log.info(f"[V3] init-from {args.init_from}: loaded "
                 f"(pretrain epoch={info['epoch']}, "
                 f"missing={len(info['missing'])}, unexpected={len(info['unexpected'])})")
```

- [ ] **Step 6: Smoke-check the flag end-to-end (CPU dry-run, optional but recommended)**

Run (only if v3 smoke fixtures are available locally; otherwise defer to server Task 6):
`python -m pytest tests/test_trainer_v3_smoke.py -v`
Expected: PASS (unchanged — confirms the new branch did not break the v3 path).

- [ ] **Step 7: Commit**

```bash
git add ampr/training/trainer.py main.py tests/test_load_checkpoint_weights.py
git commit -m "feat(train): --init-from checkpoint flag for stage-2 finetune"
```

---

## Task 3: v6 config generator

Emits 3 pretrain configs (one per branch, seed 42) and 9 finetune configs (3 branches × 3 seeds). Pretrain runs once per branch on 220K; each finetune initializes from that single pretrained checkpoint via `--init-from` (passed in the launch script, Task 7).

**Files:**
- Create: `scripts/gen_v6_configs.py`
- Test: `tests/test_gen_v6_configs.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test scripts/gen_v6_configs.py emits valid pretrain + finetune configs."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_generates_configs(tmp_path):
    from scripts.gen_v6_configs import generate
    paths = generate(out_dir=str(tmp_path), sm_dir="data/swissmodel_art",
                     pdb_emb="data/embeddings/esm2_650m_pdb.h5",
                     sm_emb="data/embeddings/esm2_650m_sm.h5",
                     sm_cmap="data/swissmodel_art/cmap_all_sm.h5")
    # 3 pretrain + 9 finetune
    assert len(paths) == 12
    pre = yaml.safe_load(Path(tmp_path / "mf_v6sm_pretrain.yaml").read_text())
    assert pre["branch"] == "MF" and pre["n_terms"] == 489
    assert pre["model"]["seq"]["d_model"] == 1280            # 650M, not 2560
    assert pre["data"]["esm2_h5"].endswith("esm2_650m_sm.h5")
    assert pre["data"]["splits"].endswith("splits_sm.json")
    ft = yaml.safe_load(Path(tmp_path / "bp_v6sm_finetune_s123.yaml").read_text())
    assert ft["branch"] == "BP" and ft["n_terms"] == 1943
    assert ft["seed"] == 123
    assert ft["data"]["esm2_h5"].endswith("esm2_650m_pdb.h5")
    assert ft["data"]["splits"].endswith("pdbch/splits.json")
    assert "s123" in ft["output"]["checkpoint_dir"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_v6_configs.py -v`
Expected: FAIL — cannot import `generate`.

- [ ] **Step 3: Write the generator**

```python
#!/usr/bin/env python
"""Generate v6 SWISS-MODEL pretrain + PDB finetune configs.

Pretrain: 3 configs (mf/bp/cc), seed 42, train on SWISS-MODEL artifacts.
Finetune: 9 configs (3 branches × seeds {42,123,2024}), train on PDB at 650M,
          initialized from the matching pretrain checkpoint (via --init-from).
"""
import argparse
from pathlib import Path

import yaml

BRANCHES = {"mf": ("MF", 489), "bp": ("BP", 1943), "cc": ("CC", 320)}
SEEDS = [42, 123, 2024]


def _common_model():
    return {
        "version": "v3",
        "structure_modality": "gnn",
        "seq": {"d_model": 1280, "n_transformer_layers": 2, "n_heads": 8, "dropout": 0.4},
        "gnn": {"node_dim": 256, "n_layers": 3, "cmap_threshold": 10.0},
        "ppi": {"in_dim": 256, "hidden": 512},
        "fusion": {"d_model": 512, "n_layers": 2, "n_heads": 8},
        "classifier": "both",
        "d_hidden": 512,
    }


def _training(epochs, lr, seed):
    return {
        "epochs": epochs, "batch_size": 64, "lr": lr,
        "lr_scheduler": "plateau", "lr_factor": 0.5, "lr_patience": 5, "lr_min": 1.0e-5,
        "weight_decay": 1.0e-2, "loss_type": "asl",
        "asl_gamma_neg": 4, "asl_gamma_pos": 0, "asl_clip": 0.05,
        "lambda_dag": 0.5, "seed": seed, "device": "auto",
        "max_seq_len": 1000, "num_workers": 16,
    }


def generate(out_dir, sm_dir, pdb_emb, sm_emb, sm_cmap):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for short, (branch, n_terms) in BRANCHES.items():
        # --- pretrain (SWISS-MODEL) ---
        ck = f"checkpoints/{short}_v6sm_pretrain/"
        cfg = {
            "branch": branch, "n_terms": n_terms, "seed": 42,
            "data": {
                "protein_order": f"{sm_dir}/protein_order_sm.json",
                "splits":        f"{sm_dir}/splits_sm.json",
                "labels":        f"{sm_dir}/labels_{short}_sm.npy",
                "dag_matrix":    f"data/pdbch/dag_matrix_{short}.npy",
                "esm2_h5":       sm_emb,
                "ppi_emb":       f"{sm_dir}/ppi_zero_sm.npy",
                "ppi_mask":      f"{sm_dir}/ppi_mask_sm.npy",
                "cmap_h5":       sm_cmap,
                "go_emb":        f"data/embeddings/go_emb_{short}_v2.npy",
            },
            "model": _common_model(),
            "training": _training(epochs=30, lr=1.0e-3, seed=42),
            "inference": {"use_dag_propagation": True},
            "output": {
                "checkpoint_dir": ck,
                "log_file": f"logs/{short}_v6sm_pretrain.log",
                "results_file": f"results/{short}_v6sm_pretrain_predictions.tsv",
            },
        }
        p = out / f"{short}_v6sm_pretrain.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(str(p))

        # --- finetune (PDB), one per seed ---
        for seed in SEEDS:
            ck = f"checkpoints/{short}_v6sm_finetune_s{seed}/"
            cfg = {
                "branch": branch, "n_terms": n_terms, "seed": seed,
                "data": {
                    "protein_order": "data/pdbch/protein_order.json",
                    "splits":        "data/pdbch/splits.json",
                    "labels":        f"data/pdbch/labels_{short}.npy",
                    "dag_matrix":    f"data/pdbch/dag_matrix_{short}.npy",
                    "esm2_h5":       pdb_emb,
                    "ppi_emb":       "data/embeddings/ppi_deepgo.npy",
                    "ppi_mask":      "data/embeddings/ppi_deepgo_mask.npy",
                    "cmap_h5":       "data/contact_maps/cmap_all.h5",
                    "go_emb":        f"data/embeddings/go_emb_{short}_v2.npy",
                    "diamond_tsv":   f"data/diamond/diamond_results_{short}.tsv",
                },
                "model": _common_model(),
                "training": _training(epochs=30, lr=1.0e-4, seed=seed),  # low LR for finetune
                "inference": {
                    "use_dag_propagation": True, "use_diamond_ensemble": True,
                    "diamond_alpha": 0.6,
                    "threshold_path": f"{ck}threshold.json",
                },
                "output": {
                    "checkpoint_dir": ck,
                    "log_file": f"logs/{short}_v6sm_finetune_s{seed}.log",
                    "results_file": f"results/{short}_v6sm_finetune_s{seed}_predictions.tsv",
                },
            }
            p = out / f"{short}_v6sm_finetune_s{seed}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            paths.append(str(p))

    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs")
    ap.add_argument("--sm-dir", default="data/swissmodel_art")
    ap.add_argument("--pdb-emb", default="data/embeddings/esm2_650m_pdb.h5")
    ap.add_argument("--sm-emb", default="data/embeddings/esm2_650m_sm.h5")
    ap.add_argument("--sm-cmap", default="data/swissmodel_art/cmap_all_sm.h5")
    args = ap.parse_args()
    paths = generate(args.out, args.sm_dir, args.pdb_emb, args.sm_emb, args.sm_cmap)
    print(f"[gen_v6] wrote {len(paths)} configs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_v6_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Generate the real configs and commit**

```bash
python scripts/gen_v6_configs.py --out configs
git add scripts/gen_v6_configs.py tests/test_gen_v6_configs.py configs/*_v6sm_*.yaml
git commit -m "feat(configs): v6 SWISS-MODEL pretrain + PDB finetune config generator"
```

---

## Task 4: Extract SWISS-MODEL data on the server

Operational (server). Produces the non-cmap artifacts and the cmap HDF5.

**Files:** none (runs existing scripts).

- [ ] **Step 1: Build non-cmap artifacts**

Run: `python scripts/build_swissmodel_artifacts.py --train-glob "data/swissmodel/**/*train*.tfrecords" --valid-glob "data/swissmodel/**/*valid*.tfrecords" --out data/swissmodel_art --dist-key cb_dist_matrix`
Expected log tail: `done — ~220297 chains (~220297 train / ~24478 valid)` (exact counts per DeepFRI Supp. Table 1).

- [ ] **Step 2: Sanity-check artifact shapes**

Run: `python -c "import numpy as np,json; o=json.load(open('data/swissmodel_art/protein_order_sm.json')); print('N',len(o)); print('mf',np.load('data/swissmodel_art/labels_mf_sm.npy').shape)"`
Expected: `N` ≈ 244,775 (train+valid); `mf (N, 489)`.

- [ ] **Step 3: Build SWISS-MODEL contact-map HDF5 (reuse existing script)**

Run: `python scripts/tfrecord_to_h5.py --input-glob "data/swissmodel/**/*.tfrecords" --out data/swissmodel_art/cmap_all_sm.h5`
Expected log tail: `done — ~244775 cmaps in data/swissmodel_art/cmap_all_sm.h5`.

- [ ] **Step 4: Commit the small JSON artifacts (npy/h5 are gitignored)**

```bash
git add data/swissmodel_art/splits_sm.json data/swissmodel_art/protein_order_sm.json
git commit -m "data: SWISS-MODEL splits + protein_order (npy/h5 gitignored)"
```
(If `data/swissmodel_art/` is not yet ignored for large files, add `*.npy`, `*.h5`, `*.fasta` under it to `.gitignore` first.)

---

## Task 5: Precompute ESM-2 650M embeddings (server, multi-GPU)

Operational. Re-embeds PDB at 650M (so both stages share dim 1280) and embeds SWISS-MODEL at 650M.

**Files:**
- Create: `scripts/launch_v6_precompute.sh`

- [ ] **Step 1: Write the launch script**

```bash
#!/usr/bin/env bash
# scripts/launch_v6_precompute.sh — ESM-2 650M residue embeddings (PDB + SWISS-MODEL).
# PDB (~37K, 1 GPU) + SWISS-MODEL (~245K, sharded across GPUs). Each shard own HDF5; merge after.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
MODEL=facebook/esm2_t33_650M_UR50D
mkdir -p logs data/embeddings

# --- PDB on GPU 0 ---
tmux kill-session -t esm650_pdb 2>/dev/null || true
tmux new-session -d -s esm650_pdb \
  "CUDA_VISIBLE_DEVICES=0 python scripts/precompute_esm2_residue.py \
     --fasta data/pdbch/sequences.fasta --protein_order data/pdbch/protein_order.json \
     --out data/embeddings/esm2_650m_pdb.h5 --model $MODEL --batch 16 --max_len 1022 \
     2>&1 | tee logs/esm650_pdb.log"

# --- SWISS-MODEL sharded across GPUs 1-7 ---
N=7
for i in $(seq 0 6); do
  gpu=$((i + 1))
  tmux kill-session -t "esm650_sm_$i" 2>/dev/null || true
  tmux new-session -d -s "esm650_sm_$i" \
    "CUDA_VISIBLE_DEVICES=$gpu python scripts/precompute_esm2_residue.py \
       --fasta data/swissmodel_art/sequences_sm.fasta \
       --protein_order data/swissmodel_art/protein_order_sm.json \
       --out data/embeddings/esm2_650m_sm_shard$i.h5 \
       --model $MODEL --batch 16 --max_len 1022 --shard $i --nshards $N \
       2>&1 | tee logs/esm650_sm_shard$i.log"
  echo "[ESM650] SM shard $i -> GPU $gpu"
done
echo "[ESM650] watch: tail -f logs/esm650_*.log"
echo "[ESM650] after all sessions end, merge SM shards:"
echo "  python scripts/merge_residue_h5.py --shards data/embeddings/esm2_650m_sm_shard*.h5 --out data/embeddings/esm2_650m_sm.h5"
```

- [ ] **Step 2: Run precompute**

Run: `bash scripts/launch_v6_precompute.sh`
Monitor: `tail -f logs/esm650_*.log` until each prints `DONE — <count> proteins`.

- [ ] **Step 3: Merge SWISS-MODEL shards**

Run: `python scripts/merge_residue_h5.py --shards data/embeddings/esm2_650m_sm_shard*.h5 --out data/embeddings/esm2_650m_sm.h5`
Expected: final count ≈ 244,775 keys.

- [ ] **Step 4: Verify key coverage vs splits**

Run: `python -c "import h5py,json; k=set(h5py.File('data/embeddings/esm2_650m_sm.h5','r').keys()); s=json.load(open('data/swissmodel_art/splits_sm.json')); print('train cov', sum(p in k for p in s['train'])/len(s['train'])); print('valid cov', sum(p in k for p in s['valid'])/len(s['valid']))"`
Expected: both coverages ≈ 1.0 (a few long-sequence drops are acceptable — `AMPRDatasetV3` filters missing keys).

- [ ] **Step 5: Commit the launch script**

```bash
git add scripts/launch_v6_precompute.sh
git commit -m "feat(scripts): ESM-2 650M precompute launcher (PDB + SWISS-MODEL)"
```

---

## Task 6: Dry-run smoke of the v6 pipeline (server)

Validates configs + dataset wiring before the long runs.

**Files:** none.

- [ ] **Step 1: Dry-run pretrain config**

Run: `python main.py --config configs/mf_v6sm_pretrain.yaml --dry_run`
Expected: log shows `[V3] Detected v3 config`, `[DATASET-V3] MF/train: 50 proteins`, one epoch completes, `[V3] Saved best checkpoint`. No shape errors.

- [ ] **Step 2: Dry-run finetune config with --init-from**

Run: `python main.py --config configs/mf_v6sm_finetune_s42.yaml --dry_run --init-from checkpoints/mf_v6sm_pretrain/best.pt`
Expected: log shows `[V3] init-from ...: loaded (pretrain epoch=1, missing=0, unexpected=0)` then one finetune epoch. `missing=0, unexpected=0` confirms architecture identity between stages.

- [ ] **Step 2b (decision gate):** If `missing`/`unexpected` > 0, the two stages' architectures diverge — stop and reconcile config (`seq.d_model`, `classifier`, `go_emb_dim`) before full runs.

---

## Task 7: Two-stage training (server)

Pretrain once per branch on SWISS-MODEL; finetune 3 seeds per branch on PDB from that checkpoint.

**Files:**
- Create: `scripts/launch_v6_train.sh`

- [ ] **Step 1: Write the orchestration script**

```bash
#!/usr/bin/env bash
# scripts/launch_v6_train.sh — stage1 pretrain (220K SM) then stage2 finetune (30K PDB, 3 seeds).
# Runs branches in parallel across GPUs; finetune seeds sequentially per branch.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

run_branch() {  # $1=short  $2=gpu
  local short="$1" gpu="$2"
  # Stage 1: pretrain (seed 42)
  CUDA_VISIBLE_DEVICES="$gpu" python main.py --config "configs/${short}_v6sm_pretrain.yaml" \
    2>&1 | tee "logs/${short}_v6sm_pretrain.run.log"
  local pre="checkpoints/${short}_v6sm_pretrain/best.pt"
  # Stage 2: finetune from the pretrained checkpoint, 3 seeds
  for seed in 42 123 2024; do
    CUDA_VISIBLE_DEVICES="$gpu" python main.py \
      --config "configs/${short}_v6sm_finetune_s${seed}.yaml" \
      --seed "$seed" --init-from "$pre" \
      2>&1 | tee "logs/${short}_v6sm_finetune_s${seed}.run.log"
  done
}

tmux kill-session -t v6_mf 2>/dev/null || true; tmux new-session -d -s v6_mf "$(declare -f run_branch); run_branch mf 0"
tmux kill-session -t v6_bp 2>/dev/null || true; tmux new-session -d -s v6_bp "$(declare -f run_branch); run_branch bp 1"
tmux kill-session -t v6_cc 2>/dev/null || true; tmux new-session -d -s v6_cc "$(declare -f run_branch); run_branch cc 2"
echo "[v6] launched mf/bp/cc on GPUs 0/1/2 — watch logs/*_v6sm_*.run.log"
```

- [ ] **Step 2: Launch training**

Run: `bash scripts/launch_v6_train.sh`
Monitor: `grep val_Fmax_dag logs/mf_v6sm_pretrain.run.log | tail`
Expected: pretrain val Fmax climbs; finetune logs show `init-from ... missing=0`.

- [ ] **Step 3: Confirm checkpoints exist**

Run: `ls checkpoints/*_v6sm_finetune_s*/best.pt`
Expected: 9 files (3 branches × 3 seeds).

- [ ] **Step 4: Commit the launch script**

```bash
git add scripts/launch_v6_train.sh
git commit -m "feat(scripts): v6 two-stage pretrain->finetune launcher"
```

---

## Task 8: Evaluation + comparison table (server)

Reuse the existing stratified evaluator and ensemble script across all identity bins.

**Files:** none (uses `scripts/evaluate_stratified.py`, `scripts/ensemble_eval.py`, `main.py --eval-only`).

- [ ] **Step 1: Per-seed stratified eval on all bins**

Run for each finetune checkpoint, e.g.:
`python main.py --config configs/mf_v6sm_finetune_s42.yaml --eval-only --checkpoint checkpoints/mf_v6sm_finetune_s42/best.pt --test-split test_LT_30 --tune-alpha`
Repeat for splits `test test_LT_30 test_LT_40 test_LT_50 test_LT_70 test_LT_95` and for bp/cc. (The existing `scripts/eval_all_v3.sh` pattern can be adapted by pointing at the `_v6sm_finetune_*` configs.)
Expected: each writes `results/..._v6sm_finetune_*.eval_<split>.json` with `raw`, `dag`, `ensemble` blocks.

- [ ] **Step 2: 3-seed ensemble per branch**

Run: `python scripts/ensemble_eval.py` adapted to the `_v6sm_finetune_s{42,123,2024}` checkpoints (same invocation the current 3-seed ensemble uses; see `scripts/ensemble_eval.py --help`).
Expected: ensembled Fmax/Smin/AUPRC per branch per bin.

- [ ] **Step 3: Build the comparison table**

Assemble a markdown table comparing, per branch and per identity bin:
`DeepFRI-MERGED` (from `data/deepfri_baseline.json`) vs `AMPR-PDB-30K` (current best, from existing results) vs `AMPR pretrain-only-220K` (stage-1 eval) vs `AMPR-B (pretrain→finetune, 3-seed ens)`.
Save to `docs/PHASE_V6_RESULTS.md`.

- [ ] **Step 4: Verdict against success criteria**

Confirm against spec §1: test Fmax ≥ AMPR-PDB-30K on every bin; targets MF > 0.66, BP > 0.55, CC ≥ DeepFRI. Record the val→test gap (val Fmax_dag from training log minus test Fmax) vs the 30K baseline gap. If criteria unmet, record the negative result per spec §7 (scaling saturated due to homology-model noise).

- [ ] **Step 5: Commit results**

```bash
git add docs/PHASE_V6_RESULTS.md results/*_v6sm_*.json
git commit -m "docs(results): AMPR-Large SWISS-MODEL expansion — eval + comparison"
```

---

## Task 9 (OPTIONAL): Data-scaling curve

Only if time remains after Task 8. Produces the thesis figure of Fmax vs training-set size.

**Files:** none (reuses configs with a subsampled split).

- [ ] **Step 1: Make subsampled SWISS-MODEL splits**

Run: `python -c "import json,random; s=json.load(open('data/swissmodel_art/splits_sm.json')); random.seed(0); tr=s['train']; [json.dump({'train':random.sample(tr,n),'valid':s['valid']}, open(f'data/swissmodel_art/splits_sm_{n}.json','w')) for n in (30000,60000,120000)]"`
Expected: three new split files.

- [ ] **Step 2: Pretrain MF at each size, finetune seed 42, eval `test`**

For each `n in 30000 60000 120000 220000`: point a copy of `mf_v6sm_pretrain.yaml` at `splits_sm_{n}.json` (220000 = full), run pretrain→finetune→eval as in Tasks 7–8.
Expected: one `test` Fmax per size.

- [ ] **Step 3: Plot and commit**

Plot Fmax vs log(n) (MF; optionally BP/CC). Save `docs/figures/v6_scaling_curve.png` and append to `docs/PHASE_V6_RESULTS.md`.

```bash
git add docs/figures/v6_scaling_curve.png docs/PHASE_V6_RESULTS.md
git commit -m "docs(results): SWISS-MODEL data-scaling curve"
```

---

## Self-Review notes

- **Spec coverage:** Component 1 (extraction) → Tasks 0,1,4; Component 2 (ESM-650M precompute, PPI mask, cmap) → Tasks 4,5; Component 3 (architecture unchanged, d_model 1280) → Task 3 generator; Component 4 (two-stage) → Tasks 2,7; Component 5 (eval pipeline) → Task 8; Component 6 (scaling curve, per-branch) → Tasks 8,9; Component 7 (risks) → GATEs in Tasks 0,6 + verdict in Task 8.
- **GATE ordering:** Task 0 (schema + label parity) must pass before any extraction; Task 6 dry-run (missing/unexpected=0) must pass before the long Task 7 runs.
- **Type/name consistency:** `build_artifacts(...)` signature, artifact filenames (`*_sm.npy/json/fasta`), config keys (`esm2_h5`, `ppi_mask`, `cmap_h5`, `seq.d_model`), and `load_checkpoint_weights(...)` return keys (`epoch/missing/unexpected`) are used identically across tasks and tests.
- **Known operational caveat:** Tasks 4/5/7/8 run on the H200 server and cannot be pytest-verified at 220K scale; each has an explicit expected-output check instead.
```
