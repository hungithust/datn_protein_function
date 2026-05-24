# PDBch dataset (= DeepFRI's nrPDB-GO_2019.06.18)

Single source of truth for AMPR's training data. Matches HEAL paper's "PDBch" splits.

## Source files (checked in, immutable)

| File | Source |
|---|---|
| `train.txt` (29,902 IDs) | `DeepFRI/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_train.txt` |
| `valid.txt` (3,323 IDs)  | `DeepFRI/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_valid.txt` |
| `test.csv` (3,416 rows with 5 identity-bin columns) | `DeepFRI/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_test.csv` |
| `annot.tsv` | `DeepFRI/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv` |
| `sequences.fasta` | `DeepFRI/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_sequences.fasta` |

## Generated artifacts (ignored by git; reproducible)

| File | Generator |
|---|---|
| `go-basic.obo` | `Invoke-WebRequest http://release.geneontology.org/2019-06-01/ontology/go-basic.obo` |
| `splits.json` | `python scripts/build_splits_from_deepfri.py --data-dir data/pdbch` |
| `labels_{mf,bp,cc}.npy` + `protein_order.json` + `go_terms_{mf,bp,cc}.json` | `python scripts/build_labels_from_annot.py --data-dir data/pdbch` |
| `dag_matrix_{mf,bp,cc}.npy` | `python scripts/build_dag_from_obo.py --data-dir data/pdbch` |

## One-shot rebuild

```powershell
Invoke-WebRequest -Uri "http://release.geneontology.org/2019-06-01/ontology/go-basic.obo" -OutFile data/pdbch/go-basic.obo
python scripts/build_splits_from_deepfri.py --data-dir data/pdbch
python scripts/build_labels_from_annot.py --data-dir data/pdbch
python scripts/build_dag_from_obo.py --data-dir data/pdbch
pytest tests/test_pdbch_smoke.py -v
```

## Identity-bin convention

`test_LT_X` = every test protein whose `<X%>` column in `test.csv` equals `1`. Bins are **cumulative**, matching HEAL paper Table S3.2:

```
test_LT_30 ⊂ test_LT_40 ⊂ test_LT_50 ⊂ test_LT_70 ⊂ test_LT_95 ⊆ test
```

When comparing AMPR results with HEAL/DeepFRI Fmax numbers, always use the same column name (`<30%` ⇄ `test_LT_30`).
