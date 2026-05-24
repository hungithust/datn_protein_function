# DeepFRI Baseline

Runs DeepFRI pretrained model on AMPR's test set to reproduce HEAL Table S3.2 numbers.

## Prerequisites

1. Download pretrained weights (see `pretrained/README.md`)
2. Clone DeepFRI repo to repo root: `git clone https://github.com/flatironinstitute/DeepFRI DeepFRI`
3. Generate contact maps: `python baselines/pdb_to_cmap.py`

## Run (on Kaggle/Colab GPU)

```bash
for ONT in mf bp cc; do
  ONTNAME=$(echo $ONT | sed 's/mf/molecular_function/;s/bp/biological_process/;s/cc/cellular_component/')
  python baselines/deepfri/run_predict.py \
    --ontology $ONT \
    --weights baselines/deepfri/pretrained/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_512_ca_10A_${ONTNAME}.hdf5 \
    --labels data/pdbch/labels_${ONT}.npy \
    --go-terms data/pdbch/go_terms_${ONT}.json \
    --out results/baselines/deepfri/predictions_${ONT}.npz
done
```

Then compute metrics:

```bash
for ONT in mf bp cc; do
  python baselines/compute_metrics.py \
    --predictions results/baselines/deepfri/predictions_${ONT}.npz \
    --labels data/pdbch/labels_${ONT}.npy \
    --out results/baselines/deepfri/metrics_${ONT}.json
done
```
