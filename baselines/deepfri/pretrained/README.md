# DeepFRI pretrained weights

Not checked into git (~400 MB). Download from DeepFRI release:

```bash
# On Kaggle/Colab (Linux):
mkdir -p baselines/deepfri/pretrained
cd baselines/deepfri/pretrained
wget https://users.flatironinstitute.org/~vgligorijevic/public_www/DeepFRI_data/trained_models.tar.gz
tar xzf trained_models.tar.gz
rm trained_models.tar.gz
```

Expected files after extract:
- `DeepFRI-MERGED_MultiGraphConv_3x512_fcd_512_ca_10A_molecular_function.hdf5`
- `..._biological_process.hdf5`
- `..._cellular_component.hdf5`
- corresponding `_model_params.json` for each

Add `baselines/deepfri/pretrained/` to `.gitignore` (already done in repo).
