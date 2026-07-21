# Membrane Screening

Code and data for benchmarking eight multitask machine-learning models for polymer gas-permeability prediction.

This repository contains only the machine-learning benchmark used in the study. Process optimization, techno-economic analysis, cost-surrogate modeling, and large-scale candidate-screening codes are not included.

## Contents

```text
data/
  train_9of10.csv
  val_1of10.csv
  fixed_test_10pct.csv

model/
  01_rf_fp_multitask.py
  02_xgb_fp_multitask.py
  03_dnn_fp_multitask.py
  04_gcn_multitask.py
  05_gin_multitask.py
  06_gat_multitask.py
  07_chemprop_mpnn_multitask.py
  08_polybert_multitask.py
```

## Data

The dataset contains 353 unique polymer structures divided into fixed training, validation, and test sets.

Targets are the log-transformed pure-gas permeabilities, `log10(P / Barrer)`, for:

`He`, `H2`, `O2`, `N2`, `CO2`, and `CH4`.

The data were derived from:

> J. Yang et al., *Science Advances* 8 (2022), eabn9545.  
> https://doi.org/10.1126/sciadv.abn9545

## Models

- Random Forest
- XGBoost
- Deep Neural Network
- GCN
- GIN
- GAT
- Chemprop
- PolyBERT

All models are trained as six-target multitask regressors using the same fixed data split.

## Usage

Install the required packages and run a script from the repository root:

```bash
python model/01_rf_fp_multitask.py
```

Replace the filename to run another model.

## Citation

Citation information for the associated paper will be added after publication.

## License

The source code is released under the MIT License. The dataset is derived from previously published data and may be subject to separate redistribution terms.
