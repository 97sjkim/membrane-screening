# Membrane Screening

Code and data for benchmarking eight multitask machine-learning models for polymer gas-permeability prediction.


## Contents

```text
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

## Data

This repository does not redistribute the dataset.

The data used in this study were obtained directly from the publicly available files released by Yang et al. in the following repository:

* [PolymerGasMembraneML](https://github.com/jsunn-y/PolymerGasMembraneML)

The processed dataset used in this work contains 353 unique polymer structures divided into fixed training, validation, and test sets.

The prediction targets are the log-transformed pure-gas permeabilities,

`log10(P / Barrer)`,

for the following gases:

`He`, `H2`, `O2`, `N2`, `CO2`, and `CH4`.


Please cite the original publication when using these data:

> J. Yang et al., “Machine learning enables interpretable discovery of innovative polymers for gas separation membranes,” *Science Advances*, vol. 8, no. 29, eabn9545, 2022.
> https://doi.org/10.1126/sciadv.abn9545



## Citation

Citation information for the associated paper will be added after publication.

## License

The source code is released under the MIT License. 
