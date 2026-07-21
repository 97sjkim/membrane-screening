# Configuration
TRAIN_CSV = 'train_9of10.csv'
VAL_CSV = 'val_1of10.csv'
TEST_CSV = 'fixed_test_10pct.csv'
OUT_ROOT = './benchmark_outputs_fixedsplit_multitask'
MODEL_NAME = 'xgb_fp_multitask'
SEED = 0
N_MEMBERS = 10
TARGETS = ['He', 'H2', 'O2', 'N2', 'CO2', 'CH4']
TARGET_TRANSFORM = 'none'
USE_TARGET_SCALING = True
BAGGING = True
SAVE_MODELS = True
# Dependencies
import os, sys, json, time, random, hashlib, itertools, platform
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Utilities
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: str | Path, obj: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def sha256_df_content(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(('shape=' + str(df.shape)).encode('utf-8'))
    h.update(('cols=' + '|'.join(map(str, df.columns))).encode('utf-8'))
    h.update(df.to_csv(index=False).encode('utf-8'))
    return h.hexdigest()

def collect_versions() -> Dict[str, str]:
    import importlib.metadata as im
    pkgs = ['numpy', 'pandas', 'scikit-learn', 'joblib', 'xgboost', 'torch', 'rdkit', 'torch-geometric', 'transformers']
    out = {'python': sys.version.replace('\n', ' '), 'platform': platform.platform()}
    for p in pkgs:
        try:
            out[p] = im.version(p)
        except Exception:
            pass
    return out

def set_global_seed(seed: int, deterministic_torch: bool=True) -> Dict[str, Any]:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    info: Dict[str, Any] = {'seed': int(seed), 'deterministic_torch': bool(deterministic_torch)}
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
                info['torch_use_deterministic_algorithms'] = True
            except Exception:
                info['torch_use_deterministic_algorithms'] = False
        info['torch_cuda_available'] = bool(torch.cuda.is_available())
    except Exception as e:
        info['torch_import'] = f'fail:{type(e).__name__}'
    return info

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, float), np.asarray(y_pred, float))))

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {'R2': float(r2_score(y_true, y_pred)), 'MAE': float(mean_absolute_error(y_true, y_pred)), 'RMSE': rmse(y_true, y_pred)}

def multi_rmse(y_true_2d: np.ndarray, y_pred_2d: np.ndarray) -> float:
    y_true_2d = np.asarray(y_true_2d, dtype=float)
    y_pred_2d = np.asarray(y_pred_2d, dtype=float)
    return float(np.sqrt(np.mean((y_true_2d - y_pred_2d) ** 2)))

def ensemble_aggregate(member_preds: np.ndarray) -> Dict[str, np.ndarray]:
    member_preds = np.asarray(member_preds, dtype=float)
    return {'mean': member_preds.mean(axis=0), 'median': np.median(member_preds, axis=0), 'std': member_preds.std(axis=0, ddof=0), 'iqr': np.quantile(member_preds, 0.75, axis=0) - np.quantile(member_preds, 0.25, axis=0)}

def eval_multitask_ensemble(y_true: np.ndarray, member_preds: np.ndarray, targets: List[str]) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    agg = ensemble_aggregate(member_preds)
    out: Dict[str, Any] = {'per_target': {}, 'overall_macro': {}, 'uncertainty': {}}
    for agg_name in ['mean', 'median']:
        per = []
        for j, tgt in enumerate(targets):
            md = metrics_dict(y_true[:, j], agg[agg_name][:, j])
            out['per_target'].setdefault(tgt, {})[agg_name] = md
            per.append(md)
        out['overall_macro'][agg_name] = {'R2': float(np.nanmean([x['R2'] for x in per])), 'MAE': float(np.nanmean([x['MAE'] for x in per])), 'RMSE': float(np.nanmean([x['RMSE'] for x in per]))}
    for j, tgt in enumerate(targets):
        out['uncertainty'][tgt] = {'std_mean': float(np.mean(agg['std'][:, j])), 'iqr_mean': float(np.mean(agg['iqr'][:, j]))}
    out['uncertainty']['__overall_macro__'] = {'std_mean': float(np.mean(agg['std'])), 'iqr_mean': float(np.mean(agg['iqr']))}
    return out

def load_splits(train_csv: str, val_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if 'SMILES' not in d.columns:
            raise ValueError(f"{name} CSV must contain a 'SMILES' column.")
    full_df = pd.concat([train_df.assign(_split='train'), val_df.assign(_split='val'), test_df.assign(_split='test')], axis=0, ignore_index=True)
    full_df.insert(0, 'global_index', np.arange(len(full_df), dtype=int))
    return (train_df, val_df, test_df, full_df)

def select_targets(df: pd.DataFrame, requested: Optional[List[str]]=None) -> List[str]:
    if requested is None:
        targets = [c for c in df.columns if c != 'SMILES' and pd.api.types.is_numeric_dtype(df[c])]
    else:
        targets = list(requested)
        missing = [t for t in targets if t not in df.columns]
        if missing:
            raise ValueError(f'Missing target columns: {missing}')
        nonnum = [t for t in targets if not pd.api.types.is_numeric_dtype(df[t])]
        if nonnum:
            raise ValueError(f'Target columns must be numeric: {nonnum}')
    if not targets:
        raise ValueError('No target columns found.')
    return targets

def apply_target_transform(Y: np.ndarray, mode: str, inverse: bool=False) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if mode == 'none':
        return Y
    if mode == 'log10':
        if inverse:
            return np.power(10.0, Y)
        return np.log10(np.clip(Y, 1e-300, None))
    raise ValueError(f'Unknown TARGET_TRANSFORM: {mode}')

@dataclass
class TargetScaler:
    enabled: bool = True
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, Y: np.ndarray) -> 'TargetScaler':
        Y = np.asarray(Y, dtype=float)
        if not self.enabled:
            self.mean_ = np.zeros(Y.shape[1], dtype=float)
            self.std_ = np.ones(Y.shape[1], dtype=float)
            return self
        self.mean_ = np.mean(Y, axis=0)
        self.std_ = np.std(Y, axis=0, ddof=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('TargetScaler must be fitted before transform().')
        return (np.asarray(Y, dtype=float) - self.mean_) / self.std_

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('TargetScaler must be fitted before inverse_transform().')
        return np.asarray(Y, dtype=float) * self.std_ + self.mean_

    def stats(self, targets: List[str]) -> Dict[str, Any]:
        if self.mean_ is None or self.std_ is None:
            return {'enabled': self.enabled, 'fitted': False}
        return {'enabled': bool(self.enabled), 'per_target': {t: {'mean': float(self.mean_[i]), 'std': float(self.std_[i])} for i, t in enumerate(targets)}}

def get_Y(df: pd.DataFrame, targets: List[str], target_transform: str) -> np.ndarray:
    Y_raw = df[targets].astype(float).to_numpy()
    return apply_target_transform(Y_raw, target_transform, inverse=False)

def leakage_checks(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def canon(s: str) -> Optional[str]:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    def scaffold(s: str) -> Optional[str]:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        try:
            sc = MurckoScaffold.GetScaffoldForMol(m)
            return Chem.MolToSmiles(sc, canonical=True) if sc is not None else ''
        except Exception:
            return None
    rep: Dict[str, Any] = {'invalid_smiles': {}, 'duplicates_within_split': {}, 'overlap': {}, 'scaffold_overlap': {}}
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    canon_map: Dict[str, List[Optional[str]]] = {}
    scaf_map: Dict[str, List[Optional[str]]] = {}
    for k, df in splits.items():
        c = [canon(s) for s in df['SMILES'].astype(str).tolist()]
        s = [scaffold(x) for x in df['SMILES'].astype(str).tolist()]
        canon_map[k] = c
        scaf_map[k] = s
        rep['invalid_smiles'][k] = int(sum((x is None for x in c)))
        c_valid = [x for x in c if x is not None]
        rep['duplicates_within_split'][k] = int(pd.Series(c_valid).duplicated().sum())

    def set_valid(lst: List[Optional[str]]) -> set:
        return set([x for x in lst if x is not None])
    tr, va, te = (set_valid(canon_map['train']), set_valid(canon_map['val']), set_valid(canon_map['test']))
    rep['overlap']['train∩test_canonical_smiles'] = sorted(list(tr & te))
    rep['overlap']['val∩test_canonical_smiles'] = sorted(list(va & te))
    rep['overlap']['train∩val_canonical_smiles'] = sorted(list(tr & va))
    tr_s, va_s, te_s = (set_valid(scaf_map['train']), set_valid(scaf_map['val']), set_valid(scaf_map['test']))
    rep['scaffold_overlap']['train∩test_bemis_murcko'] = sorted(list(tr_s & te_s))
    rep['scaffold_overlap']['val∩test_bemis_murcko'] = sorted(list(va_s & te_s))
    rep['scaffold_overlap']['train∩val_bemis_murcko'] = sorted(list(tr_s & va_s))
    return rep

@dataclass
class FPConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True

def smiles_to_morgan_fp(smiles: str, cfg: FPConfig) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.full((cfg.n_bits,), np.nan, dtype=float)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=cfg.radius, nBits=cfg.n_bits, useChirality=cfg.use_chirality)
    arr = np.zeros((cfg.n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(float)

def featurize_fp(df: pd.DataFrame, cfg: FPConfig) -> Tuple[np.ndarray, np.ndarray]:
    X_all = np.vstack([smiles_to_morgan_fp(s, cfg) for s in df['SMILES'].astype(str).tolist()])
    valid_mask = ~np.isnan(X_all).any(axis=1)
    return (X_all[valid_mask], valid_mask)

# Model
def _fit_xgb_multioutput(X_fit: np.ndarray, Y_fit: np.ndarray, X_val: Optional[np.ndarray], Y_val: Optional[np.ndarray], params: Dict[str, Any], seed: int) -> Tuple[Any, str]:
    from xgboost import XGBRegressor
    base_params = dict(params)
    mode_used = 'native_multi_output_tree'
    try:
        native_params = dict(base_params)
        native_params.setdefault('multi_strategy', 'multi_output_tree')
        m = XGBRegressor(random_state=seed, **native_params)
        try:
            if X_val is not None and Y_val is not None:
                m.fit(X_fit, Y_fit, eval_set=[(X_val, Y_val)], verbose=False)
            else:
                m.fit(X_fit, Y_fit, verbose=False)
        except TypeError:
            m.fit(X_fit, Y_fit)
        pred_test = m.predict(X_fit[:min(len(X_fit), 3)])
        pred_test = np.asarray(pred_test)
        if pred_test.ndim != 2 or pred_test.shape[1] != Y_fit.shape[1]:
            raise RuntimeError('Native XGBoost did not return a 2D multitask prediction.')
        return (m, mode_used)
    except Exception:
        from sklearn.multioutput import MultiOutputRegressor
        fallback_params = {k: v for k, v in base_params.items() if k != 'multi_strategy'}
        base = XGBRegressor(random_state=seed, **fallback_params)
        m = MultiOutputRegressor(base, n_jobs=-1)
        m.fit(X_fit, Y_fit)
        return (m, 'sklearn_MultiOutputRegressor_fallback')

def tune_xgb_multitask_val(X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray, Y_va: np.ndarray, seed: int, param_grid: Dict[str, List[Any]], max_trials: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    rng = np.random.RandomState(seed)
    rng.shuffle(combos)
    combos = combos[:min(max_trials, len(combos))]
    best = {'val_rmse': float('inf'), 'params': None, 'mode': None}
    trace = []
    for vals in combos:
        params = dict(zip(keys, vals))
        m, mode = _fit_xgb_multioutput(X_tr, Y_tr, X_va, Y_va, params=params, seed=seed)
        pred = np.asarray(m.predict(X_va))
        s = multi_rmse(Y_va, pred)
        trace.append({'params': params, 'mode_used': mode, 'val_multi_rmse_scaled': float(s)})
        if s < best['val_rmse']:
            best = {'val_rmse': float(s), 'params': params, 'mode': mode}
    if best['params'] is None:
        raise RuntimeError('XGB multitask tuning failed.')
    meta = {'selection': 'val_multi_RMSE_scaled', 'max_trials': int(max_trials), 'trace': trace, 'best_val_multi_rmse_scaled': best['val_rmse'], 'best_mode_used': best['mode']}
    return (best['params'], meta)

def fit_xgb_multitask_member(X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray, Y_va: np.ndarray, X_te: np.ndarray, params: Dict[str, Any], seed: int, bagging: bool) -> Tuple[np.ndarray, np.ndarray, Any, str]:
    rng = np.random.RandomState(seed)
    if bagging:
        idx = rng.randint(0, len(Y_tr), size=len(Y_tr))
        X_fit, Y_fit = (X_tr[idx], Y_tr[idx])
    else:
        X_fit, Y_fit = (X_tr, Y_tr)
    m, mode = _fit_xgb_multioutput(X_fit, Y_fit, X_va, Y_va, params=params, seed=seed)
    return (np.asarray(m.predict(X_va)), np.asarray(m.predict(X_te)), m, mode)

def run_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame, out_root: Path, seed: int, n_members: int, target_transform: str, requested_targets: Optional[List[str]], use_target_scaling: bool, bagging: bool, save_models: bool) -> None:
    set_global_seed(seed)
    targets = select_targets(train_df, requested_targets)
    n_targets = len(targets)
    leakage = leakage_checks(train_df, val_df, test_df)
    split_indices = {'train': full_df.loc[full_df['_split'] == 'train', 'global_index'].tolist(), 'val': full_df.loc[full_df['_split'] == 'val', 'global_index'].tolist(), 'test': full_df.loc[full_df['_split'] == 'test', 'global_index'].tolist()}
    outdir = ensure_dir(out_root / 'xgb_fp_multitask')
    models_dir = ensure_dir(outdir / 'models')
    Y_train_base = get_Y(train_df, targets, target_transform)
    Y_val_base = get_Y(val_df, targets, target_transform)
    Y_test_base = get_Y(test_df, targets, target_transform)
    target_scaler = TargetScaler(enabled=bool(use_target_scaling)).fit(Y_train_base)
    Y_train = target_scaler.transform(Y_train_base)
    Y_val = target_scaler.transform(Y_val_base)
    Y_test = target_scaler.transform(Y_test_base)
    run_config = {'model': 'xgb_fp_multitask', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seed': int(seed), 'n_members': int(n_members), 'targets': targets, 'n_targets': int(n_targets), 'target_transform': str(target_transform), 'target_scaling': target_scaler.stats(targets), 'bagging': bool(bagging), 'save_models': bool(save_models), 'data_shape': {'train': list(train_df.shape), 'val': list(val_df.shape), 'test': list(test_df.shape)}, 'data_hash': {'train_sha256': sha256_df_content(train_df), 'val_sha256': sha256_df_content(val_df), 'test_sha256': sha256_df_content(test_df)}, 'split_indices': split_indices, 'versions': collect_versions(), 'seed_control': set_global_seed(seed)}
    metrics_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {'model': 'xgb_fp_multitask', 'n_members': int(n_members), 'targets': targets, 'notes': {'multitask': True, 'target_scaling': bool(use_target_scaling), 'selection_metric': 'validation multi-target RMSE on scaled targets'}}
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        pass
    XGB_GRID = {'n_estimators': [800, 2000], 'max_depth': [3, 5, 7], 'learning_rate': [0.03, 0.05, 0.1], 'subsample': [0.7, 0.9, 1.0], 'colsample_bytree': [0.7, 0.9, 1.0], 'reg_lambda': [1.0, 5.0, 10.0], 'min_child_weight': [1.0, 5.0], 'gamma': [0.0, 0.1], 'tree_method': ['hist'], 'objective': ['reg:squarederror'], 'eval_metric': ['rmse']}
    TUNE_TRIALS = 30
    member_val_preds_scaled = []
    member_test_preds_scaled = []
    member_meta: List[Dict[str, Any]] = []
    te_df_for_pred = test_df.copy()
    va_df_for_eval = val_df.copy()
    Y_val_base_for_eval = Y_val_base
    Y_test_base_for_eval = Y_test_base
    fp_cfg = FPConfig()
    X_train_all, m_tr = featurize_fp(train_df, fp_cfg)
    X_val_all, m_va = featurize_fp(val_df, fp_cfg)
    X_test_all, m_te = featurize_fp(test_df, fp_cfg)
    tr_df = train_df.loc[m_tr].reset_index(drop=True)
    va_df = val_df.loc[m_va].reset_index(drop=True)
    te_df = test_df.loc[m_te].reset_index(drop=True)
    X_train, X_val, X_test = (X_train_all, X_val_all, X_test_all)
    Y_train_use = Y_train[m_tr]
    Y_val_use = Y_val[m_va]
    Y_test_use = Y_test[m_te]
    Y_val_base_for_eval = Y_val_base[m_va]
    Y_test_base_for_eval = Y_test_base[m_te]
    te_df_for_pred = te_df
    va_df_for_eval = va_df
    run_config['fingerprint'] = asdict(fp_cfg)
    run_config['valid_smiles_counts'] = {'train': int(m_tr.sum()), 'val': int(m_va.sum()), 'test': int(m_te.sum()), 'train_total': int(len(m_tr)), 'val_total': int(len(m_va)), 'test_total': int(len(m_te))}
    best_params, tune_meta = tune_xgb_multitask_val(X_train, Y_train_use, X_val, Y_val_use, seed=seed, param_grid=XGB_GRID, max_trials=TUNE_TRIALS)
    mode_used = []
    for i in range(n_members):
        s = seed * 1000 + 110 + i
        pv, pt, model, mode = fit_xgb_multitask_member(X_train, Y_train_use, X_val, Y_val_use, X_test, best_params, seed=s, bagging=bagging)
        member_val_preds_scaled.append(pv)
        member_test_preds_scaled.append(pt)
        mode_used.append(mode)
        member_meta.append({'member': i, 'seed': s, 'xgb_mode_used': mode})
        if save_models:
            import joblib
            ensure_dir(models_dir)
            joblib.dump(model, models_dir / f'member_{i:02d}.joblib')
    tune_meta['member_modes_used'] = mode_used
    summary['best_params'] = best_params
    summary['tuning'] = tune_meta
    member_val_preds_scaled = np.stack(member_val_preds_scaled, axis=0)
    member_test_preds_scaled = np.stack(member_test_preds_scaled, axis=0)
    member_val_preds_base = np.stack([target_scaler.inverse_transform(p) for p in member_val_preds_scaled], axis=0)
    member_test_preds_base = np.stack([target_scaler.inverse_transform(p) for p in member_test_preds_scaled], axis=0)
    Y_val_report = apply_target_transform(Y_val_base_for_eval, target_transform, inverse=True)
    Y_test_report = apply_target_transform(Y_test_base_for_eval, target_transform, inverse=True)
    member_val_preds_report = apply_target_transform(member_val_preds_base, target_transform, inverse=True)
    member_test_preds_report = apply_target_transform(member_test_preds_base, target_transform, inverse=True)
    val_eval = eval_multitask_ensemble(Y_val_report, member_val_preds_report, targets)
    test_eval = eval_multitask_ensemble(Y_test_report, member_test_preds_report, targets)
    pred_test = pd.DataFrame({'SMILES': te_df_for_pred['SMILES'].astype(str).tolist()})
    agg_test = ensemble_aggregate(member_test_preds_report)
    for j, tgt in enumerate(targets):
        pred_test[f'y_true_{tgt}'] = Y_test_report[:, j]
        pred_test[f'y_mean_{tgt}'] = agg_test['mean'][:, j]
        pred_test[f'y_median_{tgt}'] = agg_test['median'][:, j]
        pred_test[f'y_std_{tgt}'] = agg_test['std'][:, j]
        pred_test[f'y_iqr_{tgt}'] = agg_test['iqr'][:, j]
        for i in range(n_members):
            pred_test[f'y_m{i:02d}_{tgt}'] = member_test_preds_report[i, :, j]
    for split_name, ev, n_samples in [('val', val_eval, len(Y_val_report)), ('test', test_eval, len(Y_test_report))]:
        for tgt in targets:
            unc = ev['uncertainty'][tgt]
            for agg_name in ['mean', 'median']:
                m = ev['per_target'][tgt][agg_name]
                metrics_rows.append({'model': 'xgb_fp_multitask', 'target': tgt, 'split': split_name, 'agg': agg_name, 'R2': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'unc_std_mean': unc['std_mean'], 'unc_iqr_mean': unc['iqr_mean'], 'n_samples': int(n_samples), 'n_members': int(n_members)})
        unc = ev['uncertainty']['__overall_macro__']
        for agg_name in ['mean', 'median']:
            m = ev['overall_macro'][agg_name]
            metrics_rows.append({'model': 'xgb_fp_multitask', 'target': '__overall_macro__', 'split': split_name, 'agg': agg_name, 'R2': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'unc_std_mean': unc['std_mean'], 'unc_iqr_mean': unc['iqr_mean'], 'n_samples': int(n_samples), 'n_members': int(n_members)})
    summary['members'] = member_meta
    summary['val'] = val_eval
    summary['test'] = test_eval
    write_json(outdir / 'run_config.json', run_config)
    write_json(outdir / 'leakage_report.json', leakage)
    pd.DataFrame(metrics_rows).to_csv(outdir / 'metrics.csv', index=False)
    pred_test.to_csv(outdir / 'predictions.csv', index=False)
    write_json(outdir / 'summary.json', summary)
    print(f'[DONE] Multitask outputs saved under: {outdir}')
# Run
train_df, val_df, test_df, full_df = load_splits(TRAIN_CSV, VAL_CSV, TEST_CSV)
run_model(MODEL_NAME, train_df, val_df, test_df, full_df, out_root=Path(OUT_ROOT), seed=int(SEED), n_members=int(N_MEMBERS), target_transform=str(TARGET_TRANSFORM), requested_targets=TARGETS, use_target_scaling=bool(USE_TARGET_SCALING), bagging=bool(BAGGING), save_models=bool(SAVE_MODELS))
