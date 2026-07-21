# Configuration
TRAIN_CSV = 'train_9of10.csv'
VAL_CSV = 'val_1of10.csv'
TEST_CSV = 'fixed_test_10pct.csv'
OUT_ROOT = './benchmark_outputs_fixedsplit_multitask'
MODEL_NAME = 'gin_multitask'
SEED = 0
N_MEMBERS = 10
TARGETS = ['He', 'H2', 'O2', 'N2', 'CO2', 'CH4']
TARGET_TRANSFORM = 'none'
USE_TARGET_SCALING = True
BAGGING = False
SAVE_MODELS = True
PYG_HIDDEN_LAYER_1 = [256, 512]
PYG_HIDDEN_LAYER_2 = [128, 256]
PYG_HIDDEN_LAYER_3 = [64, 128]
PYG_DROPOUTS = [0.1, 0.15, 0.2, 0.25, 0.3]
PYG_LEARNING_RATES = [1e-05, 3e-05, 0.0001, 0.0003, 0.001]
PYG_BATCH_SIZES = [128, 64]
PYG_TUNE_TRIALS = 30
PYG_MAX_EPOCHS = 300
PYG_PATIENCE = 30
# Dependencies
import os, sys, json, time, random, hashlib, itertools, platform
from dataclasses import dataclass
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

# Model
def require_pyg():
    try:
        import torch_geometric
    except Exception as e:
        raise RuntimeError('torch_geometric is required for GCN/GAT/GIN.') from e

def mol_to_graph_data(smiles: str, max_atomic_num: int=100):
    require_pyg()
    from rdkit import Chem
    from rdkit.Chem import rdchem
    import torch
    from torch_geometric.data import Data
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    feats = []
    for atom in mol.GetAtoms():
        z = int(atom.GetAtomicNum())
        onehot = [0.0] * max_atomic_num
        if 1 <= z <= max_atomic_num:
            onehot[z - 1] = 1.0
        else:
            onehot[-1] = 1.0
        hyb = atom.GetHybridization()
        hyb_list = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2]
        onehot_hyb = [1.0 if hyb == h else 0.0 for h in hyb_list]
        basic = [float(atom.GetDegree()), float(atom.GetFormalCharge()), float(atom.GetTotalNumHs(includeNeighbors=True)), float(atom.GetImplicitValence()), float(atom.GetIsAromatic()), float(atom.GetMass() / 100.0)]
        feats.append(onehot + onehot_hyb + basic)
    x = torch.tensor(feats, dtype=torch.float)
    edge_list = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        edge_list.append([i, j])
        edge_list.append([j, i])
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def build_pyg_dataset(df: pd.DataFrame, Y_scaled: np.ndarray):
    require_pyg()
    import torch
    graphs = []
    valid_idx = []
    smiles = df['SMILES'].astype(str).tolist()
    for i, smi in enumerate(smiles):
        g = mol_to_graph_data(smi)
        if g is None:
            continue
        g.y = torch.tensor(Y_scaled[i], dtype=torch.float)
        graphs.append(g)
        valid_idx.append(i)
    return (graphs, np.asarray(valid_idx, dtype=int))

def pyg_models():
    require_pyg()
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GINConv, global_mean_pool

    class GINNet(nn.Module):

        def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], dropout: float):
            super().__init__()
            self.dropout = float(dropout)
            self.convs = nn.ModuleList()
            d = in_dim
            for h in hidden_dims:
                mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h))
                self.convs.append(GINConv(mlp))
                d = h
            self.lin = nn.Linear(hidden_dims[-1], out_dim)

        def forward(self, data):
            x, edge_index, batch = (data.x, data.edge_index, data.batch)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = torch.relu(x)
                x = torch.dropout(x, p=self.dropout, train=self.training)
            x = global_mean_pool(x, batch)
            return self.lin(x)
    return GINNet

def train_pyg_multitask_member(arch: str, train_graphs, val_graphs, test_graphs, out_dim: int, seed: int, params: Dict[str, Any], device: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    require_pyg()
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
    set_global_seed(seed)
    in_dim = int(train_graphs[0].x.shape[1])
    ModelClass = pyg_models()
    net = ModelClass(in_dim=in_dim, out_dim=out_dim, hidden_dims=params['hidden_dims'], dropout=params['dropout'])
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(params['lr']))
    loss_fn = nn.MSELoss()
    bs = int(params['batch_size'])
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=bs, shuffle=False)
    best = {'epoch': 0, 'val_rmse': float('inf'), 'state_dict': None}
    bad = 0
    history = {'val_multi_rmse_scaled': []}

    def eval_loader(loader) -> Tuple[np.ndarray, np.ndarray]:
        net.eval()
        ys, ps = ([], [])
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = net(batch).detach().cpu().numpy()
                y = batch.y.detach().cpu().numpy()
                if y.ndim == 1:
                    y = y.reshape(-1, out_dim)
                ys.append(y)
                ps.append(pred)
        y = np.vstack(ys) if ys else np.empty((0, out_dim))
        p = np.vstack(ps) if ps else np.empty((0, out_dim))
        return (y, p)
    for epoch in range(1, int(params['max_epochs']) + 1):
        net.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = net(batch)
            y = batch.y
            if y.ndim == 1:
                y = y.view(-1, out_dim)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        yv, pv = eval_loader(val_loader)
        v = multi_rmse(yv, pv)
        history['val_multi_rmse_scaled'].append(float(v))
        if v < best['val_rmse'] - 1e-12:
            best = {'epoch': epoch, 'val_rmse': float(v), 'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= int(params['patience']):
                break
    net.load_state_dict(best['state_dict'])
    _, pv = eval_loader(val_loader)
    _, pt = eval_loader(test_loader)
    meta = {'best_epoch': int(best['epoch']), 'best_val_multi_rmse_scaled': float(best['val_rmse']), 'history': history}
    return (pv, pt, {'state_dict': best['state_dict'], 'meta': meta})

def build_pyg_param_grid(arch: str) -> Dict[str, List[Any]]:
    hidden_dims = [[int(h1), int(h2), int(h3)] for h1, h2, h3 in itertools.product(PYG_HIDDEN_LAYER_1, PYG_HIDDEN_LAYER_2, PYG_HIDDEN_LAYER_3)]
    grid: Dict[str, List[Any]] = {'hidden_dims': hidden_dims, 'dropout': [float(x) for x in PYG_DROPOUTS], 'lr': [float(x) for x in PYG_LEARNING_RATES], 'batch_size': [int(x) for x in PYG_BATCH_SIZES], 'max_epochs': [int(PYG_MAX_EPOCHS)], 'patience': [int(PYG_PATIENCE)]}
    return grid

def tune_pyg_multitask_val(arch: str, train_graphs, val_graphs, out_dim: int, seed: int, param_grid: Dict[str, List[Any]], max_trials: int, device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    rng = np.random.RandomState(seed)
    rng.shuffle(combos)
    combos = combos[:min(int(max_trials), len(combos))]
    best = {'val_rmse': float('inf'), 'params': None, 'best_epoch': None}
    trace = []
    for trial_idx, vals in enumerate(combos, start=1):
        params = dict(zip(keys, vals))
        pv, _, pack = train_pyg_multitask_member(arch='gin', train_graphs=train_graphs, val_graphs=val_graphs, test_graphs=val_graphs, out_dim=out_dim, seed=seed + trial_idx - 1, params=params, device=device)
        s = float(pack['meta']['best_val_multi_rmse_scaled'])
        trace.append({'trial': int(trial_idx), 'params': params, 'best_epoch': int(pack['meta']['best_epoch']), 'best_val_multi_rmse_scaled': s})
        if s < best['val_rmse']:
            best = {'val_rmse': s, 'params': params, 'best_epoch': int(pack['meta']['best_epoch'])}
        del pv, pack
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    if best['params'] is None:
        raise RuntimeError(f"{'GIN'} multitask tuning failed.")
    best_params = dict(best['params'])
    best_params['best_epoch'] = int(best['best_epoch'])
    meta = {'selection': 'val_multi_RMSE_scaled', 'search_method': 'randomly_ordered_grid', 'max_trials': int(max_trials), 'num_total_combinations': int(np.prod([len(param_grid[k]) for k in keys])), 'num_evaluated_combinations': int(len(combos)), 'search_space': {'hidden_layer_1': [int(x) for x in PYG_HIDDEN_LAYER_1], 'hidden_layer_2': [int(x) for x in PYG_HIDDEN_LAYER_2], 'hidden_layer_3': [int(x) for x in PYG_HIDDEN_LAYER_3], 'dropout': [float(x) for x in PYG_DROPOUTS], 'lr': [float(x) for x in PYG_LEARNING_RATES], 'batch_size': [int(x) for x in PYG_BATCH_SIZES], 'max_epochs': int(PYG_MAX_EPOCHS), 'patience': int(PYG_PATIENCE)}, 'trace': trace, 'best_val_multi_rmse_scaled': float(best['val_rmse'])}
    return (best_params, meta)

def run_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame, out_root: Path, seed: int, n_members: int, target_transform: str, requested_targets: Optional[List[str]], use_target_scaling: bool, bagging: bool, save_models: bool) -> None:
    set_global_seed(seed)
    targets = select_targets(train_df, requested_targets)
    n_targets = len(targets)
    leakage = leakage_checks(train_df, val_df, test_df)
    split_indices = {'train': full_df.loc[full_df['_split'] == 'train', 'global_index'].tolist(), 'val': full_df.loc[full_df['_split'] == 'val', 'global_index'].tolist(), 'test': full_df.loc[full_df['_split'] == 'test', 'global_index'].tolist()}
    outdir = ensure_dir(out_root / 'gin_multitask')
    models_dir = ensure_dir(outdir / 'models')
    Y_train_base = get_Y(train_df, targets, target_transform)
    Y_val_base = get_Y(val_df, targets, target_transform)
    Y_test_base = get_Y(test_df, targets, target_transform)
    target_scaler = TargetScaler(enabled=bool(use_target_scaling)).fit(Y_train_base)
    Y_train = target_scaler.transform(Y_train_base)
    Y_val = target_scaler.transform(Y_val_base)
    Y_test = target_scaler.transform(Y_test_base)
    run_config = {'model': 'gin_multitask', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seed': int(seed), 'n_members': int(n_members), 'targets': targets, 'n_targets': int(n_targets), 'target_transform': str(target_transform), 'target_scaling': target_scaler.stats(targets), 'bagging': bool(bagging), 'save_models': bool(save_models), 'data_shape': {'train': list(train_df.shape), 'val': list(val_df.shape), 'test': list(test_df.shape)}, 'data_hash': {'train_sha256': sha256_df_content(train_df), 'val_sha256': sha256_df_content(val_df), 'test_sha256': sha256_df_content(test_df)}, 'split_indices': split_indices, 'versions': collect_versions(), 'seed_control': set_global_seed(seed)}
    metrics_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {'model': 'gin_multitask', 'n_members': int(n_members), 'targets': targets, 'notes': {'multitask': True, 'target_scaling': bool(use_target_scaling), 'selection_metric': 'validation multi-target RMSE on scaled targets'}}
    device = 'cpu'
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        pass
    TUNE_TRIALS = int(PYG_TUNE_TRIALS)
    member_val_preds_scaled = []
    member_test_preds_scaled = []
    member_meta: List[Dict[str, Any]] = []
    te_df_for_pred = test_df.copy()
    va_df_for_eval = val_df.copy()
    Y_val_base_for_eval = Y_val_base
    Y_test_base_for_eval = Y_test_base
    arch = 'gin'
    train_graphs, idx_tr = build_pyg_dataset(train_df, Y_train)
    val_graphs, idx_va = build_pyg_dataset(val_df, Y_val)
    test_graphs, idx_te = build_pyg_dataset(test_df, Y_test)
    if len(train_graphs) == 0 or len(val_graphs) == 0 or len(test_graphs) == 0:
        raise RuntimeError(f"[{'gin_multitask'}] Graph generation failed for at least one split.")
    Y_val_base_for_eval = Y_val_base[idx_va]
    Y_test_base_for_eval = Y_test_base[idx_te]
    te_df_for_pred = test_df.iloc[idx_te].reset_index(drop=True)
    va_df_for_eval = val_df.iloc[idx_va].reset_index(drop=True)
    run_config['valid_graph_counts'] = {'train': int(len(train_graphs)), 'val': int(len(val_graphs)), 'test': int(len(test_graphs)), 'train_total': int(len(train_df)), 'val_total': int(len(val_df)), 'test_total': int(len(test_df))}
    pyg_param_grid = build_pyg_param_grid(arch)
    best_params, tune_meta = tune_pyg_multitask_val(arch=arch, train_graphs=train_graphs, val_graphs=val_graphs, out_dim=n_targets, seed=seed, param_grid=pyg_param_grid, max_trials=TUNE_TRIALS, device=device)
    for i in range(n_members):
        s = seed * 1000 + 310 + i
        pv, pt, pack = train_pyg_multitask_member(arch, train_graphs, val_graphs, test_graphs, out_dim=n_targets, seed=s, params=best_params, device=device)
        member_val_preds_scaled.append(pv)
        member_test_preds_scaled.append(pt)
        member_meta.append({'member': i, 'seed': s, **pack['meta']})
        if save_models:
            import torch
            ensure_dir(models_dir)
            torch.save(pack['state_dict'], models_dir / f'member_{i:02d}.pt')
            write_json(models_dir / f'member_{i:02d}_meta.json', pack['meta'])
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
                metrics_rows.append({'model': 'gin_multitask', 'target': tgt, 'split': split_name, 'agg': agg_name, 'R2': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'unc_std_mean': unc['std_mean'], 'unc_iqr_mean': unc['iqr_mean'], 'n_samples': int(n_samples), 'n_members': int(n_members)})
        unc = ev['uncertainty']['__overall_macro__']
        for agg_name in ['mean', 'median']:
            m = ev['overall_macro'][agg_name]
            metrics_rows.append({'model': 'gin_multitask', 'target': '__overall_macro__', 'split': split_name, 'agg': agg_name, 'R2': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'unc_std_mean': unc['std_mean'], 'unc_iqr_mean': unc['iqr_mean'], 'n_samples': int(n_samples), 'n_members': int(n_members)})
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
