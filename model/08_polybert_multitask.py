# Configuration
TRAIN_CSV = 'train_9of10.csv'
VAL_CSV = 'val_1of10.csv'
TEST_CSV = 'fixed_test_10pct.csv'
OUT_ROOT = './benchmark_outputs_fixedsplit_multitask'
MODEL_NAME = 'polybert_multitask'
SEED = 0
N_MEMBERS = 10
TARGETS = ['He', 'H2', 'O2', 'N2', 'CO2', 'CH4']
TARGET_TRANSFORM = 'none'
USE_TARGET_SCALING = True
BAGGING = True
SAVE_MODELS = True
# Dependencies
import os, sys, json, time, random, hashlib, platform
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
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
def resolve_polybert_local_snapshot(base_model_id: str) -> str:
    from pathlib import Path
    import os
    p = Path(str(base_model_id))
    if p.exists() and p.is_dir():
        return str(p.resolve())
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    allow_patterns = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'spm.model', 'special_tokens_map.json', 'added_tokens.json', 'sentence_bert_config.json', 'config_sentence_transformers.json', 'modules.json', '1_Pooling/config.json']
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=str(base_model_id), repo_type='model', allow_patterns=allow_patterns, ignore_patterns=['*.safetensors', 'model.safetensors'], token=False, local_files_only=False)
    except TypeError:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=str(base_model_id), allow_patterns=allow_patterns, ignore_patterns=['*.safetensors', 'model.safetensors'], local_files_only=False)
    except Exception as e:
        raise OSError(f'Could not download a local snapshot for {base_model_id}. The public repo should contain `pytorch_model.bin`. Try the following in Anaconda Prompt: `huggingface-cli logout`, then `set HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, `set HF_HUB_DISABLE_XET=1`, and rerun the notebook. Original error: {e}') from e
    local_dir = str(local_dir)
    if not (Path(local_dir) / 'pytorch_model.bin').exists():
        raise FileNotFoundError(f'Downloaded snapshot does not contain pytorch_model.bin: {local_dir}. Check whether the Hugging Face cache is corrupted; if so, delete the cache for kuelumbus/polyBERT and rerun.')
    if not (Path(local_dir) / 'config.json').exists():
        raise FileNotFoundError(f'Downloaded snapshot does not contain config.json: {local_dir}')
    return local_dir

def train_polybert_multitask_member(train_smiles: List[str], Y_tr: np.ndarray, val_smiles: List[str], Y_va: np.ndarray, test_smiles: List[str], Y_te: np.ndarray, seed: int, base_model_id: str, max_len: int, batch_size: int, lr: float, dropout: float, weight_decay: float, max_epochs: int, patience: int) -> Dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel
    from torch.optim import AdamW
    set_global_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class MultiTaskPermeabilityDataset(Dataset):

        def __init__(self, smiles: List[str], Y: np.ndarray, tokenizer, max_length: int):
            self.smiles = list(smiles)
            self.Y = np.asarray(Y, dtype=float)
            self.tok = tokenizer
            self.max_length = int(max_length)

        def __len__(self):
            return len(self.smiles)

        def __getitem__(self, idx):
            enc = self.tok(self.smiles[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item['label'] = torch.tensor(self.Y[idx], dtype=torch.float32)
            return item

    class MultiTaskPermeabilityPredictor(nn.Module):

        def __init__(self, base_model: AutoModel, n_targets: int, dropout: float):
            super().__init__()
            self.base = base_model
            hid = base_model.config.hidden_size
            self.regressor = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(hid, 256), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(256, int(n_targets)))

        def mean_pool(self, last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-09)
            return summed / denom

        def forward(self, input_ids, attention_mask):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            emb = self.mean_pool(out.last_hidden_state, attention_mask)
            return self.regressor(emb)

    @torch.no_grad()
    def predict(model: nn.Module, loader: DataLoader, loss_fn) -> Tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        preds, trues = ([], [])
        loss_sum, n = (0.0, 0)
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            y = batch['label'].to(device)
            yhat = model(input_ids, attn)
            loss = loss_fn(yhat, y)
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            preds.append(yhat.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
        y_true = np.vstack(trues) if trues else np.empty((0, Y_tr.shape[1]))
        y_pred = np.vstack(preds) if preds else np.empty((0, Y_tr.shape[1]))
        return (loss_sum / max(n, 1), y_true, y_pred)
    local_model_dir = resolve_polybert_local_snapshot(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(local_model_dir, local_files_only=True)
    base = AutoModel.from_config(cfg)
    bin_path = Path(local_model_dir) / 'pytorch_model.bin'
    try:
        state = torch.load(str(bin_path), map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(str(bin_path), map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing_keys, unexpected_keys = base.load_state_dict(state, strict=False)
    hf_load_meta = {'base_model_id': str(base_model_id), 'local_model_dir': str(local_model_dir), 'load_method': 'AutoConfig.from_pretrained + AutoModel.from_config + manual torch.load(pytorch_model.bin)', 'missing_keys_n': int(len(missing_keys)), 'unexpected_keys_n': int(len(unexpected_keys)), 'missing_keys_preview': list(missing_keys)[:20], 'unexpected_keys_preview': list(unexpected_keys)[:20]}
    model = MultiTaskPermeabilityPredictor(base, n_targets=Y_tr.shape[1], dropout=dropout).to(device)
    tr_set = MultiTaskPermeabilityDataset(train_smiles, Y_tr, tokenizer, max_len)
    va_set = MultiTaskPermeabilityDataset(val_smiles, Y_va, tokenizer, max_len)
    te_set = MultiTaskPermeabilityDataset(test_smiles, Y_te, tokenizer, max_len)
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False)
    opt = AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    no_imp = 0
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        running, n = (0.0, 0)
        for batch in tr_loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            y = batch['label'].to(device)
            opt.zero_grad()
            yhat = model(input_ids, attn)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.size(0)
            n += y.size(0)
        train_loss = running / max(n, 1)
        val_loss, _, _ = predict(model, va_loader, loss_fn)
        sched.step(val_loss)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        if val_loss < best_val - 1e-06:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = int(epoch)
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, yv_true, yv_pred = predict(model, va_loader, loss_fn)
    test_loss, yt_true, yt_pred = predict(model, te_loader, loss_fn)
    return {'best_epoch': int(best_epoch), 'best_val_loss_scaled': float(best_val), 'history': history, 'val': {'loss': float(val_loss), 'y_true': yv_true, 'y_pred': yv_pred}, 'test': {'loss': float(test_loss), 'y_true': yt_true, 'y_pred': yt_pred}, 'state_dict': model.state_dict(), 'hf_load_meta': hf_load_meta}

def run_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame, out_root: Path, seed: int, n_members: int, target_transform: str, requested_targets: Optional[List[str]], use_target_scaling: bool, bagging: bool, save_models: bool) -> None:
    set_global_seed(seed)
    targets = select_targets(train_df, requested_targets)
    n_targets = len(targets)
    leakage = leakage_checks(train_df, val_df, test_df)
    split_indices = {'train': full_df.loc[full_df['_split'] == 'train', 'global_index'].tolist(), 'val': full_df.loc[full_df['_split'] == 'val', 'global_index'].tolist(), 'test': full_df.loc[full_df['_split'] == 'test', 'global_index'].tolist()}
    outdir = ensure_dir(out_root / 'polybert_multitask')
    models_dir = ensure_dir(outdir / 'models')
    Y_train_base = get_Y(train_df, targets, target_transform)
    Y_val_base = get_Y(val_df, targets, target_transform)
    Y_test_base = get_Y(test_df, targets, target_transform)
    target_scaler = TargetScaler(enabled=bool(use_target_scaling)).fit(Y_train_base)
    Y_train = target_scaler.transform(Y_train_base)
    Y_val = target_scaler.transform(Y_val_base)
    Y_test = target_scaler.transform(Y_test_base)
    run_config = {'model': 'polybert_multitask', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seed': int(seed), 'n_members': int(n_members), 'targets': targets, 'n_targets': int(n_targets), 'target_transform': str(target_transform), 'target_scaling': target_scaler.stats(targets), 'bagging': bool(bagging), 'save_models': bool(save_models), 'data_shape': {'train': list(train_df.shape), 'val': list(val_df.shape), 'test': list(test_df.shape)}, 'data_hash': {'train_sha256': sha256_df_content(train_df), 'val_sha256': sha256_df_content(val_df), 'test_sha256': sha256_df_content(test_df)}, 'split_indices': split_indices, 'versions': collect_versions(), 'seed_control': set_global_seed(seed)}
    metrics_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {'model': 'polybert_multitask', 'n_members': int(n_members), 'targets': targets, 'notes': {'multitask': True, 'target_scaling': bool(use_target_scaling), 'selection_metric': 'validation multi-target RMSE on scaled targets'}}
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        pass
    POLYBERT_PARAMS = {'base_model_id': 'kuelumbus/polyBERT', 'max_len': 256, 'batch_size': 32, 'lr': 2e-05, 'dropout': 0.1, 'weight_decay': 0.01, 'max_epochs': 50, 'patience': 8}
    member_val_preds_scaled = []
    member_test_preds_scaled = []
    member_meta: List[Dict[str, Any]] = []
    te_df_for_pred = test_df.copy()
    va_df_for_eval = val_df.copy()
    Y_val_base_for_eval = Y_val_base
    Y_test_base_for_eval = Y_test_base
    best_params = dict(POLYBERT_PARAMS)
    tune_meta = {'selection': 'fixed_default', 'note': 'Multitask PolyBERT with validation-loss early stopping; no hyperparameter grid search.'}
    tr_smiles = train_df['SMILES'].astype(str).tolist()
    va_smiles = val_df['SMILES'].astype(str).tolist()
    te_smiles = test_df['SMILES'].astype(str).tolist()
    for i in range(n_members):
        s = seed * 1000 + 100 + i
        rng = np.random.RandomState(s)
        if bagging:
            idx = rng.randint(0, len(tr_smiles), size=len(tr_smiles))
            tr_s = [tr_smiles[j] for j in idx]
            Y_tr_use = Y_train[idx]
        else:
            tr_s = tr_smiles
            Y_tr_use = Y_train
        pack = train_polybert_multitask_member(tr_s, Y_tr_use, va_smiles, Y_val, te_smiles, Y_test, seed=s, base_model_id=best_params['base_model_id'], max_len=best_params['max_len'], batch_size=best_params['batch_size'], lr=best_params['lr'], dropout=best_params['dropout'], weight_decay=best_params['weight_decay'], max_epochs=best_params['max_epochs'], patience=best_params['patience'])
        member_val_preds_scaled.append(pack['val']['y_pred'])
        member_test_preds_scaled.append(pack['test']['y_pred'])
        member_meta.append({'member': i, 'seed': s, 'best_epoch': pack['best_epoch'], 'best_val_loss_scaled': pack['best_val_loss_scaled']})
        if save_models:
            import torch
            ensure_dir(models_dir)
            torch.save(pack['state_dict'], models_dir / f'member_{i:02d}.pt')
            write_json(models_dir / f'member_{i:02d}_meta.json', member_meta[-1])
        del pack
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
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
                metrics_rows.append({'model': 'polybert_multitask', 'target': tgt, 'split': split_name, 'agg': agg_name, 'R2': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'unc_std_mean': unc['std_mean'], 'unc_iqr_mean': unc['iqr_mean'], 'n_samples': int(n_samples), 'n_members': int(n_members)})
        unc = ev['uncertainty']['__overall_macro__']
        for agg_name in ['mean', 'median']:
            m = ev['overall_macro'][agg_name]
            metrics_rows.append({'model': 'polybert_multitask', 'target': '__overall_macro__', 'split': split_name, 'agg': agg_name, 'R2': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'unc_std_mean': unc['std_mean'], 'unc_iqr_mean': unc['iqr_mean'], 'n_samples': int(n_samples), 'n_members': int(n_members)})
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
