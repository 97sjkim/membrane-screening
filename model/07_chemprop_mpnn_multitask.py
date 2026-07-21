from __future__ import annotations
# Configuration
TRAIN_CSV = 'train_9of10.csv'
VAL_CSV = 'val_1of10.csv'
TEST_CSV = 'fixed_test_10pct.csv'
OUT_ROOT = './benchmark_outputs_fixedsplit'
MODEL_NAME = 'chemprop_mpnn_multitask_rmse_train'
SEED = 0
N_MEMBERS = 10
TARGET_TRANSFORM = 'none'
BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
MAX_EPOCHS = 400
PATIENCE = 40
NUM_WORKERS = 0
BATCH_NORM = False
# Dependencies
import os, json, time, hashlib, platform, random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from chemprop import data, models, nn
from chemprop.models.utils import save_model
from chemprop.nn.metrics import RMSE, MAE

# Utilities
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha256_df_content(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(('shape=' + str(df.shape)).encode('utf-8'))
    h.update(('cols=' + '|'.join(map(str, df.columns))).encode('utf-8'))
    h.update(df.to_csv(index=False).encode('utf-8'))
    return h.hexdigest()

def collect_versions() -> Dict[str, str]:
    import importlib.metadata as md
    pkgs = ['numpy', 'pandas', 'rdkit', 'torch', 'lightning', 'chemprop']
    out = {'python': platform.python_version()}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = 'unknown'
    return out

def set_global_seed(seed: int) -> Dict[str, Any]:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED', ''), 'cudnn_deterministic': bool(torch.backends.cudnn.deterministic), 'cudnn_benchmark': bool(torch.backends.cudnn.benchmark)}

def load_splits(train_csv: str, val_csv: str, test_csv: str, smiles_col: str='SMILES') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    for df in (train_df, val_df, test_df):
        assert smiles_col in df.columns, f'Missing column: {smiles_col}'
        df[smiles_col] = df[smiles_col].astype(str)
    train_tag = train_df.copy()
    train_tag['_split'] = 'train'
    val_tag = val_df.copy()
    val_tag['_split'] = 'val'
    test_tag = test_df.copy()
    test_tag['_split'] = 'test'
    full_df = pd.concat([train_tag, val_tag, test_tag], axis=0, ignore_index=True)
    full_df['global_index'] = np.arange(len(full_df), dtype=int)
    return (train_df, val_df, test_df, full_df)

def select_targets(train_df: pd.DataFrame, smiles_col: str='SMILES') -> List[str]:
    targets = []
    for c in train_df.columns:
        if c == smiles_col:
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]):
            targets.append(c)
    assert len(targets) > 0, 'No numeric target columns found.'
    return targets

def apply_target_transform(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == 'none':
        return y
    if mode == 'log10':
        eps = 1e-30
        return np.log10(np.maximum(y, eps))
    raise ValueError(f'Unknown TARGET_TRANSFORM: {mode}')

def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

def murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        return scaf if scaf else None
    except Exception:
        return None

def leakage_checks(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, smiles_col: str='SMILES') -> Dict[str, Any]:
    rep: Dict[str, Any] = {}

    def split_report(df: pd.DataFrame) -> Dict[str, Any]:
        can = [canonicalize_smiles(s) for s in df[smiles_col].astype(str).tolist()]
        invalid = sum((c is None for c in can))
        can_valid = [c for c in can if c is not None]
        dup = len(can_valid) - len(set(can_valid))
        scaf = [murcko_scaffold_smiles(s) for s in df[smiles_col].astype(str).tolist()]
        scaf_valid = [s for s in scaf if s is not None]
        return {'n_rows': int(len(df)), 'invalid_smiles': int(invalid), 'canonical_duplicates_within_split': int(dup), 'unique_canonical': int(len(set(can_valid))), 'unique_scaffolds': int(len(set(scaf_valid)))}
    rep['train'] = split_report(train_df)
    rep['val'] = split_report(val_df)
    rep['test'] = split_report(test_df)

    def canon_set(df):
        return set([c for c in (canonicalize_smiles(s) for s in df[smiles_col].astype(str).tolist()) if c is not None])

    def scaf_set(df):
        return set([s for s in (murcko_scaffold_smiles(x) for x in df[smiles_col].astype(str).tolist()) if s is not None])
    train_can, val_can, test_can = (canon_set(train_df), canon_set(val_df), canon_set(test_df))
    train_sc, val_sc, test_sc = (scaf_set(train_df), scaf_set(val_df), scaf_set(test_df))
    rep['overlap'] = {'train_val_smiles': int(len(train_can & val_can)), 'train_test_smiles': int(len(train_can & test_can)), 'val_test_smiles': int(len(val_can & test_can)), 'train_val_scaffold': int(len(train_sc & val_sc)), 'train_test_scaffold': int(len(train_sc & test_sc)), 'val_test_scaffold': int(len(val_sc & test_sc))}
    return rep

def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float('nan')
    return 1.0 - ss_res / ss_tot

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def ensemble_aggregate(member_preds_2d: np.ndarray) -> Dict[str, np.ndarray]:
    mp = np.asarray(member_preds_2d, dtype=float)
    mean = np.mean(mp, axis=0)
    median = np.median(mp, axis=0)
    std = np.std(mp, axis=0, ddof=0)
    q75 = np.quantile(mp, 0.75, axis=0)
    q25 = np.quantile(mp, 0.25, axis=0)
    iqr = q75 - q25
    return {'mean': mean, 'median': median, 'std': std, 'iqr': iqr}

def eval_ensemble_1d(y_true: np.ndarray, member_preds_2d: np.ndarray) -> Dict[str, Any]:
    agg = ensemble_aggregate(member_preds_2d)
    out = {'mean': {'R2': r2_score_safe(y_true, agg['mean']), 'MAE': mae(y_true, agg['mean']), 'RMSE': rmse(y_true, agg['mean'])}, 'median': {'R2': r2_score_safe(y_true, agg['median']), 'MAE': mae(y_true, agg['median']), 'RMSE': rmse(y_true, agg['median'])}, 'uncertainty': {'std_mean': float(np.mean(agg['std'])), 'iqr_mean': float(np.mean(agg['iqr']))}}
    return out

def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = (set(a), set(b))
    if len(sa | sb) == 0:
        return float('nan')
    return float(len(sa & sb) / len(sa | sb))

def save_member_split_overlap(outdir: Path, members: List[Dict[str, Any]], split_indices: Dict[str, List[int]]) -> None:
    rows = []
    for i in range(len(members)):
        for j in range(len(members)):
            rows.append({'member_i': i, 'member_j': j, 'train_jaccard': _jaccard(split_indices['train'], split_indices['train']), 'val_jaccard': _jaccard(split_indices['val'], split_indices['val']), 'test_jaccard': _jaccard(split_indices['test'], split_indices['test']), 'note': 'Fixed CSV splits → identical indices by design'})
    pd.DataFrame(rows).to_csv(outdir / 'member_split_overlap.csv', index=False)

def build_dataset_multitask(df: pd.DataFrame, targets: List[str], smiles_col: str='SMILES', target_transform: str='none') -> data.MoleculeDataset:
    smiles = df[smiles_col].astype(str).tolist()
    Y = np.stack([apply_target_transform(df[t].values.astype(float), target_transform) for t in targets], axis=1).astype(np.float32)
    dps = [data.MoleculeDatapoint.from_smi(smiles[i], Y[i]) for i in range(len(smiles))]
    return data.MoleculeDataset(dps)

def _extract_scaler_stats(output_scaler: Any, targets: List[str]) -> Dict[str, Any]:
    means = getattr(output_scaler, 'means', None)
    stds = getattr(output_scaler, 'stds', None)
    if means is None or stds is None:
        return {'available': False, 'raw_type': str(type(output_scaler))}
    means = np.asarray(means, dtype=float).reshape(-1).tolist()
    stds = np.asarray(stds, dtype=float).reshape(-1).tolist()
    per_target = {t: {'mean': means[i], 'std': stds[i]} for i, t in enumerate(targets)}
    return {'available': True, 'per_target': per_target}

# Model
def train_one_member_multitask(train_df: pd.DataFrame, val_df: pd.DataFrame, targets: List[str], out_dir: Path, seed: int, target_transform: str, batch_size: int, max_epochs: int, patience: int, num_workers: int, batch_norm: bool) -> Tuple[str, str, Dict[str, Any]]:
    set_global_seed(seed)
    train_dset = build_dataset_multitask(train_df, targets, target_transform=target_transform)
    val_dset = build_dataset_multitask(val_df, targets, target_transform=target_transform)
    output_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(output_scaler)
    output_transform = nn.transforms.UnscaleTransform.from_standard_scaler(output_scaler)
    scaler_stats = _extract_scaler_stats(output_scaler, targets)
    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    use_gpu = torch.cuda.is_available()
    accelerator = 'gpu' if use_gpu else 'cpu'
    precision = '32-true'
    n_tasks = len(targets)
    criterion = RMSE()
    metric_list = [RMSE(), MAE()]
    ffn = nn.RegressionFFN(n_tasks=n_tasks, criterion=criterion, output_transform=output_transform)
    try:
        mpnn = models.MPNN(nn.BondMessagePassing(), nn.NormAggregation(), ffn, batch_norm=bool(batch_norm), metrics=metric_list)
    except TypeError:
        mpnn = models.MPNN(nn.BondMessagePassing(), nn.NormAggregation(), ffn, batch_norm=bool(batch_norm))
    ckpt_dir = ensure_dir(out_dir / 'ckpts')
    ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), filename=f'best-member{seed}-{{epoch}}-{{val_loss:.4f}}', monitor='val_loss', mode='min', save_top_k=1)
    es_cb = EarlyStopping(monitor='val_loss', mode='min', patience=int(patience))
    trainer = pl.Trainer(max_epochs=int(max_epochs), logger=False, enable_checkpointing=True, callbacks=[ckpt_cb, es_cb], accelerator=accelerator, devices=1, precision=precision, deterministic=True, enable_progress_bar=False)
    trainer.fit(mpnn, train_loader, val_loader)
    best_ckpt_path = ckpt_cb.best_model_path
    best_model = models.MPNN.load_from_checkpoint(best_ckpt_path)
    model_pt_path = str(out_dir / 'model.pt')
    save_model(model_pt_path, best_model)
    write_json(out_dir / 'target_scaler_stats.json', scaler_stats)
    return (str(best_ckpt_path), model_pt_path, scaler_stats)

def predict_multitask_with_ckpt(ckpt_path: str, df: pd.DataFrame, targets: List[str], batch_size: int, num_workers: int, target_transform: str) -> np.ndarray:
    dset = build_dataset_multitask(df, targets, target_transform=target_transform)
    loader = data.build_dataloader(dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model = models.MPNN.load_from_checkpoint(ckpt_path)
    model.eval()
    pred_trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator='auto', devices=1, enable_progress_bar=False)
    with torch.inference_mode():
        out_batches = pred_trainer.predict(model, loader)
        y_pred = torch.cat(out_batches, dim=0).cpu().numpy()
    return y_pred

def run_chemprop_multitask_benchmark(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame, out_root: Path, seed: int, n_members: int, target_transform: str, batch_size: int, test_batch_size: int, max_epochs: int, patience: int, num_workers: int, batch_norm: bool) -> None:
    set_global_seed(seed)
    targets = select_targets(train_df)
    leakage = leakage_checks(train_df, val_df, test_df)
    split_indices = {'train': full_df.loc[full_df['_split'] == 'train', 'global_index'].tolist(), 'val': full_df.loc[full_df['_split'] == 'val', 'global_index'].tolist(), 'test': full_df.loc[full_df['_split'] == 'test', 'global_index'].tolist()}
    run_config = {'model': MODEL_NAME, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seed': int(seed), 'n_members': int(n_members), 'target_transform': str(target_transform), 'targets': targets, 'train_params': {'batch_size': int(batch_size), 'test_batch_size': int(test_batch_size), 'max_epochs': int(max_epochs), 'patience': int(patience), 'num_workers': int(num_workers), 'batch_norm': bool(batch_norm), 'target_scaling': True}, 'data_shape': {'train': list(train_df.shape), 'val': list(val_df.shape), 'test': list(test_df.shape)}, 'data_hash': {'train_sha256': sha256_df_content(train_df), 'val_sha256': sha256_df_content(val_df), 'test_sha256': sha256_df_content(test_df)}, 'split_indices': split_indices, 'versions': collect_versions(), 'seed_control': set_global_seed(seed)}
    outdir = ensure_dir(out_root / MODEL_NAME)
    models_dir = ensure_dir(outdir / 'models')
    pred_test = pd.DataFrame({'SMILES': test_df['SMILES'].astype(str).tolist()})
    metrics_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {'model': MODEL_NAME, 'n_members': int(n_members), 'targets': {t: {} for t in targets}, 'members': [], 'notes': {'multitask': True, 'target_scaling': True}}
    Y_val = np.stack([apply_target_transform(val_df[t].values.astype(float), target_transform) for t in targets], axis=1)
    Y_test = np.stack([apply_target_transform(test_df[t].values.astype(float), target_transform) for t in targets], axis=1)
    member_val_preds = []
    member_test_preds = []
    for i in range(n_members):
        member_seed = seed * 1000 + 310 + i
        member_out = ensure_dir(models_dir / f'member_{i:02d}')
        ckpt_path, pt_path, scaler_stats = train_one_member_multitask(train_df, val_df, targets, out_dir=member_out, seed=member_seed, target_transform=target_transform, batch_size=batch_size, max_epochs=max_epochs, patience=patience, num_workers=num_workers, batch_norm=batch_norm)
        pv = predict_multitask_with_ckpt(ckpt_path, val_df, targets, batch_size=test_batch_size, num_workers=num_workers, target_transform=target_transform)
        pt = predict_multitask_with_ckpt(ckpt_path, test_df, targets, batch_size=test_batch_size, num_workers=num_workers, target_transform=target_transform)
        member_val_preds.append(pv)
        member_test_preds.append(pt)
        summary['members'].append({'member': int(i), 'seed': int(member_seed), 'best_ckpt': ckpt_path, 'model_pt': pt_path, 'target_scaler_stats_file': str((member_out / 'target_scaler_stats.json').resolve())})
    member_val_preds = np.stack(member_val_preds, axis=0)
    member_test_preds = np.stack(member_test_preds, axis=0)
    for j, tgt in enumerate(targets):
        y_va = Y_val[:, j]
        y_te = Y_test[:, j]
        mv = member_val_preds[:, :, j]
        mt = member_test_preds[:, :, j]
        val_eval = eval_ensemble_1d(y_va, mv)
        test_eval = eval_ensemble_1d(y_te, mt)
        agg = ensemble_aggregate(mt)
        pred_test[f'y_true_{tgt}'] = y_te
        pred_test[f'y_mean_{tgt}'] = agg['mean']
        pred_test[f'y_median_{tgt}'] = agg['median']
        pred_test[f'y_std_{tgt}'] = agg['std']
        pred_test[f'y_iqr_{tgt}'] = agg['iqr']
        for i in range(n_members):
            pred_test[f'y_m{i:02d}_{tgt}'] = mt[i]
        for split_name, ev, n_samples in [('val', val_eval, len(y_va)), ('test', test_eval, len(y_te))]:
            for agg_name in ['mean', 'median']:
                metrics_rows.append({'model': MODEL_NAME, 'target': tgt, 'split': split_name, 'agg': agg_name, 'R2': ev[agg_name]['R2'], 'MAE': ev[agg_name]['MAE'], 'RMSE': ev[agg_name]['RMSE'], 'unc_std_mean': ev['uncertainty']['std_mean'], 'unc_iqr_mean': ev['uncertainty']['iqr_mean'], 'n_samples': int(n_samples), 'n_members': int(n_members)})
        summary['targets'][tgt] = {'val': val_eval, 'test': test_eval}
    save_member_split_overlap(outdir, summary['members'], split_indices)
    write_json(outdir / 'run_config.json', run_config)
    write_json(outdir / 'leakage_report.json', leakage)
    pd.DataFrame(metrics_rows).to_csv(outdir / 'metrics.csv', index=False)
    pred_test.to_csv(outdir / 'predictions.csv', index=False)
    write_json(outdir / 'summary.json', summary)
    print(f'[DONE] Saved to: {outdir.resolve()}')
# Run
train_df, val_df, test_df, full_df = load_splits(TRAIN_CSV, VAL_CSV, TEST_CSV)
run_chemprop_multitask_benchmark(train_df=train_df, val_df=val_df, test_df=test_df, full_df=full_df, out_root=Path(OUT_ROOT), seed=int(SEED), n_members=int(N_MEMBERS), target_transform=str(TARGET_TRANSFORM), batch_size=int(BATCH_SIZE), test_batch_size=int(TEST_BATCH_SIZE), max_epochs=int(MAX_EPOCHS), patience=int(PATIENCE), num_workers=int(NUM_WORKERS), batch_norm=bool(BATCH_NORM))
