# ts_attention_project_quick.py

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
import time
import json
import textwrap

def main():
    # ---------------------------
    # CLI args (use parse_known_args to ignore unexpected kernel args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full (slower) experiment. Default is quick mode.')
    args = parser.parse_known_args()[0]  # <- FIX to ignore Jupyter kernel args
    QUICK = not args.full

    # ---------------------------
    # USER CONFIG (tweak here)
    MODE = "torch"   # "torch" or "numpy"
    T = 1200
    input_window = 48
    forecast_horizon = 12
    save_dir = "./ts_artifacts_quick"
    os.makedirs(save_dir, exist_ok=True)
    random.seed(0)
    np.random.seed(0)

    # Mode-dependent speed settings
    if QUICK:
        print("RUNNING IN QUICK MODE — reduced epochs, fewer folds, smaller search (for fast iteration).")
        walk_forward_steps = 3
        transformer_search_trials = 2
        transformer_epochs_default = 4
        transformer_candidates = {
            'd_model': [24, 32],
            'nhead': [2],
            'num_layers': [1],
            'lr': [1e-3],
            'epochs': [transformer_epochs_default],
        }
        gb_search_space = {
            'learning_rate': [0.1],
            'n_estimators': [30],
            'max_depth': [3]
        }
        batch_size = 32
    else:
        print("RUNNING IN FULL MODE — thorough hyperparam search and more epochs (slower).")
        walk_forward_steps = 8
        transformer_search_trials = 6
        transformer_epochs_default = 12
        transformer_candidates = {
            'd_model': [32, 64],
            'nhead': [2, 4],
            'num_layers': [1, 2],
            'lr': [1e-3, 5e-4],
            'epochs': [transformer_epochs_default],
        }
        gb_search_space = {
            'learning_rate': [0.05, 0.1],
            'n_estimators': [50, 100],
            'max_depth': [3, 5]
        }
        batch_size = 64

    # ---------------------------
    # Metrics & small utilities
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
        return np.mean(np.abs((y_true - y_pred) / denom)) * 100

    def compute_metrics(y_true, y_pred):
        mae_per_step = [mean_absolute_error(y_true[:,i], y_pred[:,i]) for i in range(y_pred.shape[1])]
        rmse_per_step = [math.sqrt(mean_squared_error(y_true[:,i], y_pred[:,i])) for i in range(y_pred.shape[1])]
        mape_per_step = [ (np.mean(np.abs((y_true[:,i] - y_pred[:,i]) / np.where(np.abs(y_true[:,i])<1e-8,1e-8,np.abs(y_true[:,i]))))*100) for i in range(y_pred.shape[1])]
        r2_per_step = [r2_score(y_true[:,i], y_pred[:,i]) for i in range(y_pred.shape[1])]
        return {
            'MAE_mean': np.mean(mae_per_step),
            'RMSE_mean': np.mean(rmse_per_step),
            'MAPE_mean': np.mean(mape_per_step),
            'R2_mean': np.mean(r2_per_step),
            'MAE_per_step': mae_per_step,
            'RMSE_per_step': rmse_per_step,
            'MAPE_per_step': mape_per_step,
            'R2_per_step': r2_per_step
        }

    def directional_accuracy(y_true, y_pred, histories):
        n, h = y_true.shape
        correct = 0
        total = n * h
        for i in range(n):
            last = histories[i]
            for j in range(h):
                true_dir = np.sign(y_true[i, j] - last)
                pred_dir = np.sign(y_pred[i, j] - last)
                if true_dir == pred_dir:
                    correct += 1
        return correct / total if total > 0 else np.nan

    # ---------------------------
    # 1) Synthetic multivariate TS (same generator)
    np.random.seed(42)
    t = np.arange(T)
    trend = 0.001 * t
    season1 = 2.0 * np.sin(2 * np.pi * t / 24)
    season2 = 1.0 * np.sin(2 * np.pi * t / 168)
    season3 = 0.5 * np.sin(2 * np.pi * t / 365)
    structural_break = np.where(t > int(T*0.6), 1.2 * np.sin(2 * np.pi * (t-int(T*0.6))/50), 0.0)
    noise = 0.5 * np.random.randn(T)

    series = np.zeros((T, 3))
    series[:,0] = 5 + trend + season1 + 0.6*season2 + noise
    series[:,1] = 3 + 0.4*trend + 0.2*season1 + 1.2*season2 + 0.2*np.random.randn(T)
    series[:,2] = 1 -0.2*trend + 0.1*season1 + 0.3*season3 + 0.3*np.random.randn(T) + structural_break

    dates = pd.date_range("2000-01-01", periods=T, freq="H")
    df = pd.DataFrame(series, columns=["x1","x2","x3"], index=dates)
    df['target'] = df['x1']
    df.to_csv(os.path.join(save_dir, "synthetic_series.csv"))

    def create_supervised(data_array, input_window, horizon):
        X, y = [], []
        for i in range(len(data_array) - input_window - horizon + 1):
            X.append(data_array[i:i+input_window])
            y.append(data_array[i+input_window:i+input_window+horizon, 0])
        return np.array(X), np.array(y)

    data_array = df[['x1','x2','x3']].values
    X_all, y_all = create_supervised(data_array, input_window, forecast_horizon)
    n_total = X_all.shape[0]
    holdout_size = int(0.15 * n_total)
    X_holdout, y_holdout = X_all[-holdout_size:], y_all[-holdout_size:]
    X_work, y_work = X_all[:-holdout_size], y_all[:-holdout_size]

    def stdize_with_stats(X, mean, std):
        return (X - mean) / std

    def make_lag_features(X_window):
        feats = []
        for lag in range(input_window):
            feats.append(X_window[:, input_window-1-lag, 0])
        return np.vstack(feats).T

    # ---------------------------
    # Torch availability
    USE_TORCH = False
    if MODE.lower() == "torch":
        try:
            import torch, torch.nn as nn
            USE_TORCH = True
        except Exception as e:
            print("PyTorch not available; falling back to NUMPY mode. Error:", e)
            USE_TORCH = False
            MODE = "numpy"

    if USE_TORCH:
        import torch, torch.nn as nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]
                return x

        class TransformerForecast(nn.Module):
            def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, horizon=12):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.pos_enc = PositionalEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(d_model * input_window, 128), nn.ReLU(), nn.Linear(128, horizon))
                self.probe_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
            def forward(self, x):
                x = self.input_proj(x)
                x = self.pos_enc(x)
                enc_out = self.encoder(x)
                attn_out, attn_weights = self.probe_attn(enc_out, enc_out, enc_out, need_weights=True, average_attn_weights=False)
                out = self.head(enc_out)
                return out, attn_weights

    # ---------------------------
    # Numpy attention helper
    def numpy_attention_contexts(X_window, d_model=24, seed=0):
        np.random.seed(seed)
        Wq = np.random.randn(X_window.shape[2], d_model) * 0.1
        Wk = np.random.randn(X_window.shape[2], d_model) * 0.1
        Wv = np.random.randn(X_window.shape[2], d_model) * 0.1
        Q = X_window @ Wq
        K = X_window @ Wk
        V = X_window @ Wv
        dk = d_model
        contexts = []
        attn_weights_all = []
        for i in range(X_window.shape[0]):
            q = Q[i]; k = K[i]; v = V[i]
            scores = q @ k.T / math.sqrt(dk)
            exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights = exp / np.sum(exp, axis=1, keepdims=True)
            context = weights @ v
            contexts.append(context.mean(axis=0))
            attn_weights_all.append(weights)
        return np.vstack(contexts), np.array(attn_weights_all)

    # ---------------------------
    # Lightweight time-series hyperparameter helpers (quick versions)
    def time_series_grid_search_gb(X, y, param_grid, initial_train_size, val_size, step):
        configs = list(ParameterGrid(param_grid))
        scores = {str(cfg): [] for cfg in configs}
        n = X.shape[0]
        start = initial_train_size
        while start + val_size <= n:
            train_idx = np.arange(0, start)
            val_idx = np.arange(start, start + val_size)
            Xtr = X[train_idx]; ytr = y[train_idx]
            Xv = X[val_idx]; yv = y[val_idx]
            for cfg in configs:
                preds = np.zeros((len(val_idx), y.shape[1]))
                for h in range(y.shape[1]):
                    m = GradientBoostingRegressor(**cfg)
                    m.fit(Xtr, ytr[:, h])
                    preds[:, h] = m.predict(Xv)
                mse = ((preds - yv) ** 2).mean()
                scores[str(cfg)].append(mse)
            start += step
            # quick mode: break earlier to limit runtime
            if QUICK and len(scores[str(configs[0])]) >= 2:
                break
        avg_scores = {cfg: np.mean(v) if len(v)>0 else np.inf for cfg, v in scores.items()}
        best_cfg = min(avg_scores, key=avg_scores.get)
        return eval(best_cfg)

    def time_series_random_search_transformer(X, y, space, initial_train_size, val_size, step, max_trials=3):
        keys = list(space.keys())
        candidates = []
        for _ in range(max_trials):
            cfg = {}
            for k in keys:
                cfg[k] = random.choice(space[k])
            candidates.append(cfg)
        scores = {str(cfg): [] for cfg in candidates}
        n = X.shape[0]
        start = initial_train_size
        while start + val_size <= n:
            train_idx = np.arange(0, start)
            val_idx = np.arange(start, start + val_size)
            Xtr = X[train_idx]; ytr = y[train_idx]
            Xv = X[val_idx]; yv = y[val_idx]
            mean = Xtr.mean(axis=(0,1)); std = Xtr.std(axis=(0,1)) + 1e-8
            Xtr_s = stdize_with_stats(Xtr, mean, std); Xv_s = stdize_with_stats(Xv, mean, std)
            for cfg in candidates:
                try:
                    model = TransformerForecast(n_features=Xtr.shape[2],
                                                d_model=cfg['d_model'],
                                                nhead=cfg['nhead'],
                                                num_layers=cfg['num_layers'],
                                                dim_feedforward=max(128, cfg['d_model']*2),
                                                horizon=ytr.shape[1]).to(device)
                    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
                    loss_fn = nn.MSELoss()
                    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32).to(device)
                    Ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
                    for epoch in range(int(cfg.get('epochs', transformer_epochs_default))):
                        model.train()
                        perm = np.random.permutation(len(Xtr_t))
                        for i in range(0, len(perm), batch_size):
                            idx = perm[i:i+batch_size]
                            xb = Xtr_t[idx]; yb = Ytr_t[idx]
                            opt.zero_grad()
                            pred, _ = model(xb)
                            loss = loss_fn(pred, yb)
                            loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad():
                        pred_val, _ = model(torch.tensor(Xv_s, dtype=torch.float32).to(device))
                    mse = ((pred_val.cpu().numpy() - yv) ** 2).mean()
                    scores[str(cfg)].append(mse)
                    del model
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception:
                    scores[str(cfg)].append(np.inf)
            start += step
            if QUICK and start > initial_train_size + 2*step:
                break
        avg_scores = {cfg: np.mean(v) if len(v)>0 else np.inf for cfg, v in scores.items()}
        best_cfg = min(avg_scores, key=avg_scores.get)
        return eval(best_cfg)

    # ---------------------------
    # Walk-forward evaluators (simplified)
    def walk_forward_gb(X_all, y_all, n_folds=3, initial_train_fraction=0.6):
        n = X_all.shape[0]
        fold_size = max(1, (n - int(initial_train_fraction*n)) // n_folds)
        train_end = int(initial_train_fraction*n)
        preds_collect = []; trues_collect = []; histories=[]
        best_gb = time_series_grid_search_gb(make_lag_features(X_all), y_all, gb_search_space, initial_train_size=max(50,int(0.4*n)), val_size=max(20,int(0.1*n)), step=max(10, int(0.05*n)))
        print("Selected GB config:", best_gb)
        fold = 0
        while train_end + fold_size <= n:
            Xtr = X_all[:train_end]; ytr = y_all[:train_end]
            Xte = X_all[train_end:train_end+fold_size]; yte = y_all[train_end:train_end+fold_size]
            Xtr_base = make_lag_features(Xtr); Xte_base = make_lag_features(Xte)
            models = []
            for h in range(ytr.shape[1]):
                m = GradientBoostingRegressor(**best_gb)
                m.fit(Xtr_base, ytr[:, h]); models.append(m)
            preds = np.column_stack([m.predict(Xte_base) for m in models])
            preds_collect.append(preds); trues_collect.append(yte); histories.append(Xte[:,-1,0])
            train_end += fold_size; fold += 1
            if QUICK and fold >= n_folds:
                break
        preds_all = np.vstack(preds_collect); trues_all = np.vstack(trues_collect)
        histories_all = np.concatenate(histories)
        metrics = compute_metrics(trues_all, preds_all)
        metrics['DirectionalAccuracy'] = directional_accuracy(trues_all, preds_all, histories_all)
        return preds_all, trues_all, metrics

    def walk_forward_numpy_attn(X_all, y_all, n_folds=3, initial_train_fraction=0.6):
        n = X_all.shape[0]
        fold_size = max(1, (n - int(initial_train_fraction*n)) // n_folds)
        train_end = int(initial_train_fraction*n)
        preds_collect=[]; trues_collect=[]; histories=[]; attns=[]
        fold = 0
        while train_end + fold_size <= n:
            Xtr = X_all[:train_end]; ytr = y_all[:train_end]
            Xte = X_all[train_end:train_end+fold_size]; yte = y_all[train_end:train_end+fold_size]
            mean = Xtr.mean(axis=(0,1)); std = Xtr.std(axis=(0,1))+1e-8
            Xtr_s = stdize_with_stats(Xtr, mean, std); Xte_s = stdize_with_stats(Xte, mean, std)
            ctx_tr, attn_tr = numpy_attention_contexts(Xtr_s, d_model=24, seed=0)
            ctx_te, attn_te = numpy_attention_contexts(Xte_s, d_model=24, seed=0)
            mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=0)
            mlp.fit(ctx_tr, ytr)
            preds = mlp.predict(ctx_te)
            preds_collect.append(preds); trues_collect.append(yte); histories.append(Xte[:,-1,0]); attns.append(attn_te)
            train_end += fold_size; fold += 1
            if QUICK and fold >= n_folds:
                break
        preds_all = np.vstack(preds_collect); trues_all = np.vstack(trues_collect)
        metrics = compute_metrics(trues_all, preds_all); metrics['DirectionalAccuracy'] = directional_accuracy(trues_all, preds_all, np.concatenate(histories))
        return preds_all, trues_all, metrics, attns

    def walk_forward_transformer(X_all, y_all, n_folds=3, initial_train_fraction=0.6):
        if not USE_TORCH:
            raise RuntimeError("Torch not available.")
        n = X_all.shape[0]
        fold_size = max(1, (n - int(initial_train_fraction*n)) // n_folds)
        train_end = int(initial_train_fraction*n)
        preds_collect=[]; trues_collect=[]; histories=[]; attns=[]
        best_cfg = time_series_random_search_transformer(X_all, y_all, transformer_candidates, initial_train_size=max(50,int(0.4*n)), val_size=max(20,int(0.1*n)), step=max(10,int(0.05*n)), max_trials=transformer_search_trials)
        print("Selected Transformer cfg:", best_cfg)
        fold = 0
        while train_end + fold_size <= n:
            Xtr = X_all[:train_end]; ytr = y_all[:train_end]
            Xte = X_all[train_end:train_end+fold_size]; yte = y_all[train_end:train_end+fold_size]
            mean = Xtr.mean(axis=(0,1)); std = Xtr.std(axis=(0,1))+1e-8
            Xtr_s = stdize_with_stats(Xtr, mean, std); Xte_s = stdize_with_stats(Xte, mean, std)
            Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32).to(device)
            Ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
            Xte_t = torch.tensor(Xte_s, dtype=torch.float32).to(device)
            cfg = best_cfg.copy()
            epochs = int(cfg.get('epochs', transformer_epochs_default))
            model = TransformerForecast(n_features=Xtr.shape[2], d_model=cfg['d_model'], nhead=cfg['nhead'], num_layers=cfg['num_layers'], dim_feedforward=max(128,cfg['d_model']*2), horizon=ytr.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            loss_fn = nn.MSELoss()
            for epoch in range(epochs):
                model.train()
                perm = np.random.permutation(len(Xtr_t))
                for i in range(0, len(perm), batch_size):
                    idx = perm[i:i+batch_size]
                    xb = Xtr_t[idx]; yb = Ytr_t[idx]
                    opt.zero_grad()
                    pred, _ = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                preds, attw = model(Xte_t)
            preds = preds.cpu().numpy(); attw = attw.cpu().numpy()
            preds_collect.append(preds); trues_collect.append(yte); histories.append(Xte[:,-1,0]); attns.append(attw)
            train_end += fold_size; fold += 1
            if QUICK and fold >= n_folds:
                break
            del model; 
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        preds_all = np.vstack(preds_collect); trues_all = np.vstack(trues_collect)
        metrics = compute_metrics(trues_all, preds_all); metrics['DirectionalAccuracy'] = directional_accuracy(trues_all, preds_all, np.concatenate(histories))
        return preds_all, trues_all, metrics, attns, best_cfg

    # ---------------------------
    # Run quick experiments
    print("Starting experiments (quick mode)" if QUICK else "Starting experiments (full mode)")
    preds_gb, trues_gb, metrics_gb = walk_forward_gb(X_work, y_work, n_folds=walk_forward_steps, initial_train_fraction=0.6)
    print("GB walk-forward metrics:", metrics_gb)

    if USE_TORCH:
        preds_tf, trues_tf, metrics_tf, attn_tf_list, tf_cfg = walk_forward_transformer(X_work, y_work, n_folds=walk_forward_steps, initial_train_fraction=0.6)
        print("Transformer walk-forward metrics:", metrics_tf)
    else:
        preds_attn, trues_attn, metrics_attn, attn_list = walk_forward_numpy_attn(X_work, y_work, n_folds=walk_forward_steps, initial_train_fraction=0.6)
        print("NumPy-attn walk-forward metrics:", metrics_attn)

    # Final holdout evaluation (train final models on full work set)
    mean_full = X_work.mean(axis=(0,1)); std_full = X_work.std(axis=(0,1))+1e-8
    X_work_s = stdize_with_stats(X_work, mean_full, std_full)
    X_holdout_s = stdize_with_stats(X_holdout, mean_full, std_full)

    # GB final
    X_work_base = make_lag_features(X_work); X_holdout_base = make_lag_features(X_holdout)
    best_gb_final = time_series_grid_search_gb(X_work_base, y_work, gb_search_space, initial_train_size=max(50,int(0.4*X_work_base.shape[0])), val_size=max(20,int(0.1*X_work_base.shape[0])), step=max(10,int(0.05*X_work_base.shape[0])))
    gb_models_final = []
    for h in range(y_work.shape[1]):
        m = GradientBoostingRegressor(**best_gb_final); m.fit(X_work_base, y_work[:,h]); gb_models_final.append(m)
    preds_gb_holdout = np.column_stack([m.predict(X_holdout_base) for m in gb_models_final])
    metrics_gb_holdout = compute_metrics(y_holdout, preds_gb_holdout); metrics_gb_holdout['DirectionalAccuracy'] = directional_accuracy(y_holdout, preds_gb_holdout, X_holdout[:,-1,0])
    print("GB holdout metrics:", metrics_gb_holdout)

    # Attention final
    if USE_TORCH:
        tf_cfg_use = tf_cfg if 'tf_cfg' in locals() else {'d_model':32,'nhead':2,'num_layers':1,'lr':1e-3,'epochs':transformer_epochs_default}
        model_final = TransformerForecast(n_features=X_work.shape[2], d_model=tf_cfg_use['d_model'], nhead=tf_cfg_use['nhead'], num_layers=tf_cfg_use['num_layers'], dim_feedforward=max(128,tf_cfg_use['d_model']*2), horizon=y_work.shape[1]).to(device)
        opt = torch.optim.Adam(model_final.parameters(), lr=tf_cfg_use['lr'])
        loss_fn = nn.MSELoss()
        Xtr_t = torch.tensor(X_work_s, dtype=torch.float32).to(device); Ytr_t = torch.tensor(y_work, dtype=torch.float32).to(device)
        epochs = int(tf_cfg_use.get('epochs', transformer_epochs_default))
        for epoch in range(epochs):
            model_final.train()
            perm = np.random.permutation(len(Xtr_t))
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i+batch_size]; xb = Xtr_t[idx]; yb = Ytr_t[idx]
                opt.zero_grad(); pred, _ = model_final(xb); loss = loss_fn(pred, yb); loss.backward(); opt.step()
        model_final.eval()
        with torch.no_grad():
            preds_tf_holdout, attn_holdout = model_final(torch.tensor(X_holdout_s, dtype=torch.float32).to(device))
        preds_tf_holdout = preds_tf_holdout.cpu().numpy(); attn_holdout = attn_holdout.cpu().numpy()
        metrics_tf_holdout = compute_metrics(y_holdout, preds_tf_holdout); metrics_tf_holdout['DirectionalAccuracy'] = directional_accuracy(y_holdout, preds_tf_holdout, X_holdout[:,-1,0])
        print("Transformer holdout metrics:", metrics_tf_holdout)
    else:
        ctx_work, attn_work = numpy_attention_contexts(X_work_s, d_model=24, seed=0)
        ctx_holdout, attn_holdout = numpy_attention_contexts(X_holdout_s, d_model=24, seed=0)
        mlp_final = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=0)
        mlp_final.fit(ctx_work, y_work)
        preds_attn_holdout = mlp_final.predict(ctx_holdout)
        metrics_attn_holdout = compute_metrics(y_holdout, preds_attn_holdout); metrics_attn_holdout['DirectionalAccuracy'] = directional_accuracy(y_holdout, preds_attn_holdout, X_holdout[:,-1,0])
        print("NumPy-attn holdout metrics:", metrics_attn_holdout)

    # Save a compact results CSV
    rows = []
    if USE_TORCH:
        rows.append(["Transformer_walk", metrics_tf['MAE_mean'], metrics_tf['RMSE_mean'], metrics_tf['MAPE_mean'], metrics_tf['R2_mean'], metrics_tf['DirectionalAccuracy']])
        rows.append(["Transformer_holdout", metrics_tf_holdout['MAE_mean'], metrics_tf_holdout['RMSE_mean'], metrics_tf_holdout['MAPE_mean'], metrics_tf_holdout['R2_mean'], metrics_tf_holdout['DirectionalAccuracy']])
    else:
        rows.append(["NumPyAttn_walk", metrics_attn['MAE_mean'], metrics_attn['RMSE_mean'], metrics_attn['MAPE_mean'], metrics_attn['R2_mean'], metrics_attn['DirectionalAccuracy']])
        rows.append(["NumPyAttn_holdout", metrics_attn_holdout['MAE_mean'], metrics_attn_holdout['RMSE_mean'], metrics_attn_holdout['MAPE_mean'], metrics_attn_holdout['R2_mean'], metrics_attn_holdout['DirectionalAccuracy']])
    rows.append(["GB_walk", metrics_gb['MAE_mean'], metrics_gb['RMSE_mean'], metrics_gb['MAPE_mean'], metrics_gb['R2_mean'], metrics_gb['DirectionalAccuracy']])
    rows.append(["GB_holdout", metrics_gb_holdout['MAE_mean'], metrics_gb_holdout['RMSE_mean'], metrics_gb_holdout['MAPE_mean'], metrics_gb_holdout['R2_mean'], metrics_gb_holdout['DirectionalAccuracy']])
    df_results = pd.DataFrame(rows, columns=["model","MAE","RMSE","MAPE","R2","DirAcc"])
    df_results.to_csv(os.path.join(save_dir, "results_quick.csv"), index=False)
    print("Results saved to:", os.path.join(save_dir, "results_quick.csv"))

    # Save attention for inspection
    try:
        if USE_TORCH:
            np.save(os.path.join(save_dir, "attn_holdout.npy"), attn_holdout)
        else:
            np.save(os.path.join(save_dir, "attn_holdout.npy"), attn_holdout)
    except Exception:
        pass

    print("Done. Artifacts in", save_dir)

if __name__ == "__main__":
    main()
