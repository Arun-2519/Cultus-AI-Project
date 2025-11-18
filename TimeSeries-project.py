import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Statsmodels for SARIMA baseline
import statsmodels.api as sm

# -----------------------------
# 1) DATA GENERATION + SAVE
# -----------------------------

def generate_dataset(n=2000, seed=42, save_path="./lstm_timeseries_dataset.csv"):
    np.random.seed(seed)
    date_index = pd.date_range(start="2000-01-01", periods=n, freq="D")

    # Components
    trend = 0.0005 * np.arange(n)               # slow linear trend
    seasonal_yearly = 2.0 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    seasonal_weekly = 0.5 * np.sin(2 * np.pi * np.arange(n) / 7.0)
    ar_component = np.zeros(n)
    phi = [0.6, -0.2]
    for t in range(2, n):
        ar_component[t] = phi[0] * ar_component[t-1] + phi[1] * ar_component[t-2]

    exog1 = 0.8 * np.sin(2 * np.pi * np.arange(n) / 90.0) + 0.2 * np.random.normal(size=n)
    exog2 = np.cos(2 * np.pi * np.arange(n) / 30.0) + 0.1 * np.random.normal(size=n)
    exog3 = np.random.normal(scale=0.5, size=n)

    noise = np.random.normal(scale=0.8, size=n)
    target = 3.0 + trend + seasonal_yearly + seasonal_weekly + 1.5 * ar_component + 0.7 * exog1 - 0.4 * exog2 + 0.3 * exog3 + noise

    max_lag = 14
    lags = {f"lag_{lag}": pd.Series(target).shift(lag).fillna(method="bfill").values for lag in range(1, max_lag+1)}

    df = pd.DataFrame({
        "timestamp": date_index,
        "target": target,
        "exog1": exog1,
        "exog2": exog2,
        "exog3": exog3,
        "trend": trend,
        "seasonal_yearly": seasonal_yearly,
        "seasonal_weekly": seasonal_weekly
    })
    for k, v in lags.items():
        df[k] = v

    df.to_csv(save_path, index=False)
    print(f"Dataset saved to: {save_path}")
    return df

# -----------------------------
# 2) PYTORCH SEQUENCE DATASET
# -----------------------------

class SequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col='target', seq_len=30):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return x, y

# -----------------------------
# 3) LSTM MODEL (PyTorch)
# -----------------------------

class LSTMForecast(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        direction = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out.squeeze(1)

# -----------------------------
# Utility: train / evaluate
# -----------------------------

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, weight_decay=1e-4, device='cpu', dropout_schedule=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        # optional dropout schedule: a function mapping epoch->dropout value
        if dropout_schedule is not None and hasattr(model, 'lstm'):
            new_dropout = float(dropout_schedule(ep))
            # update dropout in LSTM and fc dropout layers (if present)
            try:
                model.lstm.dropout = new_dropout
            except Exception:
                pass
            # update fc dropout layers
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = new_dropout

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_losses.append(mean_squared_error(yb.cpu().numpy(), preds.cpu().numpy()))
        val_rmse = math.sqrt(np.mean(val_losses))

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = model.state_dict()

        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep}/{epochs} - train_loss: {np.mean(train_losses):.4f} - val_rmse: {val_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -----------------------------
# Baselines: SARIMA and FFNN
# -----------------------------

def sarima_forecast(train_series, test_steps, order=(2,0,1), seasonal_order=(1,1,1,365)):
    # Note: seasonal_order's seasonal_periods is large (365) because daily data has yearly seasonality
    model = sm.tsa.statespace.SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=test_steps).predicted_mean
    return fc

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# -----------------------------
# Permutation Importance for sequences
# -----------------------------

def permutation_importance_sequence(model, X_test, y_test, feature_idx_to_permute, seq_len=30, device='cpu'):
    # X_test: numpy array shape (n_samples, seq_len, n_features)
    baseline_preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy()
    baseline_rmse = math.sqrt(mean_squared_error(y_test, baseline_preds))

    X_perm = X_test.copy()
    # permute along the sample axis while preserving time order inside each sample for that feature index
    for i in range(X_perm.shape[0]):
        # shuffle the feature values across samples
        X_perm[:, :, feature_idx_to_permute] = np.random.permutation(X_perm[:, :, feature_idx_to_permute])

    perm_preds = model(torch.tensor(X_perm, dtype=torch.float32).to(device)).detach().cpu().numpy()
    perm_rmse = math.sqrt(mean_squared_error(y_test, perm_preds))
    importance = perm_rmse - baseline_rmse
    return importance, baseline_rmse, perm_rmse

# -----------------------------
# FULL PIPELINE: load/generate data, preprocess, train, evaluate
# -----------------------------

def run_full_pipeline(csv_path='./lstm_timeseries_dataset.csv', seq_len=30, test_size=0.2, device='cpu'):
    # ensure dataset exists
    if not os.path.exists(csv_path):
        print("Dataset not found locally — generating now...")
        df = generate_dataset(save_path=csv_path)
    else:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    # features to use for sequence model (exclude timestamp and target)
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'target']]

    # Split train/val/test by time (no shuffling)
    n = len(df)
    test_n = int(n * test_size)
    train_val_n = n - test_n
    val_n = int(train_val_n * 0.1)
    train_n = train_val_n - val_n

    train_df = df.iloc[:train_n].copy()
    val_df = df.iloc[train_n:train_n+val_n].copy()
    test_df = df.iloc[train_n+val_n:].copy()

    # scaler fitted on train features only
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    for d in [train_df, val_df, test_df]:
        d[feature_cols] = scaler.transform(d[feature_cols].values)

    train_ds = SequenceDataset(train_df, feature_cols, seq_len=seq_len)
    val_ds = SequenceDataset(val_df, feature_cols, seq_len=seq_len)
    test_ds = SequenceDataset(test_df, feature_cols, seq_len=seq_len)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_features = len(feature_cols)
    model = LSTMForecast(n_features=n_features, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=False)

    # dropout schedule: increase dropout slowly to regularize later epochs
    def dropout_schedule(epoch):
        return min(0.5, 0.1 + (epoch / 100) * 0.4)

    model = train_model(model, train_loader, val_loader, epochs=60, lr=1e-3, weight_decay=1e-4, device=device, dropout_schedule=dropout_schedule)

    # Evaluate on test set
    model.to(device)
    model.eval()
    X_test = []
    y_test = []
    with torch.no_grad():
        for xb, yb in test_loader:
            X_test.append(xb.numpy())
            y_test.append(yb.numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy()

    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100

    print(f"LSTM Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

    # Baseline 1: SARIMA trained on the unscaled target series (last portion as test)
    train_series = df['target'].iloc[:train_n+val_n]
    test_steps = len(test_df) - seq_len
    try:
        sarima_pred = sarima_forecast(train_series.values, test_steps=test_steps)
        # align length
        sarima_y = df['target'].iloc[train_n+val_n+seq_len:train_n+val_n+seq_len+test_steps].values
        sarima_rmse = math.sqrt(mean_squared_error(sarima_y, sarima_pred))
        print(f"SARIMA Test RMSE (approx): {sarima_rmse:.4f}")
    except Exception as e:
        print("SARIMA failed (likely because statsmodels couldn't converge).", e)

    # Baseline 2: Feedforward NN on lag features (flatten last seq)
    # Prepare flattened lag features: use last seq_len timesteps flattened
    X_flat = []
    y_flat = []
    for i in range(len(df) - seq_len):
        X_flat.append(df[feature_cols].values[i:i+seq_len].flatten())
        y_flat.append(df['target'].values[i+seq_len])
    X_flat = np.array(X_flat)
    y_flat = np.array(y_flat)
    # train/val/test splits aligned
    X_train, X_val, X_test_flat = X_flat[:train_n-seq_len], X_flat[train_n-seq_len:train_n+val_n-seq_len], X_flat[train_n+val_n-seq_len:]
    y_train, y_val, y_test_flat = y_flat[:train_n-seq_len], y_flat[train_n-seq_len:train_n+val_n-seq_len], y_flat[train_n+val_n-seq_len:]

    ffnn = FFNN(input_dim=X_train.shape[1], hidden=256, dropout=0.3)
    optimizer = torch.optim.AdamW(ffnn.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    ffnn.to(device)
    # small training loop
    for epoch in range(25):
        ffnn.train()
        perm = np.random.permutation(len(X_train))
        batch = 128
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i+batch]
            xb = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
            yb = torch.tensor(y_train[idx], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            preds = ffnn(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"FFNN epoch {epoch} loss: {np.mean(losses):.4f}")

    ffnn.eval()
    with torch.no_grad():
        pred_ffnn = ffnn(torch.tensor(X_test_flat, dtype=torch.float32).to(device)).cpu().numpy()
    ffnn_rmse = math.sqrt(mean_squared_error(y_test_flat, pred_ffnn))
    print(f"FFNN (lag-flattened) RMSE: {ffnn_rmse:.4f}")

    # Permutation importance on LSTM (feature-level)
    print("Computing permutation importances (this may take a little while)...")
    importances = {}
    for i, feat in enumerate(feature_cols):
        imp, base, perm = permutation_importance_sequence(model, X_test, y_test, feature_idx_to_permute=i, seq_len=seq_len, device=device)
        importances[feat] = imp
    sorted_imps = sorted(importances.items(), key=lambda x: -abs(x[1]))
    print("Permutation importances (feature -> delta RMSE):")
    for feat, imp in sorted_imps:
        print(f"  {feat}: {imp:.4f}")

    results = {
        'lstm_rmse': rmse,
        'ffnn_rmse': ffnn_rmse,
        'sarima_rmse': sarima_rmse if 'sarima_rmse' in locals() else None,
        'permutation_importances': importances
    }
    return results

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == '__main__':
    csv_path = './lstm_timeseries_dataset.csv'
    generate_dataset(save_path=csv_path)
    # If GPU available, use it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    results = run_full_pipeline(csv_path=csv_path, seq_len=30, test_size=0.2, device=device)
    print('\nSummary of key results:')
    print(results)
    print('\nNotes:')
    print(' - To reproduce exactly, ensure package versions are stable (see top of file).')
    print(' - Training time will depend on CPU/GPU; on a typical laptop CPU expect several minutes; on a GPU it may be under 10 minutes.')
