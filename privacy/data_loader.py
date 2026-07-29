# privacy/data_loader.py
"""
Data loader for the privacy module (Opacus DP training).

Fixes applied:
  - Scaler is now fitted on the training split only (no train/test leakage).
  - Uses the same median imputation as fl/data_utils.py instead of dropna().
  - drop_last=True is set on the train loader (Opacus hard requirement).
"""
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

LABEL_COL  = "label"
BATCH_SIZE = 64


def load_client_data(
    csv_path: str,
    batch_size: int = BATCH_SIZE,
    test_size: float = 0.2,
    random_state: int = 42,
    impute: str = "median",
):
    """
    Load a client CSV, impute missing values, split into train/test,
    fit the scaler on the training split only, and return a DataLoader
    suitable for Opacus (drop_last=True).

    Labels are shifted to 0-based automatically regardless of original range.

    Parameters
    ----------
    csv_path     : path to the client CSV
    batch_size   : DataLoader batch size (default 64)
    test_size    : fraction held out for evaluation (default 0.2)
    random_state : random seed (default 42)
    impute       : "median" | "mean" | "mode" | "drop" (default "median")

    Returns
    -------
    loader      : DataLoader  (drop_last=True for Opacus)
    delta       : float       1 / n_train_samples
    num_classes : int
    input_dim   : int
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != LABEL_COL]

    # ── Missing value handling ────────────────────────────────────────────────
    total_missing = df[feature_cols].isna().sum().sum()
    if total_missing > 0:
        if impute == "drop":
            before = len(df)
            df = df.dropna()
            warnings.warn(
                f"[data_loader] {csv_path}: dropped {before - len(df)} rows "
                f"with missing values. Consider impute='median'.",
                UserWarning, stacklevel=2,
            )
        elif impute == "median":
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        elif impute == "mean":
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
        elif impute == "mode":
            for col in feature_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
    # ─────────────────────────────────────────────────────────────────────────

    X     = df[feature_cols].values.astype("float32")
    raw_y = df[LABEL_COL].values

    # Auto-detect label encoding
    unique = sorted(np.unique(raw_y).tolist())
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique):
        offset = int(min(unique))
        y = raw_y.astype(np.int64) - offset
    else:
        label_map = {name: idx for idx, name in enumerate(unique)}
        y = np.array([label_map[v] for v in raw_y], dtype=np.int64)

    num_classes = len(unique)
    input_dim   = X.shape[1]

    # ── Split first, then fit scaler on train only (no leakage) ──────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    # (test split not used here but kept consistent with fl/data_utils.py)
    # ─────────────────────────────────────────────────────────────────────────

    dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.LongTensor(y_tr),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,   # Opacus hard requirement
    )

    delta = 1.0 / len(dataset)
    print(f"Loaded {csv_path}: train={len(dataset)}, "
          f"features={input_dim}, classes={num_classes}, δ={delta:.2e}")
    return loader, delta, num_classes, input_dim
