# privacy/data_loader.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Confirmed from contracts: 50 PCA features, labels 1–6
FEATURE_COLS = [f"pca_{i}" for i in range(50)]
LABEL_COL = "label"
BATCH_SIZE = 64

def load_client_data(csv_path, client_delta_n):
    """
    Loads a client CSV, converts labels to 0-indexed, returns DataLoader.
    drop_last=True is mandatory for Opacus.
    
    client_delta_n: number of training samples — used to compute delta
    """
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_cols].values.astype("float32")
    y = df[LABEL_COL].values.astype("int64")
    
    # CRITICAL: convert labels 1–6 → 0–5
    y = y - 1
    
    # Verify
    assert X.shape[1] == 50, f"Expected 50 features, got {X.shape[1]}"
    assert set(y).issubset(set(range(6))), f"Labels out of range: {set(y)}"
    
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y)
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True  # Opacus hard requirement
    )
    
    delta = 1.0 / len(dataset)
    print(f"Loaded {csv_path}: {len(dataset)} samples, δ={delta:.2e}")
    return loader, delta