"""
security/attacks/build_surrogate.py

Create a surrogate model by querying the target model on available (public) features
and training a local model on the soft/hard labels. This simulates a black-box
attacker who can only obtain model outputs.

Usage:
    python security/attacks/build_surrogate.py --model-path <target.pth> --out results/attacks/surrogate.pth
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "fl"))

from model import get_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

DEFAULT_MODEL = os.path.join(_ROOT, "results", "fl_rounds", "global_model_latest.pth")
DEFAULT_OUT   = os.path.join(_ROOT, "results", "attacks", "surrogate.pth")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")


def _normalize_state_dict(state):
    if not isinstance(state, dict):
        return state
    if any(key.startswith("_module.") for key in state):
        return {key.removeprefix("_module."): value for key, value in state.items()}
    if any(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_public_features(label_col: str = "label"):
    csvs = sorted(
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    )
    if not csvs:
        raise FileNotFoundError("No CSVs in data/processed/ to build public features from")
    frames = [pd.read_csv(p).dropna() for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def main(model_path: str = DEFAULT_MODEL, out_path: str = DEFAULT_OUT,
         epochs: int = 8, lr: float = 1e-3, batch_size: int = 64):
    X = load_public_features()
    input_dim = X.shape[1]

    # Load target model to produce oracle labels (soft probs)
    # We load onto CPU for portability
    target = get_model(input_dim=input_dim, num_classes=get_num_classes())
    state = _normalize_state_dict(torch.load(model_path, map_location="cpu"))
    target.load_state_dict(state)
    target.eval()

    with torch.no_grad():
        logits = target(torch.from_numpy(X).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        hard_labels = probs.argmax(axis=1)

    # Train surrogate: same architecture, trained on (X, hard_labels)
    surrogate = get_model(input_dim=input_dim, num_classes=probs.shape[1])
    opt = optim.Adam(surrogate.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(hard_labels).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        surrogate.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = surrogate(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
        avg = total_loss / len(dataset)
        print(f"[surrogate] Epoch {epoch} loss={avg:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(surrogate.state_dict(), out_path)
    print(f"Saved surrogate -> {out_path}")


def get_num_classes():
    # infer number of classes from the first processed CSV
    csvs = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv'))
    first = os.path.join(PROCESSED_DIR, csvs[0])
    df = pd.read_csv(first, usecols=["label"]) 
    return int(df['label'].nunique())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(model_path=args.model_path, out_path=args.out, epochs=args.epochs, lr=args.lr)
