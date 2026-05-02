"""
evaluation/cross_validation.py

5-fold stratified cross-validation for IntelliClave HAR classifier.

Runs CV on each client CSV independently, then on the combined dataset,
giving a split-independent estimate of model accuracy and F1.

Uses M1's model (HARClassifier) and data pipeline (StandardScaler, label
conversion) — no FL or DP involved, pure local evaluation.

Usage:
    python evaluation/cross_validation.py
    python evaluation/cross_validation.py --folds 10 --epochs 15
    python evaluation/cross_validation.py --client 1

Outputs:
    results/cross_validation.json   — full per-fold + summary results
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# ── resolve paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_FL   = os.path.join(_ROOT, "fl")
sys.path.insert(0, _FL)

from model import get_model                  # M1's HARClassifier
from data_utils import load_class_weights, ACTIVITY_NAMES  # M1's helpers
# ─────────────────────────────────────────────────────────────────────────────

N_CLASSES   = 6
LABEL_COL   = "label"
BATCH_SIZE  = 32
DEVICE      = torch.device("cpu")


# ── data helpers ──────────────────────────────────────────────────────────────

def load_raw(csv_path: str):
    """Load CSV → (X float32, y int64 0-based)."""
    df = pd.read_csv(csv_path).dropna()
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64) - 1   # 1-6 → 0-5
    return X, y


def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


# ── training helpers ──────────────────────────────────────────────────────────

def get_criterion():
    w = load_class_weights(device=DEVICE)
    return nn.CrossEntropyLoss(weight=w)


def train_fold(model, loader, criterion, epochs: int):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for X_b, y_b in loader:
            opt.zero_grad()
            criterion(model(X_b), y_b).backward()
            opt.step()


def eval_fold(model, loader):
    """Return y_true, y_pred, y_prob arrays."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b)
            probs  = torch.softmax(logits, dim=1).numpy()
            preds  = logits.argmax(1).numpy()
            y_true.extend(y_b.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(y_true, y_pred, y_prob):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_cls  = f1_score(y_true, y_pred, average=None,
                        zero_division=0, labels=list(range(N_CLASSES)))
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = None
    return {
        "accuracy":     round(float(acc), 6),
        "macro_f1":     round(float(macro_f1), 6),
        "auc_roc":      round(float(auc), 6) if auc is not None else None,
        "per_class_f1": {ACTIVITY_NAMES[i]: round(float(v), 6)
                         for i, v in enumerate(per_cls)},
    }


# ── core CV function ──────────────────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray,
           n_folds: int = 5, epochs: int = 10,
           label: str = "") -> dict:
    """
    Stratified k-fold CV on (X, y).
    Returns dict with per-fold metrics and summary statistics.
    """
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds  = []
    input_dim = X.shape[1]

    print(f"\n{'='*55}")
    print(f"Cross-validation: {label}  ({n_folds} folds, {epochs} epochs/fold)")
    print(f"  Samples: {len(y)}  |  Features: {input_dim}  |  Classes: {N_CLASSES}")
    print(f"{'='*55}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale per fold — fit on train only (no leakage)
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        train_loader = make_loader(X_tr, y_tr, shuffle=True)
        val_loader   = make_loader(X_val, y_val, shuffle=False)

        model     = get_model(input_dim, N_CLASSES).to(DEVICE)
        criterion = get_criterion()

        train_fold(model, train_loader, criterion, epochs)
        y_true, y_pred, y_prob = eval_fold(model, val_loader)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["fold"] = fold_idx
        folds.append(metrics)

        print(f"  Fold {fold_idx}/{n_folds}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"auc={metrics['auc_roc']}")

    # ── summary statistics ────────────────────────────────────────────────────
    accs  = [f["accuracy"]  for f in folds]
    f1s   = [f["macro_f1"]  for f in folds]
    aucs  = [f["auc_roc"]   for f in folds if f["auc_roc"] is not None]

    summary = {
        "accuracy":  {"mean": round(np.mean(accs), 6),
                      "std":  round(np.std(accs),  6),
                      "min":  round(np.min(accs),  6),
                      "max":  round(np.max(accs),  6)},
        "macro_f1":  {"mean": round(np.mean(f1s),  6),
                      "std":  round(np.std(f1s),   6),
                      "min":  round(np.min(f1s),   6),
                      "max":  round(np.max(f1s),   6)},
        "auc_roc":   {"mean": round(np.mean(aucs), 6),
                      "std":  round(np.std(aucs),  6)} if aucs else None,
    }

    print(f"\n  Summary ({n_folds}-fold):")
    print(f"    Accuracy : {summary['accuracy']['mean']:.4f} "
          f"± {summary['accuracy']['std']:.4f}")
    print(f"    Macro F1 : {summary['macro_f1']['mean']:.4f} "
          f"± {summary['macro_f1']['std']:.4f}")
    if summary["auc_roc"]:
        print(f"    AUC-ROC  : {summary['auc_roc']['mean']:.4f} "
              f"± {summary['auc_roc']['std']:.4f}")

    return {"label": label, "n_folds": n_folds, "epochs": epochs,
            "n_samples": len(y), "folds": folds, "summary": summary}


# ── main ──────────────────────────────────────────────────────────────────────

def main(n_folds: int = 5, epochs: int = 10, client: int = None):
    client_csvs = {
        1: os.path.join(_ROOT, "data", "processed", "client1.csv"),
        2: os.path.join(_ROOT, "data", "processed", "client2.csv"),
        3: os.path.join(_ROOT, "data", "processed", "client3.csv"),
    }

    targets = [client] if client else [1, 2, 3, "combined"]
    all_results = {}

    raw = {}   # cache raw arrays for combined run
    for cid in [1, 2, 3]:
        raw[cid] = load_raw(client_csvs[cid])

    for target in targets:
        if target == "combined":
            X = np.concatenate([raw[c][0] for c in [1, 2, 3]])
            y = np.concatenate([raw[c][1] for c in [1, 2, 3]])
            label = "combined (all clients)"
        else:
            X, y = raw[target]
            label = f"client{target} (FitLife)" if target == 1 else \
                    f"client{target} (MediTrack)" if target == 2 else \
                    f"client{target} (CareWatch)"

        result = run_cv(X, y, n_folds=n_folds, epochs=epochs, label=label)
        all_results[str(target)] = result

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(_ROOT, "results", "cross_validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved → {out_path}")

    # ── final table ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'Dataset':<30} {'Accuracy':>10} {'Macro F1':>10} {'AUC-ROC':>10}")
    print(f"{'-'*55}")
    for key, res in all_results.items():
        s = res["summary"]
        auc_str = f"{s['auc_roc']['mean']:.4f}" if s["auc_roc"] else "  N/A  "
        print(f"{res['label']:<30} "
              f"{s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}  "
              f"{s['macro_f1']['mean']:.4f}±{s['macro_f1']['std']:.4f}  "
              f"{auc_str}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",  type=int, default=5,
                        help="Number of CV folds (default 5)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per fold (default 10)")
    parser.add_argument("--client", type=int, default=None,
                        choices=[1, 2, 3],
                        help="Run CV on one client only (default: all + combined)")
    args = parser.parse_args()
    main(n_folds=args.folds, epochs=args.epochs, client=args.client)
