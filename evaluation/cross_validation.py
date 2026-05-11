"""
evaluation/cross_validation.py

Stratified k-fold cross-validation for the IntelliClave classifier.

Works with any tabular CSV dataset — class names and number of classes are
inferred from the data at runtime.

Usage:
    # Default: run CV on all client CSVs found in data/processed/
    python evaluation/cross_validation.py

    # Specify CSV files explicitly
    python evaluation/cross_validation.py --csvs data/processed/client1.csv data/processed/client2.csv

    # Single file
    python evaluation/cross_validation.py --csvs data/processed/client1.csv

    # Tune folds / epochs
    python evaluation/cross_validation.py --folds 10 --epochs 15

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

from model import get_model                          # FLClassifier
from data_utils import load_class_weights            # generic weight loader
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COL  = "label"
BATCH_SIZE = 32
DEVICE     = torch.device("cpu")


# ── data helpers ──────────────────────────────────────────────────────────────

def load_raw(csv_path: str):
    """
    Load CSV → (X float32, y int64 0-based, class_names list).

    Handles both numeric and string labels.
    """
    df = pd.read_csv(csv_path).dropna()
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols].values.astype(np.float32)
    raw_y = df[LABEL_COL].values

    unique = sorted(np.unique(raw_y).tolist())
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique):
        offset = int(min(unique))
        y = raw_y.astype(np.int64) - offset
        class_names = [f"class_{int(v)}" for v in unique]
    else:
        label_map = {name: idx for idx, name in enumerate(unique)}
        y = np.array([label_map[v] for v in raw_y], dtype=np.int64)
        class_names = [str(v) for v in unique]

    return X, y, class_names


def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


# ── training helpers ──────────────────────────────────────────────────────────

def get_criterion(num_classes: int):
    w = load_class_weights(num_classes=num_classes, device=DEVICE)
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


def compute_metrics(y_true, y_pred, y_prob, class_names):
    n_classes = len(class_names)
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_cls  = f1_score(
        y_true, y_pred, average=None,
        zero_division=0, labels=list(range(n_classes))
    )
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = None
    return {
        "accuracy":     round(float(acc), 6),
        "macro_f1":     round(float(macro_f1), 6),
        "auc_roc":      round(float(auc), 6) if auc is not None else None,
        "per_class_f1": {
            class_names[i]: round(float(v), 6)
            for i, v in enumerate(per_cls)
        },
    }


# ── core CV function ──────────────────────────────────────────────────────────

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list,
    n_folds: int = 5,
    epochs: int = 10,
    label: str = "",
) -> dict:
    """
    Stratified k-fold CV on (X, y).
    Returns dict with per-fold metrics and summary statistics.
    """
    n_classes = len(class_names)
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds     = []
    input_dim = X.shape[1]

    print(f"\n{'='*55}")
    print(f"Cross-validation: {label}  ({n_folds} folds, {epochs} epochs/fold)")
    print(f"  Samples: {len(y)}  |  Features: {input_dim}  |  Classes: {n_classes}")
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

        model     = get_model(input_dim, n_classes).to(DEVICE)
        criterion = get_criterion(n_classes)

        train_fold(model, train_loader, criterion, epochs)
        y_true, y_pred, y_prob = eval_fold(model, val_loader)
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        metrics["fold"] = fold_idx
        folds.append(metrics)

        print(f"  Fold {fold_idx}/{n_folds}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"auc={metrics['auc_roc']}")

    # ── summary statistics ────────────────────────────────────────────────────
    accs = [f["accuracy"]  for f in folds]
    f1s  = [f["macro_f1"]  for f in folds]
    aucs = [f["auc_roc"]   for f in folds if f["auc_roc"] is not None]

    summary = {
        "accuracy": {
            "mean": round(np.mean(accs), 6),
            "std":  round(np.std(accs),  6),
            "min":  round(np.min(accs),  6),
            "max":  round(np.max(accs),  6),
        },
        "macro_f1": {
            "mean": round(np.mean(f1s), 6),
            "std":  round(np.std(f1s),  6),
            "min":  round(np.min(f1s),  6),
            "max":  round(np.max(f1s),  6),
        },
        "auc_roc": {
            "mean": round(np.mean(aucs), 6),
            "std":  round(np.std(aucs),  6),
        } if aucs else None,
    }

    print(f"\n  Summary ({n_folds}-fold):")
    print(f"    Accuracy : {summary['accuracy']['mean']:.4f} "
          f"± {summary['accuracy']['std']:.4f}")
    print(f"    Macro F1 : {summary['macro_f1']['mean']:.4f} "
          f"± {summary['macro_f1']['std']:.4f}")
    if summary["auc_roc"]:
        print(f"    AUC-ROC  : {summary['auc_roc']['mean']:.4f} "
              f"± {summary['auc_roc']['std']:.4f}")

    return {
        "label": label,
        "n_folds": n_folds,
        "epochs": epochs,
        "n_samples": len(y),
        "class_names": class_names,
        "folds": folds,
        "summary": summary,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def discover_client_csvs() -> list:
    """Return all client*.csv files found in data/processed/, sorted."""
    processed_dir = os.path.join(_ROOT, "data", "processed")
    if not os.path.isdir(processed_dir):
        return []
    return sorted(
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir)
        if f.endswith(".csv")
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(csv_paths: list, n_folds: int = 5, epochs: int = 10):
    if not csv_paths:
        csv_paths = discover_client_csvs()
    if not csv_paths:
        raise FileNotFoundError(
            "No CSV files found. Pass --csvs or place CSVs in data/processed/."
        )

    all_results = {}
    raw = {}

    for path in csv_paths:
        X, y, class_names = load_raw(path)
        raw[path] = (X, y, class_names)

    for path in csv_paths:
        X, y, class_names = raw[path]
        label = os.path.splitext(os.path.basename(path))[0]
        result = run_cv(X, y, class_names, n_folds=n_folds, epochs=epochs, label=label)
        all_results[label] = result

    # Combined run if more than one CSV
    if len(csv_paths) > 1:
        # All CSVs must share the same feature schema
        X_all = np.concatenate([raw[p][0] for p in csv_paths])
        y_all = np.concatenate([raw[p][1] for p in csv_paths])
        # Use class names from the first file (assumed consistent)
        class_names_all = raw[csv_paths[0]][2]
        result = run_cv(
            X_all, y_all, class_names_all,
            n_folds=n_folds, epochs=epochs,
            label="combined (all clients)",
        )
        all_results["combined"] = result

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
        print(
            f"{res['label']:<30} "
            f"{s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}  "
            f"{s['macro_f1']['mean']:.4f}±{s['macro_f1']['std']:.4f}  "
            f"{auc_str}"
        )
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs", nargs="*", default=None,
        help="Paths to client CSV files. Defaults to all CSVs in data/processed/.",
    )
    parser.add_argument("--folds",  type=int, default=5,
                        help="Number of CV folds (default 5)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per fold (default 10)")
    args = parser.parse_args()
    main(csv_paths=args.csvs or [], n_folds=args.folds, epochs=args.epochs)
