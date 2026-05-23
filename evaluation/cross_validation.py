"""
evaluation/cross_validation.py

Stratified k-fold cross-validation for the IntelliClave classifier.

Two modes (Issue 8 fix)
-----------------------
--mode centralized (default, existing behaviour)
    Runs standard stratified k-fold CV on each client's data independently,
    then on all clients combined. The combined run shuffles data across client
    boundaries — this is a CENTRALIZED BASELINE, not a federated metric.
    It measures what accuracy a single model trained on all data could achieve.
    Clearly labelled as such in output and printed summary.

--mode federated
    Federated cross-validation that respects client boundaries.
    Each fold uses one client as the validation set and trains on the
    remaining clients. This measures how well the global model generalises
    to a held-out client — the correct metric for federated learning.

    With 3 clients this gives 3-fold leave-one-client-out CV:
      Fold 1: train on clients 2+3, validate on client 1
      Fold 2: train on clients 1+3, validate on client 2
      Fold 3: train on clients 1+2, validate on client 3

    This is the federated analogue of cross-validation and directly measures
    the generalisation gap between clients — the key non-IID diagnostic.

Usage:
    # Centralized baseline (default)
    python evaluation/cross_validation.py

    # Federated leave-one-client-out CV
    python evaluation/cross_validation.py --mode federated

    # Both modes (centralized + federated reported separately — not compared)
    python evaluation/cross_validation.py --mode both

    # Tune folds / epochs
    python evaluation/cross_validation.py --folds 10 --epochs 15

Outputs:
    results/cross_validation.json
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


# ── Federated leave-one-client-out CV (Issue 8 fix) ──────────────────────────

def run_federated_cv(
    csv_paths: list,
    epochs: int = 10,
) -> dict:
    """
    Federated leave-one-client-out cross-validation.

    Each fold holds out one client as the validation set and trains on
    the remaining clients. This respects client data boundaries — no
    data from the validation client is seen during training.

    This is the correct federated CV metric. It measures:
      - How well the global model generalises to an unseen client
      - The per-client generalisation gap (non-IID diagnostic)
      - Whether any client is systematically underserved by the global model

    Parameters
    ----------
    csv_paths : list of paths to client CSVs (one per client)
    epochs    : training epochs per fold

    Returns
    -------
    dict with per-fold metrics and summary
    """
    n_clients = len(csv_paths)
    if n_clients < 2:
        raise ValueError(
            "Federated CV requires at least 2 clients. "
            "With 1 client there is no held-out client to validate on."
        )

    print(f"\n{'='*60}")
    print(f"Federated Leave-One-Client-Out CV  ({n_clients} clients)")
    print(f"  Each fold: train on {n_clients-1} clients, validate on 1")
    print(f"  Epochs per fold: {epochs}")
    if n_clients < 5:
        print(
            f"  WARNING: Only {n_clients} folds — high variance; confidence intervals "
            f"are wide. Regenerate data with >= 5 clients for stable estimates:\n"
            f"    python data/datascripts/pipeline.py --n-clients 5"
        )
    print(f"{'='*60}")

    # Load all client data
    all_data = []
    for path in csv_paths:
        X, y, class_names = load_raw(path)
        all_data.append((X, y, os.path.splitext(os.path.basename(path))[0]))

    n_classes = len(class_names)
    input_dim = all_data[0][0].shape[1]
    folds = []

    for val_idx in range(n_clients):
        val_name = all_data[val_idx][2]
        train_names = [all_data[i][2] for i in range(n_clients) if i != val_idx]

        print(f"\n  Fold {val_idx+1}/{n_clients}: "
              f"train={train_names}  val={val_name}")

        # Combine training clients
        X_tr = np.concatenate([all_data[i][0] for i in range(n_clients) if i != val_idx])
        y_tr = np.concatenate([all_data[i][1] for i in range(n_clients) if i != val_idx])
        X_val, y_val = all_data[val_idx][0], all_data[val_idx][1]

        # Scale: fit on training clients only — no leakage from val client
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
        metrics["fold"]             = val_idx + 1
        metrics["val_client"]       = val_name
        metrics["train_clients"]    = train_names
        metrics["val_n_samples"]    = len(y_val)
        metrics["train_n_samples"]  = len(y_tr)
        folds.append(metrics)

        print(f"    acc={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"auc={metrics['auc_roc']}")

    accs = [f["accuracy"]  for f in folds]
    f1s  = [f["macro_f1"]  for f in folds]
    aucs = [f["auc_roc"]   for f in folds if f["auc_roc"] is not None]

    # Generalisation gap: difference between best and worst client accuracy
    gen_gap = round(max(accs) - min(accs), 6)

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
        },
        "auc_roc": {
            "mean": round(np.mean(aucs), 6),
            "std":  round(np.std(aucs),  6),
        } if aucs else None,
        "generalisation_gap": gen_gap,
        "generalisation_gap_note": (
            "Max - min accuracy across held-out clients. "
            "High gap indicates the global model underserves some clients."
        ),
    }

    print(f"\n  Federated CV Summary ({n_clients}-fold leave-one-client-out):")
    print(f"    Accuracy : {summary['accuracy']['mean']:.4f} "
          f"+/- {summary['accuracy']['std']:.4f}")
    print(f"    Macro F1 : {summary['macro_f1']['mean']:.4f}")
    print(f"    Gen gap  : {gen_gap:.4f}  "
          f"(max-min accuracy across clients)")
    if gen_gap > 0.05:
        print(f"    WARNING: generalisation gap {gen_gap:.4f} > 0.05 — "
              "global model underserves at least one client.")

    return {
        "label":      "federated (leave-one-client-out)",
        "mode":       "federated",
        "n_clients":  n_clients,
        "n_folds":    n_clients,
        "epochs":     epochs,
        "folds":      folds,
        "summary":    summary,
        "low_fold_warning": (
            f"Only {n_clients} LOOCV folds — high variance; use >= 5 clients "
            f"for stabler generalisation estimates."
            if n_clients < 5 else None
        ),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(csv_paths: list, n_folds: int = 5, epochs: int = 10, mode: str = "centralized"):
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

    run_centralized = mode in ("centralized", "both")
    run_federated   = mode in ("federated",   "both")

    # ── Centralized CV ────────────────────────────────────────────────────────
    if run_centralized:
        print("\n" + "=" * 60)
        print("MODE: CENTRALIZED BASELINE")
        print("  Folds shuffle data within each client independently.")
        print("  The 'combined' run shuffles across ALL client boundaries.")
        print("  This is NOT a federated metric — it is a centralized upper")
        print("  bound showing what a single model on all data could achieve.")
        print("=" * 60)

        for path in csv_paths:
            X, y, class_names = raw[path]
            label = os.path.splitext(os.path.basename(path))[0]
            result = run_cv(X, y, class_names, n_folds=n_folds,
                            epochs=epochs, label=label)
            result["mode"] = "centralized_per_client"
            all_results[label] = result

        if len(csv_paths) > 1:
            X_all = np.concatenate([raw[p][0] for p in csv_paths])
            y_all = np.concatenate([raw[p][1] for p in csv_paths])
            class_names_all = raw[csv_paths[0]][2]
            result = run_cv(
                X_all, y_all, class_names_all,
                n_folds=n_folds, epochs=epochs,
                label="combined — CENTRALIZED BASELINE (shuffles across clients)",
            )
            result["mode"] = "centralized_combined"
            result["centralized_baseline_note"] = (
                "This run concatenates all client data and shuffles across "
                "client boundaries. It measures centralized model performance, "
                "NOT federated performance. Use --mode federated for the "
                "correct federated metric."
            )
            all_results["combined_centralized"] = result

    # ── Federated CV ──────────────────────────────────────────────────────────
    if run_federated and len(csv_paths) >= 2:
        result = run_federated_cv(csv_paths, epochs=epochs)
        all_results["federated_loocv"] = result
    elif run_federated and len(csv_paths) < 2:
        print("\nWARNING: Federated CV requires >= 2 client CSVs. Skipping.")

    # ── Statistical fairness metadata (--mode both) ─────────────────────────
    n_clients_cv = len(csv_paths)
    if mode == "both":
        fairness = {
            "direct_comparison_suppressed": True,
            "reason": (
                "Centralized k-fold and federated leave-one-client-out use "
                "different fold counts and validation units — side-by-side "
                "accuracy numbers are not statistically comparable."
            ),
            "centralized_folds": n_folds,
            "federated_folds":   n_clients_cv,
            "effective_sample_size_note": (
                f"Federated LOOCV with {n_clients_cv} clients yields only "
                f"{n_clients_cv} folds — variance is high and confidence "
                "intervals are wide. Use >= 5 clients for meaningful "
                "federated generalisation estimates."
            ),
            "recommendation": (
                "Report centralized and federated blocks separately. "
                "Do not rank one against the other."
            ),
        }
        all_results["_evaluation_metadata"] = fairness
        print("\n" + "=" * 60)
        print("STATISTICAL FAIRNESS (--mode both)")
        print(f"  {fairness['reason']}")
        print(f"  {fairness['effective_sample_size_note']}")
        print("  Direct comparison table SUPPRESSED — see per-mode sections above.")
        print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(_ROOT, "results", "cross_validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    # ── Final table (per mode — never mix centralized vs federated) ─────────
    print(f"\n{'='*60}")
    if mode == "both":
        for block_mode, title in (
            ("centralized", "CENTRALIZED BASELINE (k-fold)"),
            ("federated",   "FEDERATED LOOCV (client-held-out)"),
        ):
            block = {
                k: v for k, v in all_results.items()
                if k != "_evaluation_metadata"
                and (
                    (block_mode == "centralized" and v.get("mode", "").startswith("centralized"))
                    or (block_mode == "federated" and v.get("mode") == "federated")
                )
            }
            if not block:
                continue
            print(f"\n{title}")
            print(f"{'Dataset':<38} {'Accuracy':>10} {'Macro F1':>10}")
            print(f"{'-'*60}")
            for _key, res in block.items():
                s = res["summary"]
                print(
                    f"{res['label'][:38]:<38} "
                    f"{s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}  "
                    f"{s['macro_f1']['mean']:.4f}"
                )
                if "generalisation_gap" in s:
                    print(f"  {'':38} gen_gap={s['generalisation_gap']:.4f}")
        print(f"\n{'='*60}")
    else:
        print(f"{'Dataset':<38} {'Accuracy':>10} {'Macro F1':>10} {'Mode'}")
        print(f"{'-'*60}")
        for key, res in all_results.items():
            if key == "_evaluation_metadata":
                continue
            s = res["summary"]
            m = res.get("mode", "")
            print(
                f"{res['label'][:38]:<38} "
                f"{s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}  "
                f"{s['macro_f1']['mean']:.4f}  "
                f"{m}"
            )
            if "generalisation_gap" in s:
                print(f"  {'':38} gen_gap={s['generalisation_gap']:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-validation for IntelliClave FL classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csvs", nargs="*", default=None,
        help="Paths to client CSV files. Defaults to all CSVs in data/processed/.",
    )
    parser.add_argument("--folds",  type=int, default=5,
                        help="Number of CV folds for centralized mode (default 5)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per fold (default 10)")
    parser.add_argument(
        "--mode", default="centralized",
        choices=["centralized", "federated", "both"],
        help=(
            "centralized: standard k-fold CV (default). "
            "federated: leave-one-client-out CV respecting client boundaries. "
            "both: run both modes (no direct accuracy comparison — different fold counts)."
        ),
    )
    args = parser.parse_args()
    main(csv_paths=args.csvs or [], n_folds=args.folds,
         epochs=args.epochs, mode=args.mode)
