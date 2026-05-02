"""
security/attacks/gradient_poisoning.py

Attack 3: Gradient Poisoning (Byzantine / Label-Flip attack)

Goal: Simulate a malicious FL client that poisons the global model by
      flipping labels on its local training data before sending updates.
      Measure how much the global model degrades when 1 of 3 clients is
      compromised.

Method:
  - Run a clean FL simulation (FedAvg, 5 rounds) as baseline.
  - Re-run with client 1 poisoned: labels flipped to a target class.
  - Compare final global model accuracy and per-class F1 between clean and
    poisoned runs.
  - Also test partial poisoning (10%, 30%, 50%, 100% of labels flipped).

Threat model (STRIDE: Tampering):
  A malicious participant (e.g. a rogue company) submits poisoned gradient
  updates to degrade the global model or cause targeted misclassification.
  This is especially relevant in healthcare FL where a competitor could
  deliberately degrade a rival's model quality.

Output:
  results/attacks/gradient_poisoning.json
"""

import json
import os
import sys
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "fl"))

from model import get_model            # noqa: E402
from data_utils import (               # noqa: E402
    ACTIVITY_NAMES, load_class_weights,
)

CLIENT_CSVS = [os.path.join(_ROOT, "data", "processed", f"client{i}.csv")
               for i in range(1, 4)]
OUT_PATH   = os.path.join(_ROOT, "results", "attacks", "gradient_poisoning.json")
INPUT_DIM  = 50
N_CLASSES  = 6
LABEL_COL  = "label"
FL_ROUNDS  = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR         = 1e-3
RANDOM_SEED = 42

# Poison config: flip source class → target class
POISON_SOURCE = None   # None = flip ALL classes
POISON_TARGET = 1      # flip to WALKING_UPSTAIRS (index 1, label 2)
POISON_RATES  = [0.0, 0.1, 0.3, 0.5, 1.0]   # fraction of labels flipped


# ── data helpers ──────────────────────────────────────────────────────────────

def load_client(csv_path: str, poison_rate: float = 0.0,
                poison_target: int = POISON_TARGET):
    """
    Load one client CSV → (train_loader, test_loader, n_train).
    If poison_rate > 0, randomly flip that fraction of training labels
    to poison_target before building the DataLoader.
    """
    df = pd.read_csv(csv_path).dropna()
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64) - 1   # 1-6 → 0-5

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # ── label flip poisoning ──────────────────────────────────────────────────
    n_poisoned = 0
    if poison_rate > 0.0:
        rng = np.random.default_rng(RANDOM_SEED)
        n_flip = int(len(y_tr) * poison_rate)
        flip_idx = rng.choice(len(y_tr), size=n_flip, replace=False)
        y_tr = y_tr.copy()
        y_tr[flip_idx] = poison_target
        n_poisoned = n_flip
    # ─────────────────────────────────────────────────────────────────────────

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_loader, test_loader, len(y_tr), n_poisoned


# ── FL simulation ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for X_b, y_b in loader:
        optimizer.zero_grad()
        criterion(model(X_b), y_b).backward()
        optimizer.step()


def get_weights(model) -> List[np.ndarray]:
    return [v.cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model, weights: List[np.ndarray]):
    sd = OrderedDict(
        (k, torch.tensor(v))
        for k, v in zip(model.state_dict().keys(), weights)
    )
    model.load_state_dict(sd, strict=True)


def fedavg(weight_list: List[List[np.ndarray]],
           sizes: List[int]) -> List[np.ndarray]:
    """Weighted average of weight arrays."""
    total = sum(sizes)
    avg = [np.zeros_like(w) for w in weight_list[0]]
    for weights, n in zip(weight_list, sizes):
        for i, w in enumerate(weights):
            avg[i] += w * (n / total)
    return avg


def evaluate_global(model, test_loaders: List[DataLoader]) -> Tuple[float, float]:
    """Evaluate global model on all clients' test sets combined."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for loader in test_loaders:
            for X_b, y_b in loader:
                preds = model(X_b).argmax(1).numpy()
                all_true.extend(y_b.numpy().tolist())
                all_pred.extend(preds.tolist())
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
    per_cls = f1_score(all_true, all_pred, average=None,
                       zero_division=0, labels=list(range(N_CLASSES)))
    return acc, f1, per_cls.tolist()


def run_fl(poison_rate: float = 0.0,
           poisoned_client_idx: int = 0) -> dict:
    """
    Run a full FL simulation with FedAvg for FL_ROUNDS rounds.
    Client at poisoned_client_idx has poison_rate fraction of labels flipped.
    Returns final accuracy, macro F1, and per-class F1.
    """
    class_weights = load_class_weights()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Load all clients — only the poisoned one gets flipped labels
    clients = []
    test_loaders = []
    for i, csv in enumerate(CLIENT_CSVS):
        rate = poison_rate if i == poisoned_client_idx else 0.0
        tr_loader, te_loader, n_tr, n_poisoned = load_client(csv, poison_rate=rate)
        clients.append((tr_loader, n_tr))
        test_loaders.append(te_loader)

    # Initialise global model
    global_model = get_model(INPUT_DIM, N_CLASSES)
    global_weights = get_weights(global_model)

    for rnd in range(1, FL_ROUNDS + 1):
        local_weights = []
        local_sizes   = []

        for tr_loader, n_tr in clients:
            local_model = get_model(INPUT_DIM, N_CLASSES)
            set_weights(local_model, global_weights)
            opt = optim.Adam(local_model.parameters(), lr=LR)
            for _ in range(LOCAL_EPOCHS):
                train_one_epoch(local_model, tr_loader, opt, criterion)
            local_weights.append(get_weights(local_model))
            local_sizes.append(n_tr)

        global_weights = fedavg(local_weights, local_sizes)
        set_weights(global_model, global_weights)

    acc, f1, per_cls = evaluate_global(global_model, test_loaders)
    return {"accuracy": round(acc, 6), "macro_f1": round(f1, 6),
            "per_class_f1": {ACTIVITY_NAMES[i]: round(v, 6)
                             for i, v in enumerate(per_cls)}}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Attack 3: Gradient Poisoning (Label-Flip)")
    print(f"  FL rounds: {FL_ROUNDS}  |  Local epochs: {LOCAL_EPOCHS}")
    print(f"  Poisoned client: Client 1 (FitLife)")
    print(f"  Flip target: class {POISON_TARGET} ({ACTIVITY_NAMES[POISON_TARGET]})")
    print("=" * 55)

    # ── baseline (clean) ──────────────────────────────────────────────────────
    print("\n[1] Running clean baseline (poison_rate=0.0)...")
    baseline = run_fl(poison_rate=0.0)
    print(f"    Accuracy : {baseline['accuracy']:.4f}")
    print(f"    Macro F1 : {baseline['macro_f1']:.4f}")

    # ── poison rate sweep ─────────────────────────────────────────────────────
    sweep_results = []
    print("\n[2] Poison rate sweep...")
    for rate in POISON_RATES:
        if rate == 0.0:
            result = baseline.copy()
            result["poison_rate"] = 0.0
            result["n_poisoned_labels"] = 0
        else:
            print(f"    poison_rate={rate:.0%}...", end=" ", flush=True)
            result = run_fl(poison_rate=rate, poisoned_client_idx=0)
            result["poison_rate"] = rate
            # approximate poisoned label count
            _, _, n_tr, n_p = load_client(CLIENT_CSVS[0], poison_rate=rate)
            result["n_poisoned_labels"] = n_p
            acc_drop = baseline["accuracy"] - result["accuracy"]
            f1_drop  = baseline["macro_f1"] - result["macro_f1"]
            print(f"acc={result['accuracy']:.4f} "
                  f"(Δ={-acc_drop:+.4f})  "
                  f"f1={result['macro_f1']:.4f} "
                  f"(Δ={-f1_drop:+.4f})")

        sweep_results.append(result)

    # ── risk assessment ───────────────────────────────────────────────────────
    full_poison = next(r for r in sweep_results if r["poison_rate"] == 1.0)
    acc_drop_full = baseline["accuracy"] - full_poison["accuracy"]
    f1_drop_full  = baseline["macro_f1"]  - full_poison["macro_f1"]

    risk = ("HIGH"   if acc_drop_full > 0.10 else
            "MEDIUM" if acc_drop_full > 0.03 else
            "LOW")

    summary = {
        "baseline_accuracy":    baseline["accuracy"],
        "baseline_macro_f1":    baseline["macro_f1"],
        "full_poison_accuracy": full_poison["accuracy"],
        "full_poison_macro_f1": full_poison["macro_f1"],
        "accuracy_drop":        round(float(acc_drop_full), 6),
        "f1_drop":              round(float(f1_drop_full), 6),
        "risk_level":           risk,
        "verdict": (
            "VULNERABLE — a single poisoned client significantly degrades the global model"
            if risk == "HIGH" else
            "MODERATE — poisoning causes measurable but limited degradation"
            if risk == "MEDIUM" else
            "RESISTANT — FedAvg dilutes the poisoned updates effectively"
        ),
        "mitigation_note": (
            "FedAvg is known to be vulnerable to Byzantine attacks. "
            "Mitigations include: robust aggregation (Krum, Trimmed Mean), "
            "anomaly detection on client updates, and DP-SGD which limits "
            "the influence of any single gradient."
        ),
    }

    output = {
        "attack":        "gradient_poisoning",
        "config": {
            "fl_rounds":          FL_ROUNDS,
            "local_epochs":       LOCAL_EPOCHS,
            "poisoned_client":    1,
            "poison_target_class": POISON_TARGET,
            "poison_target_name": ACTIVITY_NAMES[POISON_TARGET],
        },
        "sweep":   sweep_results,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Baseline  : acc={baseline['accuracy']:.4f}  f1={baseline['macro_f1']:.4f}")
    print(f"  100% flip : acc={full_poison['accuracy']:.4f}  "
          f"f1={full_poison['macro_f1']:.4f}")
    print(f"  Acc drop  : {acc_drop_full:.4f}")
    print(f"  Risk      : {risk}")
    print(f"  Verdict   : {summary['verdict']}")
    print(f"{'='*55}")
    print(f"\n✅ Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
