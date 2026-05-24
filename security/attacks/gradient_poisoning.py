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
    load_class_weights, infer_csv_schema,
)
import json as _json

sys.path.insert(0, os.path.join(_ROOT, "config"))
from constants import (                # noqa: E402
    DEFAULT_N_CLIENTS, LABEL_COL, RANDOM_SEED,
)

PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
META_PATH     = os.path.join(_ROOT, "results", "fl_rounds", "model_meta.json")
CLIENT_CSVS   = sorted(
    os.path.join(PROCESSED_DIR, f)
    for f in os.listdir(PROCESSED_DIR)
    if f.endswith(".csv")
) if os.path.isdir(PROCESSED_DIR) else []
OUT_PATH   = os.path.join(_ROOT, "results", "attacks", "gradient_poisoning.json")
FL_ROUNDS  = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR         = 1e-3

POISON_TARGET = 1      # flip to class index 1
POISON_RATES  = [0.0, 0.1, 0.3, 0.5, 1.0]


def _load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return _json.load(f)
    if not CLIENT_CSVS:
        raise FileNotFoundError("No model_meta.json and no CSVs in data/processed/")
    input_dim, _ = infer_csv_schema(CLIENT_CSVS[0], LABEL_COL)
    df = pd.read_csv(CLIENT_CSVS[0], usecols=[LABEL_COL])
    num_classes = int(df[LABEL_COL].nunique())
    return {"input_dim": input_dim, "num_classes": num_classes,
            "class_names": [f"class_{i}" for i in range(num_classes)]}

_META = _load_meta()
INPUT_DIM  = _META["input_dim"]
N_CLASSES  = _META["num_classes"]
CLASS_NAMES = _META["class_names"]


# ── data helpers ──────────────────────────────────────────────────────────────

def load_client_csv(csv_path: str, poison_rate: float = 0.0,
                poison_target: int = POISON_TARGET,
                batch_size: int = BATCH_SIZE):
    """
    Load one client CSV → (train_loader, test_loader, n_train, n_poisoned).
    Labels are auto-shifted to 0-based regardless of original range.
    If poison_rate > 0, randomly flip that fraction of training labels.
    """
    df = pd.read_csv(csv_path).dropna()
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols].values.astype(np.float32)
    raw_y = df[LABEL_COL].values

    # Auto-detect label offset — works for 0-based, 1-based, or any range
    unique = sorted(np.unique(raw_y).tolist())
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique):
        offset = int(min(unique))
        y = raw_y.astype(np.int64) - offset
    else:
        label_map = {v: i for i, v in enumerate(unique)}
        y = np.array([label_map[v] for v in raw_y], dtype=np.int64)

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
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
        batch_size=batch_size, shuffle=False,
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
           poisoned_client_idx: int = 0,
           fl_rounds: int = FL_ROUNDS,
           local_epochs: int = LOCAL_EPOCHS,
           batch_size: int = BATCH_SIZE,
           lr: float = LR,
           poison_target: int = POISON_TARGET) -> dict:
    """
    Run a full FL simulation with FedAvg for fl_rounds rounds.
    Client at poisoned_client_idx has poison_rate fraction of labels flipped.
    Returns final accuracy, macro F1, and per-class F1.
    """
    class_weights = load_class_weights(num_classes=N_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Load all clients — only the poisoned one gets flipped labels
    clients = []
    test_loaders = []
    for i, csv in enumerate(CLIENT_CSVS):
        rate = poison_rate if i == poisoned_client_idx else 0.0
        tr_loader, te_loader, n_tr, n_poisoned = load_client_csv(
            csv, poison_rate=rate, poison_target=poison_target,
            batch_size=batch_size,
        )
        clients.append((tr_loader, n_tr))
        test_loaders.append(te_loader)

    # Initialise global model
    global_model = get_model(INPUT_DIM, N_CLASSES)
    global_weights = get_weights(global_model)

    for rnd in range(1, fl_rounds + 1):
        local_weights = []
        local_sizes   = []

        for tr_loader, n_tr in clients:
            local_model = get_model(INPUT_DIM, N_CLASSES)
            set_weights(local_model, global_weights)
            opt = optim.Adam(local_model.parameters(), lr=lr)
            for _ in range(local_epochs):
                train_one_epoch(local_model, tr_loader, opt, criterion)
            local_weights.append(get_weights(local_model))
            local_sizes.append(n_tr)

        global_weights = fedavg(local_weights, local_sizes)
        set_weights(global_model, global_weights)

    acc, f1, per_cls = evaluate_global(global_model, test_loaders)
    return {"accuracy": round(acc, 6), "macro_f1": round(f1, 6),
            "per_class_f1": {CLASS_NAMES[i]: round(v, 6)
                             for i, v in enumerate(per_cls)}}


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    fl_rounds: int = FL_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    poison_target: int = POISON_TARGET,
    poison_rates: list = None,
    poisoned_client_idx: int = 0,
    out_path: str = OUT_PATH,
):
    if poison_rates is None:
        poison_rates = POISON_RATES

    print("=" * 55)
    print("Attack 3: Gradient Poisoning (Label-Flip)")
    print(f"  FL rounds: {fl_rounds}  |  Local epochs: {local_epochs}")
    print(f"  Poisoned client: Client {poisoned_client_idx + 1}")
    print(f"  Flip target: class {poison_target} ({CLASS_NAMES[poison_target] if poison_target < len(CLASS_NAMES) else poison_target})")
    print("=" * 55)

    # ── baseline (clean) ──────────────────────────────────────────────────────
    print("\n[1] Running clean baseline (poison_rate=0.0)...")
    baseline = run_fl(
        poison_rate=0.0,
        fl_rounds=fl_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        poison_target=poison_target,
    )
    print(f"    Accuracy : {baseline['accuracy']:.4f}")
    print(f"    Macro F1 : {baseline['macro_f1']:.4f}")

    # ── poison rate sweep ─────────────────────────────────────────────────────
    sweep_results = []
    print("\n[2] Poison rate sweep...")
    for rate in poison_rates:
        if rate == 0.0:
            result = baseline.copy()
            result["poison_rate"] = 0.0
            result["n_poisoned_labels"] = 0
        else:
            print(f"    poison_rate={rate:.0%}...", end=" ", flush=True)
            result = run_fl(
                poison_rate=rate,
                poisoned_client_idx=poisoned_client_idx,
                fl_rounds=fl_rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                poison_target=poison_target,
            )
            result["poison_rate"] = rate
            # approximate poisoned label count
            _, _, n_tr, n_p = load_client_csv(
                CLIENT_CSVS[poisoned_client_idx],
                poison_rate=rate,
                poison_target=poison_target,
            )
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
            "fl_rounds":          fl_rounds,
            "local_epochs":       local_epochs,
            "poisoned_client":    poisoned_client_idx + 1,
            "poison_target_class": poison_target,
            "poison_target_name": CLASS_NAMES[poison_target] if poison_target < len(CLASS_NAMES) else str(poison_target),
        },
        "sweep":   sweep_results,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Baseline  : acc={baseline['accuracy']:.4f}  f1={baseline['macro_f1']:.4f}")
    print(f"  100% flip : acc={full_poison['accuracy']:.4f}  "
          f"f1={full_poison['macro_f1']:.4f}")
    print(f"  Acc drop  : {acc_drop_full:.4f}")
    print(f"  Risk      : {risk}")
    print(f"  Verdict   : {summary['verdict']}")
    print(f"{'='*55}")
    print(f"\n✅ Saved → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Gradient Poisoning (Label-Flip) attack simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fl-rounds",    type=int,   default=FL_ROUNDS,
                        help="Number of FL rounds per simulation.")
    parser.add_argument("--local-epochs", type=int,   default=LOCAL_EPOCHS,
                        help="Local training epochs per client per round.")
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE,
                        help="DataLoader batch size.")
    parser.add_argument("--lr",           type=float, default=LR,
                        help="Client learning rate.")
    parser.add_argument("--poison-target",type=int,   default=POISON_TARGET,
                        help="Class index to flip labels to.")
    parser.add_argument("--poison-rates", nargs="+",  type=float,
                        default=POISON_RATES,
                        help="Poison rate values to sweep (e.g. 0.0 0.1 0.5 1.0).")
    parser.add_argument("--poisoned-client", type=int, default=0,
                        help="0-based index of the client to poison.")
    parser.add_argument("--out", default=OUT_PATH,
                        help="Output JSON path.")
    args = parser.parse_args()
    main(
        fl_rounds=args.fl_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        poison_target=args.poison_target,
        poison_rates=args.poison_rates,
        poisoned_client_idx=args.poisoned_client,
        out_path=args.out,
    )
