# privacy/epsilon_sweep.py
"""
Epsilon sweep experiment for IntelliClave — uses REAL HAR client data.

Experiment 1: Accuracy vs Epsilon (privacy-utility tradeoff)
             Trains on real client1.csv under different epsilon values

Experiment 2: Epsilon over FL Rounds
             Simulates 5 FL rounds with ε=10, tracks budget consumption

Outputs:
    results/epsilon_sweep.json
    results/epsilon_rounds.json
    results/plots/accuracy_vs_epsilon.png
    results/plots/epsilon_over_rounds.png
    results/plots/dp_summary.png
"""

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")  # suppress Opacus/PyTorch version warnings

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Resolve paths ─────────────────────────────────────────────────────────────
# Works correctly whether run as: python privacy/epsilon_sweep.py
#                              or: cd privacy && python epsilon_sweep.py
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_FL_DIR      = os.path.join(_PROJECT_ROOT, 'fl')

sys.path.insert(0, _FL_DIR)       # for M1's model.py and data_utils.py
sys.path.insert(0, _THIS_DIR)     # for dp_trainer.py

from model import get_model                            # M1's model
from data_utils import load_class_weights              # M1's class weights loader
from dp_trainer import DPTrainer                       # M2's DP wrapper
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ───────────────────────────────────────────────────────────────────
# Use client1.csv for sweep — largest active-profile dataset
CLIENT_CSV   = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'client1.csv')
LABEL_COL    = 'label'

# Sweep settings
EPSILONS     = [1.0, 2.0, 5.0, 10.0, 20.0]
SWEEP_EPOCHS = 5      # local epochs per epsilon test
BATCH_SIZE   = 64     # must be consistent — Opacus is sensitive to this

# FL rounds simulation settings
FL_ROUNDS    = 5
LOCAL_EPOCHS = 3
FIXED_EPS    = 10.0

# Output directories
os.makedirs(os.path.join(_PROJECT_ROOT, 'results'), exist_ok=True)
os.makedirs(os.path.join(_PROJECT_ROOT, 'results', 'plots'), exist_ok=True)

SWEEP_JSON  = os.path.join(_PROJECT_ROOT, 'results', 'epsilon_sweep.json')
ROUNDS_JSON = os.path.join(_PROJECT_ROOT, 'results', 'epsilon_rounds.json')
PLOT_DIR    = os.path.join(_PROJECT_ROOT, 'results', 'plots')
# ─────────────────────────────────────────────────────────────────────────────


# ── Data loader using real HAR CSV ───────────────────────────────────────────
def load_har_data(csv_path: str, batch_size: int = 64, test_split: float = 0.2):
    """
    Loads a real client CSV, applies StandardScaler (matching M1's data_utils),
    converts labels 1-6 → 0-5, returns train_loader, test_loader, n_train, n_features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n❌ CSV not found: {csv_path}"
            f"\n   Make sure M1's processed CSVs are in data/processed/"
        )

    df = pd.read_csv(csv_path).dropna()
    feature_cols = [c for c in df.columns if c != LABEL_COL]

    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64) - 1  # 1-6 → 0-5

    # Train/test split (same as M1's data_utils: 80/20, stratified)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )

    # StandardScaler — matches M1's pipeline exactly
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
        batch_size=batch_size, shuffle=False
    )

    n_train    = len(y_tr)
    n_features = X_tr.shape[1]
    print(f"  Loaded {csv_path.split(os.sep)[-1]}: "
          f"train={n_train}, test={len(y_te)}, features={n_features}")
    return train_loader, test_loader, n_train, n_features


def evaluate(model, loader):
    """Compute accuracy on a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch).argmax(1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return correct / total if total > 0 else 0.0


# ── Load class weights from M1 ────────────────────────────────────────────────
def get_criterion():
    weights = load_class_weights()  # returns tensor or None
    return nn.CrossEntropyLoss(weight=weights)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Accuracy vs Epsilon on REAL HAR data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("EXPERIMENT 1: Accuracy vs Epsilon")
print("Using: Real HAR client1.csv data")
print("=" * 55)

results = []

for eps in EPSILONS:
    print(f"\n▶ Testing ε = {eps}...")

    try:
        train_loader, test_loader, n_train, n_features = load_har_data(
            CLIENT_CSV, batch_size=BATCH_SIZE
        )
        delta = 1.0 / n_train  # standard DP rule: δ = 1/n

        model = get_model(input_dim=n_features, num_classes=6)

        # Use RDP accountant — more stable than PRV on real-sized datasets
        # DPTrainer wraps Opacus; accountant type controlled by Opacus internally
        trainer = DPTrainer(
            model=model,
            target_epsilon=eps,
            target_delta=delta,
            epochs=SWEEP_EPOCHS
        )
        trainer.attach(train_loader)

        for epoch in range(SWEEP_EPOCHS):
            metrics = trainer.train_one_round()

        acc_train = evaluate(trainer.get_model(), train_loader)
        acc_test  = evaluate(trainer.get_model(), test_loader)
        actual_eps = trainer.get_epsilon()

        results.append({
            "target_epsilon":  eps,
            "actual_epsilon":  round(actual_eps, 4),
            "train_accuracy":  round(acc_train, 4),
            "test_accuracy":   round(acc_test, 4),
            "delta":           round(delta, 8),
            "n_train":         n_train
        })
        print(f"  ✅ train_acc={acc_train:.4f} | test_acc={acc_test:.4f} "
              f"| actual_ε={actual_eps:.4f} | δ={delta:.2e}")

    except RuntimeError as e:
        print(f"  ⚠️  Skipped ε={eps}: {e}")
        results.append({
            "target_epsilon": eps,
            "actual_epsilon": None,
            "train_accuracy": None,
            "test_accuracy":  None,
            "error": str(e)
        })

with open(SWEEP_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved → {SWEEP_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Epsilon over FL Rounds on REAL HAR data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("EXPERIMENT 2: Epsilon over FL Rounds")
print(f"Using: Real HAR client1.csv | ε={FIXED_EPS} | {FL_ROUNDS} rounds × {LOCAL_EPOCHS} epochs")
print("=" * 55)

rounds_data = []

try:
    train_loader, test_loader, n_train, n_features = load_har_data(
        CLIENT_CSV, batch_size=BATCH_SIZE
    )
    delta = 1.0 / n_train
    total_epochs = LOCAL_EPOCHS * FL_ROUNDS   # ← correct fix: tell Opacus full duration

    model = get_model(input_dim=n_features, num_classes=6)

    trainer_r = DPTrainer(
        model=model,
        target_epsilon=FIXED_EPS,
        target_delta=delta,
        epochs=total_epochs   # 3 local × 5 rounds = 15 total
    )
    trainer_r.attach(train_loader)

    print(f"  PrivacyEngine attached: total_epochs={total_epochs} "
          f"(δ={delta:.2e})\n")

    for round_num in range(1, FL_ROUNDS + 1):
        # Run LOCAL_EPOCHS training passes (simulating one FL round)
        for _ in range(LOCAL_EPOCHS):
            trainer_r.train_one_round()

        eps_consumed    = trainer_r.get_epsilon()
        acc_train       = evaluate(trainer_r.get_model(), train_loader)
        acc_test        = evaluate(trainer_r.get_model(), test_loader)
        budget_remaining = FIXED_EPS - eps_consumed

        rounds_data.append({
            "fl_round":        round_num,
            "epsilon_consumed": round(float(eps_consumed), 4),
            "budget_remaining": round(float(budget_remaining), 4),
            "budget_exhausted": bool(eps_consumed >= FIXED_EPS),  # fix: cast to Python bool
            "train_accuracy":  round(float(acc_train), 4),
            "test_accuracy":   round(float(acc_test), 4)
        })
        status = "⚠️  EXHAUSTED" if eps_consumed >= FIXED_EPS else "✅"
        print(f"  Round {round_num}: ε={eps_consumed:.4f} | "
              f"test_acc={acc_test:.4f} | remaining={budget_remaining:.4f} {status}")

except RuntimeError as e:
    print(f"  ❌ Experiment 2 failed: {e}")
    rounds_data = []

with open(ROUNDS_JSON, "w") as f:
    json.dump(rounds_data, f, indent=2)
print(f"\n✅ Saved → {ROUNDS_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("Generating plots...")
print("=" * 55)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Plot 1: Accuracy vs Epsilon ───────────────────────────────────────────
    valid = [r for r in results if r.get("test_accuracy") is not None]

    if valid:
        eps_vals      = [r["target_epsilon"] for r in valid]
        acc_test_vals = [r["test_accuracy"] * 100 for r in valid]
        acc_train_vals= [r["train_accuracy"] * 100 for r in valid]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(eps_vals, acc_test_vals, marker='o', linewidth=2.5,
                color='#2E75B6', markersize=9, markerfacecolor='white',
                markeredgewidth=2.5, label="Test Accuracy (DP)")
        ax.plot(eps_vals, acc_train_vals, marker='s', linewidth=1.8,
                color='#1D9E75', markersize=7, markerfacecolor='white',
                markeredgewidth=2, linestyle='--', label="Train Accuracy (DP)")

        # Shaded zones
        max_x = max(eps_vals) + 2
        ax.axvspan(0,   3,   alpha=0.07, color='red',    label="High privacy zone (ε < 3)")
        ax.axvspan(3,   12,  alpha=0.07, color='orange', label="Balanced zone (3–12)")
        ax.axvspan(12,  max_x, alpha=0.07, color='green', label="Low privacy zone (ε > 12)")

        ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.5,
                   label="Our target ε = 10")

        for x, y_t in zip(eps_vals, acc_test_vals):
            ax.annotate(f"{y_t:.1f}%", (x, y_t),
                        textcoords="offset points", xytext=(0, 11),
                        ha='center', fontsize=9, color='#2E75B6', fontweight='bold')

        ax.set_xlabel("Target Epsilon (ε)  —  Lower = Stronger Privacy", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Privacy-Utility Tradeoff — Real HAR Data\n"
                     "IntelliClave FL+DP (client1.csv, UCI HAR)", fontsize=13, fontweight='bold')
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        fig.tight_layout()
        path = os.path.join(PLOT_DIR, 'accuracy_vs_epsilon.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")

    # ── Plot 2: Epsilon over FL Rounds ────────────────────────────────────────
    if rounds_data:
        round_nums    = [r["fl_round"] for r in rounds_data]
        eps_consumed  = [r["epsilon_consumed"] for r in rounds_data]
        acc_test_r    = [r["test_accuracy"] * 100 for r in rounds_data]
        remaining     = [r["budget_remaining"] for r in rounds_data]

        fig, ax1 = plt.subplots(figsize=(9, 5))
        color_eps = '#D85A30'
        color_acc = '#2E75B6'
        color_rem = '#1D9E75'

        # Epsilon line
        ax1.plot(round_nums, eps_consumed, marker='s', linewidth=2.5,
                 color=color_eps, markersize=9, markerfacecolor='white',
                 markeredgewidth=2.5, label="ε consumed (cumulative)")
        ax1.fill_between(round_nums, eps_consumed, alpha=0.1, color=color_eps)

        # Budget remaining as dashed line
        ax1.plot(round_nums, remaining, marker='^', linewidth=1.8,
                 color=color_rem, markersize=7, markerfacecolor='white',
                 markeredgewidth=2, linestyle=':', label="Budget remaining")

        # Budget limit line
        ax1.axhline(y=FIXED_EPS, color='red', linestyle='--',
                    linewidth=1.8, label=f"Budget limit ε = {FIXED_EPS}")

        for x, e in zip(round_nums, eps_consumed):
            ax1.annotate(f"{e:.2f}", (x, e),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=8, color=color_eps)

        ax1.set_xlabel("FL Round", fontsize=12)
        ax1.set_ylabel("Epsilon (ε)", fontsize=12, color=color_eps)
        ax1.tick_params(axis='y', labelcolor=color_eps)
        ax1.set_ylim(0, FIXED_EPS * 1.4)
        ax1.set_xticks(round_nums)

        # Accuracy on right axis
        ax2 = ax1.twinx()
        ax2.plot(round_nums, acc_test_r, marker='o', linewidth=2.5,
                 color=color_acc, markersize=9, markerfacecolor='white',
                 markeredgewidth=2.5, linestyle='--', label="Test Accuracy (%)")
        ax2.set_ylabel("Test Accuracy (%)", fontsize=12, color=color_acc)
        ax2.tick_params(axis='y', labelcolor=color_acc)
        ax2.set_ylim(0, 105)

        ax1.set_title(f"Privacy Budget Consumption Across FL Rounds\n"
                      f"Real HAR data | ε target={FIXED_EPS} | "
                      f"{FL_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs",
                      fontsize=13, fontweight='bold')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#FAFAFA')
        fig.tight_layout()
        path = os.path.join(PLOT_DIR, 'epsilon_over_rounds.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")

    # ── Plot 3: Combined Summary ──────────────────────────────────────────────
    if valid and rounds_data:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 5))

        # Left panel
        ax_l.plot(eps_vals, acc_test_vals, marker='o', linewidth=2.5,
                  color='#2E75B6', markersize=8, markerfacecolor='white',
                  markeredgewidth=2.5, label="Test Accuracy")
        ax_l.plot(eps_vals, acc_train_vals, marker='s', linewidth=1.5,
                  color='#1D9E75', markersize=6, markerfacecolor='white',
                  markeredgewidth=2, linestyle='--', label="Train Accuracy")
        ax_l.axvline(x=10, color='gray', linestyle='--', linewidth=1.5,
                     label="Our ε = 10")
        ax_l.axvspan(0,  3,  alpha=0.07, color='red')
        ax_l.axvspan(3,  12, alpha=0.07, color='orange')
        ax_l.axvspan(12, max(eps_vals)+2, alpha=0.07, color='green')
        for x, y_t in zip(eps_vals, acc_test_vals):
            ax_l.annotate(f"{y_t:.1f}%", (x, y_t),
                          textcoords="offset points", xytext=(0, 8),
                          ha='center', fontsize=8)
        ax_l.set_xlabel("Target Epsilon (ε)", fontsize=11)
        ax_l.set_ylabel("Accuracy (%)", fontsize=11)
        ax_l.set_title("Privacy-Utility Tradeoff\n(Real HAR Data)", fontsize=12, fontweight='bold')
        ax_l.set_ylim(0, 105)
        ax_l.grid(True, alpha=0.3)
        ax_l.legend(fontsize=8)

        # Right panel
        ax_r.bar(round_nums, eps_consumed, color='#D85A30', alpha=0.65,
                 label="ε consumed", zorder=2)
        ax_r.axhline(y=FIXED_EPS, color='red', linestyle='--',
                     linewidth=1.8, label=f"Budget ε={FIXED_EPS}")
        for x, e in zip(round_nums, eps_consumed):
            ax_r.text(x, e + 0.1, f"{e:.2f}", ha='center', fontsize=8,
                      color='#D85A30', fontweight='bold')
        ax_r2 = ax_r.twinx()
        ax_r2.plot(round_nums, acc_test_r, marker='o', linewidth=2.2,
                   color='#2E75B6', markersize=7, markerfacecolor='white',
                   markeredgewidth=2, label="Test Accuracy")
        ax_r.set_xlabel("FL Round", fontsize=11)
        ax_r.set_ylabel("ε consumed", fontsize=11, color='#D85A30')
        ax_r2.set_ylabel("Test Accuracy (%)", fontsize=11, color='#2E75B6')
        ax_r.set_title(f"Budget Consumption per Round\n(ε target={FIXED_EPS})",
                       fontsize=12, fontweight='bold')
        ax_r.set_ylim(0, FIXED_EPS * 1.4)
        ax_r2.set_ylim(0, 105)
        ax_r.set_xticks(round_nums)
        ax_r.grid(True, alpha=0.3, axis='y')
        lines_r,  lbl_r  = ax_r.get_legend_handles_labels()
        lines_r2, lbl_r2 = ax_r2.get_legend_handles_labels()
        ax_r.legend(lines_r + lines_r2, lbl_r + lbl_r2, fontsize=8)

        fig.suptitle("IntelliClave — Differential Privacy Evaluation\n"
                     "UCI HAR Dataset | Federated Learning + Opacus DP",
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        path = os.path.join(PLOT_DIR, 'dp_summary.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {path}")

    print(f"\nAll plots saved to: {PLOT_DIR}")

except ImportError:
    print("⚠️  matplotlib not installed. Run: pip install matplotlib")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("RESULTS SUMMARY")
print("=" * 55)

print(f"\n{'Target ε':>10} | {'Actual ε':>10} | {'Test Acc':>10} | {'Train Acc':>10} | δ")
print("-" * 62)
for r in results:
    if r.get("test_accuracy") is not None:
        print(f"{r['target_epsilon']:>10} | {r['actual_epsilon']:>10.4f} | "
              f"{r['test_accuracy']*100:>9.2f}% | "
              f"{r['train_accuracy']*100:>9.2f}% | "
              f"{r.get('delta', 0):.2e}")
    else:
        print(f"{r['target_epsilon']:>10} | {'skipped':>10} | {'—':>10} | {'—':>10}")

if rounds_data:
    print(f"\n{'Round':>7} | {'ε consumed':>12} | {'Remaining':>10} | {'Test Acc':>10} | Exhausted")
    print("-" * 62)
    for r in rounds_data:
        print(f"{r['fl_round']:>7} | {r['epsilon_consumed']:>12.4f} | "
              f"{r['budget_remaining']:>10.4f} | "
              f"{r['test_accuracy']*100:>9.2f}% | "
              f"{r['budget_exhausted']}")