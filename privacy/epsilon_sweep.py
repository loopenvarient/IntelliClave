# privacy/epsilon_sweep.py
"""
Epsilon sweep experiment for IntelliClave — privacy-utility tradeoff.

Experiment 1: Accuracy vs Epsilon
             Trains on a client CSV under different epsilon values.

Experiment 2: Epsilon over FL Rounds
             Simulates FL rounds with a fixed epsilon, tracks budget consumption.

Usage:
    python privacy/epsilon_sweep.py                          # both experiments, defaults
    python privacy/epsilon_sweep.py --exp 1                  # only experiment 1
    python privacy/epsilon_sweep.py --exp 2                  # only experiment 2
    python privacy/epsilon_sweep.py --csv data/processed/client2.csv
    python privacy/epsilon_sweep.py --epsilons 1 5 10 20
    python privacy/epsilon_sweep.py --fl-rounds 10 --local-epochs 5
    python privacy/epsilon_sweep.py --no-plots               # skip matplotlib

Outputs:
    results/epsilon_sweep.json
    results/epsilon_rounds.json
    results/plots/accuracy_vs_epsilon.png
    results/plots/epsilon_over_rounds.png
    results/plots/dp_summary.png
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Resolve paths ─────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_FL_DIR       = os.path.join(_PROJECT_ROOT, "fl")

sys.path.insert(0, _FL_DIR)
sys.path.insert(0, _THIS_DIR)

from model import get_model          # FLClassifier  # noqa: E402
from data_utils import load_class_weights  # noqa: E402
from dp_trainer import DPTrainer     # noqa: E402
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_client_data(
    csv_path: str,
    label_col: str = "label",
    batch_size: int = 64,
    test_split: float = 0.2,
):
    """
    Load any client CSV, auto-detect label encoding, scale features.
    Scaler is fitted on the training split only (no leakage).
    Returns train_loader, test_loader, n_train, n_features, n_classes.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Make sure processed CSVs are in data/processed/"
        )

    df = pd.read_csv(csv_path).dropna()
    feature_cols = [c for c in df.columns if c != label_col]

    X     = df[feature_cols].values.astype(np.float32)
    raw_y = df[label_col].values

    unique = sorted(np.unique(raw_y).tolist(), key=str)
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique):
        offset = int(min(unique))
        y = raw_y.astype(np.int64) - offset
    else:
        label_map = {v: i for i, v in enumerate(unique)}
        y = np.array([label_map[v] for v in raw_y], dtype=np.int64)

    n_classes = len(unique)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
        batch_size=batch_size, shuffle=False,
    )

    n_train    = len(y_tr)
    n_features = X_tr.shape[1]
    print(f"  Loaded {os.path.basename(csv_path)}: "
          f"train={n_train}, test={len(y_te)}, "
          f"features={n_features}, classes={n_classes}")
    return train_loader, test_loader, n_train, n_features, n_classes


def evaluate_acc(model, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in loader:
            preds    = model(X_b).argmax(1)
            correct += (preds == y_b).sum().item()
            total   += len(y_b)
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Accuracy vs Epsilon
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment1(
    csv_path: str,
    epsilons: list,
    sweep_epochs: int,
    batch_size: int,
    out_path: str,
) -> list:
    print("\n" + "=" * 55)
    print("EXPERIMENT 1: Accuracy vs Epsilon")
    print(f"  CSV    : {os.path.basename(csv_path)}")
    print(f"  ε list : {epsilons}")
    print(f"  epochs : {sweep_epochs}")
    print("=" * 55)

    results = []

    for eps in epsilons:
        print(f"\n▶ Testing ε = {eps}...")
        try:
            train_loader, test_loader, n_train, n_features, n_classes = load_client_data(
                csv_path, batch_size=batch_size
            )
            delta = 1.0 / n_train
            model = get_model(input_dim=n_features, num_classes=n_classes)

            trainer = DPTrainer(
                model=model,
                target_epsilon=eps,
                target_delta=delta,
                epochs=sweep_epochs,
                num_classes=n_classes,
            )
            trainer.attach(train_loader)

            for _ in range(sweep_epochs):
                trainer.train_one_round()

            acc_train  = evaluate_acc(trainer.get_model(), train_loader)
            acc_test   = evaluate_acc(trainer.get_model(), test_loader)
            actual_eps = trainer.get_epsilon()

            results.append({
                "target_epsilon": eps,
                "actual_epsilon": round(actual_eps, 4),
                "train_accuracy": round(acc_train, 4),
                "test_accuracy":  round(acc_test, 4),
                "delta":          round(delta, 8),
                "n_train":        n_train,
            })
            print(f"  ✅ train={acc_train:.4f} | test={acc_test:.4f} "
                  f"| actual_ε={actual_eps:.4f} | δ={delta:.2e}")

        except RuntimeError as e:
            print(f"  ⚠️  Skipped ε={eps}: {e}")
            results.append({
                "target_epsilon": eps,
                "actual_epsilon": None,
                "train_accuracy": None,
                "test_accuracy":  None,
                "error": str(e),
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Epsilon over FL Rounds
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment2(
    csv_path: str,
    fixed_eps: float,
    fl_rounds: int,
    local_epochs: int,
    batch_size: int,
    out_path: str,
) -> list:
    print("\n" + "=" * 55)
    print("EXPERIMENT 2: Epsilon over FL Rounds")
    print(f"  CSV          : {os.path.basename(csv_path)}")
    print(f"  ε target     : {fixed_eps}")
    print(f"  FL rounds    : {fl_rounds}")
    print(f"  Local epochs : {local_epochs}")
    print("=" * 55)

    rounds_data = []

    try:
        train_loader, test_loader, n_train, n_features, n_classes = load_client_data(
            csv_path, batch_size=batch_size
        )
        delta        = 1.0 / n_train
        total_epochs = local_epochs * fl_rounds

        model = get_model(input_dim=n_features, num_classes=n_classes)
        trainer = DPTrainer(
            model=model,
            target_epsilon=fixed_eps,
            target_delta=delta,
            epochs=total_epochs,
            num_classes=n_classes,
        )
        trainer.attach(train_loader)
        print(f"  PrivacyEngine attached: total_epochs={total_epochs} (δ={delta:.2e})\n")

        for round_num in range(1, fl_rounds + 1):
            for _ in range(local_epochs):
                trainer.train_one_round()

            eps_consumed     = trainer.get_epsilon()
            acc_test         = evaluate_acc(trainer.get_model(), test_loader)
            budget_remaining = fixed_eps - eps_consumed

            rounds_data.append({
                "fl_round":         round_num,
                "epsilon_consumed": round(float(eps_consumed), 4),
                "budget_remaining": round(float(budget_remaining), 4),
                "budget_exhausted": bool(eps_consumed >= fixed_eps),
                "test_accuracy":    round(float(acc_test), 4),
            })
            status = "⚠️  EXHAUSTED" if eps_consumed >= fixed_eps else "✅"
            print(f"  Round {round_num}: ε={eps_consumed:.4f} | "
                  f"test={acc_test:.4f} | remaining={budget_remaining:.4f} {status}")

    except RuntimeError as e:
        print(f"  ❌ Experiment 2 failed: {e}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rounds_data, f, indent=2)
    print(f"\n✅ Saved → {out_path}")
    return rounds_data


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(results: list, rounds_data: list, fixed_eps: float,
               fl_rounds: int, local_epochs: int, plot_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    valid = [r for r in results if r.get("test_accuracy") is not None]

    # Plot 1: Accuracy vs Epsilon
    if valid:
        eps_vals       = [r["target_epsilon"] for r in valid]
        acc_test_vals  = [r["test_accuracy"]  * 100 for r in valid]
        acc_train_vals = [r["train_accuracy"] * 100 for r in valid]
        max_x = max(eps_vals) + 2

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(eps_vals, acc_test_vals,  "o-",  color="#2E75B6", lw=2.5, ms=9,
                mfc="white", mew=2.5, label="Test Accuracy (DP)")
        ax.plot(eps_vals, acc_train_vals, "s--", color="#1D9E75", lw=1.8, ms=7,
                mfc="white", mew=2,   label="Train Accuracy (DP)", alpha=0.8)
        ax.axvspan(0,  3,     alpha=0.07, color="red",    label="High privacy (ε < 3)")
        ax.axvspan(3,  12,    alpha=0.07, color="orange", label="Balanced (3–12)")
        ax.axvspan(12, max_x, alpha=0.07, color="green",  label="Low privacy (ε > 12)")
        ax.axvline(x=fixed_eps, color="gray", ls="--", lw=1.5,
                   label=f"Target ε = {fixed_eps}")
        for x, y_t in zip(eps_vals, acc_test_vals):
            ax.annotate(f"{y_t:.1f}%", (x, y_t),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color="#2E75B6", fontweight="bold")
        ax.set_xlabel("Target Epsilon (ε)  —  Lower = Stronger Privacy", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Privacy-Utility Tradeoff\nIntelliClave FL+DP",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(0, max_x); ax.set_ylim(0, 105)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3); ax.set_facecolor("#FAFAFA")
        fig.tight_layout()
        p = os.path.join(plot_dir, "accuracy_vs_epsilon.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"✅ Saved: {p}")

    # Plot 2: Epsilon over FL Rounds
    if rounds_data:
        rnds = [r["fl_round"]         for r in rounds_data]
        eps  = [r["epsilon_consumed"] for r in rounds_data]
        acc  = [r["test_accuracy"] * 100 for r in rounds_data]
        rem  = [r["budget_remaining"] for r in rounds_data]

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(rnds, eps, "s-",  color="#D85A30", lw=2.5, ms=9, mfc="white",
                 mew=2.5, label="ε consumed")
        ax1.fill_between(rnds, eps, alpha=0.1, color="#D85A30")
        ax1.plot(rnds, rem, "^:",  color="#1D9E75", lw=1.8, ms=7, mfc="white",
                 mew=2, label="Budget remaining")
        ax1.axhline(y=fixed_eps, color="red", ls="--", lw=1.8,
                    label=f"Budget limit ε={fixed_eps}")
        for x, e in zip(rnds, eps):
            ax1.annotate(f"{e:.2f}", (x, e), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8, color="#D85A30")
        ax1.set_xlabel("FL Round", fontsize=12)
        ax1.set_ylabel("Epsilon (ε)", fontsize=12, color="#D85A30")
        ax1.tick_params(axis="y", labelcolor="#D85A30")
        ax1.set_ylim(0, fixed_eps * 1.4); ax1.set_xticks(rnds)

        ax2 = ax1.twinx()
        ax2.plot(rnds, acc, "o--", color="#2E75B6", lw=2.5, ms=9, mfc="white",
                 mew=2.5, label="Test Accuracy (%)")
        ax2.set_ylabel("Test Accuracy (%)", fontsize=12, color="#2E75B6")
        ax2.tick_params(axis="y", labelcolor="#2E75B6"); ax2.set_ylim(0, 105)

        ax1.set_title(f"Privacy Budget Consumption Across FL Rounds\n"
                      f"ε target={fixed_eps} | {fl_rounds} rounds × {local_epochs} epochs",
                      fontsize=13, fontweight="bold")
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, fontsize=9, loc="upper left")
        ax1.grid(True, alpha=0.3); ax1.set_facecolor("#FAFAFA")
        fig.tight_layout()
        p = os.path.join(plot_dir, "epsilon_over_rounds.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"✅ Saved: {p}")

    print(f"\nAll plots saved to: {plot_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list, rounds_data: list):
    print("\n" + "=" * 62)
    print("RESULTS SUMMARY")
    print("=" * 62)
    if results:
        print(f"\n{'Target ε':>10} | {'Actual ε':>10} | {'Test Acc':>10} | {'Train Acc':>10}")
        print("-" * 50)
        for r in results:
            if r.get("test_accuracy") is not None:
                print(f"{r['target_epsilon']:>10} | {r['actual_epsilon']:>10.4f} | "
                      f"{r['test_accuracy']*100:>9.2f}% | "
                      f"{r['train_accuracy']*100:>9.2f}%")
            else:
                print(f"{r['target_epsilon']:>10} | {'skipped':>10} | {'—':>10} | {'—':>10}")

    if rounds_data:
        print(f"\n{'Round':>7} | {'ε consumed':>12} | {'Remaining':>10} | "
              f"{'Test Acc':>10} | Exhausted")
        print("-" * 62)
        for r in rounds_data:
            print(f"{r['fl_round']:>7} | {r['epsilon_consumed']:>12.4f} | "
                  f"{r['budget_remaining']:>10.4f} | "
                  f"{r['test_accuracy']*100:>9.2f}% | "
                  f"{r['budget_exhausted']}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Epsilon sweep — privacy-utility tradeoff experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", default=os.path.join(_PROJECT_ROOT, "data", "processed", "client1.csv"),
        help="Client CSV to use for experiments.",
    )
    parser.add_argument(
        "--exp", type=int, choices=[1, 2], default=None,
        help="Run only experiment 1 or 2. Omit to run both.",
    )
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0, 20.0],
        help="Epsilon values to sweep (Experiment 1).",
    )
    parser.add_argument(
        "--sweep-epochs", type=int, default=5,
        help="Training epochs per epsilon value (Experiment 1).",
    )
    parser.add_argument(
        "--fixed-eps", type=float, default=10.0,
        help="Fixed epsilon for FL rounds experiment (Experiment 2).",
    )
    parser.add_argument(
        "--fl-rounds", type=int, default=5,
        help="Number of FL rounds to simulate (Experiment 2).",
    )
    parser.add_argument(
        "--local-epochs", type=int, default=3,
        help="Local epochs per FL round (Experiment 2).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "results"),
        help="Directory for output JSON files.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating matplotlib plots.",
    )
    args = parser.parse_args()

    sweep_json  = os.path.join(args.out_dir, "epsilon_sweep.json")
    rounds_json = os.path.join(args.out_dir, "epsilon_rounds.json")
    plot_dir    = os.path.join(args.out_dir, "plots")

    results     = []
    rounds_data = []

    run_exp1 = args.exp in (None, 1)
    run_exp2 = args.exp in (None, 2)

    if run_exp1:
        results = run_experiment1(
            csv_path=args.csv,
            epsilons=args.epsilons,
            sweep_epochs=args.sweep_epochs,
            batch_size=args.batch_size,
            out_path=sweep_json,
        )

    if run_exp2:
        rounds_data = run_experiment2(
            csv_path=args.csv,
            fixed_eps=args.fixed_eps,
            fl_rounds=args.fl_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            out_path=rounds_json,
        )

    if not args.no_plots:
        make_plots(
            results=results,
            rounds_data=rounds_data,
            fixed_eps=args.fixed_eps,
            fl_rounds=args.fl_rounds,
            local_epochs=args.local_epochs,
            plot_dir=plot_dir,
        )

    print_summary(results, rounds_data)


if __name__ == "__main__":
    main()
