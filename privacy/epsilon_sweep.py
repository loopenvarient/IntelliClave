# privacy/epsilon_sweep.py
"""
Epsilon sweep experiment for IntelliClave — privacy-utility tradeoff.

Experiment 1: Accuracy vs Epsilon
             Trains on a client CSV under different epsilon values at a fixed
             clipping norm. Now reports noise_multiplier alongside accuracy so
             you can see why accuracy collapses at low epsilon.

Experiment 2: Epsilon over FL Rounds
             Simulates FL rounds with a fixed epsilon, tracks budget consumption.

Experiment 3: Joint (epsilon × clipping_norm) grid sweep
             The critical missing experiment. Sweeps both epsilon and
             max_grad_norm together to produce a 2D accuracy surface.
             This is the only way to find the jointly optimal configuration —
             sweeping epsilon alone at a fixed clipping norm is misleading
             because the two parameters interact through the noise multiplier.

Usage:
    python privacy/epsilon_sweep.py                          # exp 1+2, defaults
    python privacy/epsilon_sweep.py --exp 1                  # only experiment 1
    python privacy/epsilon_sweep.py --exp 2                  # only experiment 2
    python privacy/epsilon_sweep.py --exp 3                  # only experiment 3
    python privacy/epsilon_sweep.py --csv data/processed/client2.csv
    python privacy/epsilon_sweep.py --epsilons 1 5 10 20
    python privacy/epsilon_sweep.py --max-grad-norm 1.5      # override clipping norm
    python privacy/epsilon_sweep.py --fl-rounds 10 --local-epochs 5
    python privacy/epsilon_sweep.py --no-plots               # skip matplotlib

Outputs:
    results/epsilon_sweep.json
    results/epsilon_rounds.json
    results/epsilon_norm_grid.json          (Experiment 3)
    results/plots/accuracy_vs_epsilon.png
    results/plots/epsilon_over_rounds.png
    results/plots/epsilon_norm_heatmap.png  (Experiment 3)
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
    max_grad_norm: float = 2.0,
) -> list:
    print("\n" + "=" * 55)
    print("EXPERIMENT 1: Accuracy vs Epsilon")
    print(f"  CSV           : {os.path.basename(csv_path)}")
    print(f"  ε list        : {epsilons}")
    print(f"  epochs        : {sweep_epochs}")
    print(f"  max_grad_norm : {max_grad_norm}")
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
                max_grad_norm=max_grad_norm,
                epochs=sweep_epochs,
                num_classes=n_classes,
            )
            trainer.attach(train_loader)
            report = trainer.calibration_report()
            print(f"  noise_multiplier={report['noise_multiplier']}  "
                  f"noise_std={report['noise_std']}  "
                  f"→ {report['interpretation']}")

            for _ in range(sweep_epochs):
                trainer.train_one_round()

            acc_train  = evaluate_acc(trainer.get_model(), train_loader)
            acc_test   = evaluate_acc(trainer.get_model(), test_loader)
            actual_eps = trainer.get_epsilon()

            results.append({
                "target_epsilon":   eps,
                "actual_epsilon":   round(actual_eps, 4),
                "max_grad_norm":    max_grad_norm,
                "noise_multiplier": report["noise_multiplier"],
                "noise_std":        report["noise_std"],
                "train_accuracy":   round(acc_train, 4),
                "test_accuracy":    round(acc_test, 4),
                "delta":            round(delta, 8),
                "n_train":          n_train,
                "interpretation":   report["interpretation"],
            })
            print(f"  ✅ train={acc_train:.4f} | test={acc_test:.4f} "
                  f"| actual_ε={actual_eps:.4f} | δ={delta:.2e}")

        except RuntimeError as e:
            print(f"  ⚠️  Skipped ε={eps}: {e}")
            results.append({
                "target_epsilon": eps,
                "actual_epsilon": None,
                "max_grad_norm":  max_grad_norm,
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
    max_grad_norm: float = 2.0,
) -> list:
    print("\n" + "=" * 55)
    print("EXPERIMENT 2: Epsilon over FL Rounds")
    print(f"  CSV           : {os.path.basename(csv_path)}")
    print(f"  ε target      : {fixed_eps}")
    print(f"  FL rounds     : {fl_rounds}")
    print(f"  Local epochs  : {local_epochs}")
    print(f"  max_grad_norm : {max_grad_norm}")
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
            max_grad_norm=max_grad_norm,
            epochs=total_epochs,
            num_classes=n_classes,
        )
        trainer.attach(train_loader)
        report = trainer.calibration_report()
        print(f"  PrivacyEngine attached: total_epochs={total_epochs} (δ={delta:.2e})")
        print(f"  noise_multiplier={report['noise_multiplier']}  "
              f"noise_std={report['noise_std']}  → {report['interpretation']}\n")

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
                "max_grad_norm":    max_grad_norm,
                "noise_multiplier": report["noise_multiplier"],
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
# Experiment 3 — Joint (epsilon × clipping_norm) grid sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment3(
    csv_path: str,
    epsilons: list,
    clipping_norms: list,
    sweep_epochs: int,
    batch_size: int,
    out_path: str,
) -> list:
    """
    Joint grid sweep over (epsilon, max_grad_norm).

    This is the experiment that was missing. Sweeping epsilon alone at a fixed
    clipping norm is misleading because the noise multiplier Opacus computes
    depends on both parameters together:

        noise_multiplier = f(epsilon, delta, max_grad_norm, epochs, batch_size)

    A low clipping norm at low epsilon produces a very high noise multiplier,
    which destroys gradient signal. A higher clipping norm at the same epsilon
    may produce a much lower noise multiplier and acceptable accuracy.

    The grid sweep finds the jointly optimal (epsilon, max_grad_norm) pair.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Joint (epsilon × clipping_norm) grid sweep")
    print(f"  CSV            : {os.path.basename(csv_path)}")
    print(f"  ε values       : {epsilons}")
    print(f"  clipping norms : {clipping_norms}")
    print(f"  epochs         : {sweep_epochs}")
    print("=" * 60)

    # Load data once — reused for every cell in the grid
    train_loader, test_loader, n_train, n_features, n_classes = load_client_data(
        csv_path, batch_size=batch_size
    )
    delta = 1.0 / n_train

    grid_results = []
    total = len(epsilons) * len(clipping_norms)
    done  = 0

    for eps in epsilons:
        for norm in clipping_norms:
            done += 1
            print(f"\n[{done}/{total}] ε={eps}  max_grad_norm={norm}")
            try:
                model   = get_model(input_dim=n_features, num_classes=n_classes)
                trainer = DPTrainer(
                    model=model,
                    target_epsilon=eps,
                    target_delta=delta,
                    max_grad_norm=norm,
                    epochs=sweep_epochs,
                    num_classes=n_classes,
                )
                trainer.attach(train_loader)
                report = trainer.calibration_report()
                print(f"  noise_multiplier={report['noise_multiplier']}  "
                      f"→ {report['interpretation']}")

                for _ in range(sweep_epochs):
                    trainer.train_one_round()

                acc_test  = evaluate_acc(trainer.get_model(), test_loader)
                actual_eps = trainer.get_epsilon()

                grid_results.append({
                    "target_epsilon":   eps,
                    "actual_epsilon":   round(actual_eps, 4),
                    "max_grad_norm":    norm,
                    "noise_multiplier": report["noise_multiplier"],
                    "noise_std":        report["noise_std"],
                    "test_accuracy":    round(acc_test, 4),
                    "delta":            round(delta, 8),
                    "interpretation":   report["interpretation"],
                })
                print(f"  test_acc={acc_test:.4f}  actual_ε={actual_eps:.4f}")

            except RuntimeError as e:
                print(f"  ⚠️  Skipped: {e}")
                grid_results.append({
                    "target_epsilon": eps,
                    "max_grad_norm":  norm,
                    "test_accuracy":  None,
                    "error":          str(e),
                })

    # Find the best (epsilon, norm) pair
    valid = [r for r in grid_results if r.get("test_accuracy") is not None]
    if valid:
        best = max(valid, key=lambda r: r["test_accuracy"])
        print(f"\n  Best configuration:")
        print(f"    ε={best['target_epsilon']}  "
              f"max_grad_norm={best['max_grad_norm']}  "
              f"→ test_acc={best['test_accuracy']:.4f}  "
              f"noise_multiplier={best['noise_multiplier']}")

    output = {
        "grid":  grid_results,
        "best":  best if valid else None,
        "epsilons":       epsilons,
        "clipping_norms": clipping_norms,
        "sweep_epochs":   sweep_epochs,
        "n_train":        n_train,
        "delta":          round(delta, 8),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved → {out_path}")
    return grid_results


# ─────────────────────────────────────────────────────────────────────────────

def make_plots(results: list, rounds_data: list, fixed_eps: float,
               fl_rounds: int, local_epochs: int, plot_dir: str,
               grid_results: list = None):
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

    # Plot 3: epsilon × clipping_norm heatmap
    if grid_results:
        valid_grid = [r for r in grid_results if r.get("test_accuracy") is not None]
        if valid_grid:
            import pandas as _pd
            df_grid = _pd.DataFrame(valid_grid)
            eps_vals  = sorted(df_grid["target_epsilon"].unique())
            norm_vals = sorted(df_grid["max_grad_norm"].unique())

            # Build accuracy matrix: rows=clipping_norm, cols=epsilon
            matrix = np.full((len(norm_vals), len(eps_vals)), np.nan)
            for _, row in df_grid.iterrows():
                ri = norm_vals.index(row["max_grad_norm"])
                ci = eps_vals.index(row["target_epsilon"])
                matrix[ri, ci] = row["test_accuracy"] * 100

            fig, ax = plt.subplots(figsize=(max(7, len(eps_vals) * 1.2),
                                            max(5, len(norm_vals) * 0.9)))
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                           vmin=max(0, np.nanmin(matrix) - 5),
                           vmax=min(100, np.nanmax(matrix) + 5))
            plt.colorbar(im, ax=ax, label="Test Accuracy (%)")

            ax.set_xticks(range(len(eps_vals)))
            ax.set_xticklabels([str(e) for e in eps_vals])
            ax.set_yticks(range(len(norm_vals)))
            ax.set_yticklabels([str(n) for n in norm_vals])
            ax.set_xlabel("Target Epsilon (ε)", fontsize=12)
            ax.set_ylabel("Max Gradient Norm (clipping norm)", fontsize=12)
            ax.set_title("DP-SGD Accuracy: Joint (ε × clipping_norm) Grid\n"
                         "IntelliClave — find the jointly optimal configuration",
                         fontsize=12, fontweight="bold")

            # Annotate each cell with accuracy value
            for ri in range(len(norm_vals)):
                for ci in range(len(eps_vals)):
                    val = matrix[ri, ci]
                    if not np.isnan(val):
                        ax.text(ci, ri, f"{val:.1f}%", ha="center", va="center",
                                fontsize=9, fontweight="bold",
                                color="black" if 30 < val < 85 else "white")

            fig.tight_layout()
            p = os.path.join(plot_dir, "epsilon_norm_heatmap.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {p}")


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
        "--exp", type=int, choices=[1, 2, 3], default=None,
        help="Run only experiment 1, 2, or 3. Omit to run 1+2.",
    )
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0, 20.0],
        help="Epsilon values to sweep (Experiments 1 and 3).",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=2.0,
        help="Gradient clipping norm for Experiments 1 and 2. "
             "Use --clipping-norms for the Experiment 3 grid.",
    )
    parser.add_argument(
        "--clipping-norms", nargs="+", type=float,
        default=[0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
        help="Clipping norm values for the Experiment 3 grid sweep.",
    )
    parser.add_argument(
        "--sweep-epochs", type=int, default=5,
        help="Training epochs per configuration (Experiments 1 and 3).",
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
    grid_json   = os.path.join(args.out_dir, "epsilon_norm_grid.json")
    plot_dir    = os.path.join(args.out_dir, "plots")

    results     = []
    rounds_data = []
    grid_results = []

    run_exp1 = args.exp in (None, 1)
    run_exp2 = args.exp in (None, 2)
    run_exp3 = args.exp == 3

    if run_exp1:
        results = run_experiment1(
            csv_path=args.csv,
            epsilons=args.epsilons,
            sweep_epochs=args.sweep_epochs,
            batch_size=args.batch_size,
            out_path=sweep_json,
            max_grad_norm=args.max_grad_norm,
        )

    if run_exp2:
        rounds_data = run_experiment2(
            csv_path=args.csv,
            fixed_eps=args.fixed_eps,
            fl_rounds=args.fl_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            out_path=rounds_json,
            max_grad_norm=args.max_grad_norm,
        )

    if run_exp3:
        grid_results = run_experiment3(
            csv_path=args.csv,
            epsilons=args.epsilons,
            clipping_norms=args.clipping_norms,
            sweep_epochs=args.sweep_epochs,
            batch_size=args.batch_size,
            out_path=grid_json,
        )

    if not args.no_plots:
        make_plots(
            results=results,
            rounds_data=rounds_data,
            fixed_eps=args.fixed_eps,
            fl_rounds=args.fl_rounds,
            local_epochs=args.local_epochs,
            plot_dir=plot_dir,
            grid_results=grid_results,
        )

    print_summary(results, rounds_data)


if __name__ == "__main__":
    main()
