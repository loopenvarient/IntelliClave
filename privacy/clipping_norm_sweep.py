"""
privacy/clipping_norm_sweep.py

Clipping norm sensitivity analysis for DP-SGD.

This is the specific diagnostic that was missing from the evaluation suite.
It answers the question: given a fixed target epsilon, what clipping norm
produces the best accuracy without sacrificing the privacy guarantee?

Why this matters
----------------
Opacus computes the noise multiplier σ from (epsilon, delta, max_grad_norm,
epochs, batch_size) together. The relationship is:

    noise_std = σ × max_grad_norm

If max_grad_norm is too low:
  - Gradients are heavily clipped → gradient signal is destroyed
  - Opacus still adds noise proportional to max_grad_norm
  - Net effect: low signal + noise → accuracy collapse
  - This is the likely cause of ε=1 → 50.89% accuracy

If max_grad_norm is too high:
  - Gradients are barely clipped → individual sample influence is large
  - Opacus adds more noise to compensate → noise_std grows
  - Net effect: the privacy guarantee weakens or accuracy degrades

The optimal max_grad_norm is dataset- and model-specific. This script
finds it empirically by sweeping clipping norms at a fixed epsilon and
reporting accuracy, noise_multiplier, and noise_std for each.

Usage
-----
    # Default: sweep at ε=10 (the project's production epsilon)
    python privacy/clipping_norm_sweep.py

    # Sweep at ε=1 to diagnose the accuracy collapse
    python privacy/clipping_norm_sweep.py --epsilon 1.0

    # Custom norm range
    python privacy/clipping_norm_sweep.py --norms 0.1 0.5 1.0 2.0 5.0 10.0

    # All three epsilons of interest
    python privacy/clipping_norm_sweep.py --epsilon 1.0 --norms 0.1 0.5 1.0 2.0 5.0
    python privacy/clipping_norm_sweep.py --epsilon 5.0 --norms 0.1 0.5 1.0 2.0 5.0
    python privacy/clipping_norm_sweep.py --epsilon 10.0 --norms 0.1 0.5 1.0 2.0 5.0

Output
------
    results/clipping_norm_sweep_eps{epsilon}.json
    results/plots/clipping_norm_sweep_eps{epsilon}.png
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_FL_DIR       = os.path.join(_PROJECT_ROOT, "fl")

sys.path.insert(0, _FL_DIR)
sys.path.insert(0, _THIS_DIR)

from model import get_model        # noqa: E402
from dp_trainer import DPTrainer   # noqa: E402

# Default sweep values
DEFAULT_EPSILON     = 10.0
DEFAULT_NORMS       = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
DEFAULT_EPOCHS      = 5
DEFAULT_BATCH_SIZE  = 64
DEFAULT_CSV         = os.path.join(_PROJECT_ROOT, "data", "processed", "client1.csv")


def load_data(csv_path: str, batch_size: int, label_col: str = "label"):
    df = pd.read_csv(csv_path).dropna()
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols].values.astype(np.float32)
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
        batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader, len(y_tr), X_tr.shape[1], n_classes


def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in loader:
            preds    = model(X_b).argmax(1)
            correct += (preds == y_b).sum().item()
            total   += len(y_b)
    return correct / total if total > 0 else 0.0


def run_sweep(
    csv_path: str,
    epsilon: float,
    clipping_norms: list,
    epochs: int,
    batch_size: int,
    out_path: str,
) -> list:
    print("=" * 60)
    print(f"Clipping Norm Sensitivity Analysis  (ε={epsilon})")
    print(f"  CSV    : {os.path.basename(csv_path)}")
    print(f"  norms  : {clipping_norms}")
    print(f"  epochs : {epochs}")
    print("=" * 60)

    train_loader, test_loader, n_train, n_features, n_classes = load_data(
        csv_path, batch_size
    )
    delta = 1.0 / n_train
    print(f"  n_train={n_train}  n_features={n_features}  "
          f"n_classes={n_classes}  δ={delta:.2e}\n")

    results = []
    print(f"  {'norm':>8} | {'σ (noise_mult)':>16} | {'noise_std':>10} | "
          f"{'test_acc':>10} | interpretation")
    print("  " + "-" * 72)

    for norm in clipping_norms:
        try:
            model   = get_model(input_dim=n_features, num_classes=n_classes)
            trainer = DPTrainer(
                model=model,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=norm,
                epochs=epochs,
                num_classes=n_classes,
            )
            trainer.attach(train_loader)
            report = trainer.calibration_report()

            for _ in range(epochs):
                trainer.train_one_round()

            acc       = evaluate(trainer.get_model(), test_loader)
            actual_eps = trainer.get_epsilon()

            nm  = report["noise_multiplier"]
            ns  = report["noise_std"]
            interp = report["interpretation"]

            print(f"  {norm:>8.2f} | {nm if nm else 'n/a':>16} | "
                  f"{ns if ns else 'n/a':>10} | "
                  f"{acc*100:>9.2f}% | {interp}")

            results.append({
                "max_grad_norm":    norm,
                "target_epsilon":   epsilon,
                "actual_epsilon":   round(actual_eps, 4),
                "noise_multiplier": nm,
                "noise_std":        ns,
                "test_accuracy":    round(acc, 4),
                "delta":            round(delta, 8),
                "n_train":          n_train,
                "interpretation":   interp,
            })

        except RuntimeError as e:
            print(f"  {norm:>8.2f} | {'ERROR':>16} | {'':>10} | {'':>10} | {e}")
            results.append({
                "max_grad_norm":  norm,
                "target_epsilon": epsilon,
                "test_accuracy":  None,
                "error":          str(e),
            })

    # Find best norm
    valid = [r for r in results if r.get("test_accuracy") is not None]
    best  = max(valid, key=lambda r: r["test_accuracy"]) if valid else None

    print("\n" + "=" * 60)
    if best:
        print(f"  Best clipping norm : {best['max_grad_norm']}")
        print(f"  Test accuracy      : {best['test_accuracy']*100:.2f}%")
        print(f"  Noise multiplier   : {best['noise_multiplier']}")
        print(f"  Noise std          : {best['noise_std']}")
        print(f"  Interpretation     : {best['interpretation']}")

    # Diagnosis: was the default norm (2.0) near-optimal?
    default_result = next((r for r in valid if r["max_grad_norm"] == 2.0), None)
    if default_result and best and best["max_grad_norm"] != 2.0:
        acc_gap = best["test_accuracy"] - default_result["test_accuracy"]
        if acc_gap > 0.02:
            print(f"\n  ⚠ Default norm=2.0 is suboptimal by {acc_gap*100:.1f}% accuracy.")
            print(f"    Recommended: max_grad_norm={best['max_grad_norm']}")
            print(f"    Update DPTrainer default or pass --max-grad-norm "
                  f"{best['max_grad_norm']} to fl/run_client.py")
    elif default_result and best and best["max_grad_norm"] == 2.0:
        print(f"\n  ✓ Default norm=2.0 is near-optimal for ε={epsilon}.")
    print("=" * 60)

    output = {
        "epsilon":        epsilon,
        "clipping_norms": clipping_norms,
        "epochs":         epochs,
        "results":        results,
        "best":           best,
        "default_norm_result": default_result,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved → {out_path}")
    return results


def make_plot(results: list, epsilon: float, plot_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot.")
        return

    valid = [r for r in results if r.get("test_accuracy") is not None]
    if not valid:
        return

    norms    = [r["max_grad_norm"]    for r in valid]
    accs     = [r["test_accuracy"] * 100 for r in valid]
    noise_ms = [r["noise_multiplier"] or 0 for r in valid]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(norms, accs, "o-", color="#2E75B6", lw=2.5, ms=9,
             mfc="white", mew=2.5, label="Test Accuracy (%)")
    for x, y in zip(norms, accs):
        ax1.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=8, color="#2E75B6", fontweight="bold")
    ax1.set_xlabel("Max Gradient Norm (clipping norm C)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="#2E75B6")
    ax1.tick_params(axis="y", labelcolor="#2E75B6")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(norms, noise_ms, "s--", color="#D85A30", lw=2, ms=7,
             mfc="white", mew=2, label="Noise Multiplier σ", alpha=0.85)
    ax2.set_ylabel("Noise Multiplier σ", fontsize=12, color="#D85A30")
    ax2.tick_params(axis="y", labelcolor="#D85A30")

    # Mark the default norm
    ax1.axvline(x=2.0, color="gray", ls=":", lw=1.5, label="Default norm=2.0")

    ax1.set_title(
        f"Clipping Norm Sensitivity Analysis  (ε={epsilon})\n"
        "Higher norm → less clipping but more noise. Find the sweet spot.",
        fontsize=12, fontweight="bold",
    )
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")
    fig.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    p = os.path.join(plot_dir, f"clipping_norm_sweep_eps{epsilon}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved → {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Clipping norm sensitivity analysis for DP-SGD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",     default=DEFAULT_CSV)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help="Target epsilon to hold fixed during the sweep.")
    parser.add_argument("--norms",   nargs="+", type=float, default=DEFAULT_NORMS,
                        help="Clipping norm values to sweep.")
    parser.add_argument("--epochs",  type=int,  default=DEFAULT_EPOCHS,
                        help="Training epochs per norm value.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "results"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    out_path = os.path.join(
        args.out_dir, f"clipping_norm_sweep_eps{args.epsilon}.json"
    )
    plot_dir = os.path.join(args.out_dir, "plots")

    results = run_sweep(
        csv_path=args.csv,
        epsilon=args.epsilon,
        clipping_norms=args.norms,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_path=out_path,
    )

    if not args.no_plot:
        make_plot(results, args.epsilon, plot_dir)


if __name__ == "__main__":
    main()
