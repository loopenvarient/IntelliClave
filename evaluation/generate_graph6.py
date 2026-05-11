"""
evaluation/generate_graph6.py

Graph 6 — Final Results Summary (4-panel figure)

Panels:
  A. Privacy-Utility Tradeoff: Test accuracy vs epsilon (sweep)
  B. FL+DP Training Curve: Accuracy & epsilon per round
  C. Per-Client Accuracy: Baseline vs FL+DP (bar chart)
  D. Cross-Validation Summary: Mean accuracy ± std per client

All labels, baselines, and client names are derived from the result JSON
files — no dataset-specific values are hardcoded here.

Output:
  results/graphs/graph6_final_results.png

Run:
    python evaluation/generate_graph6.py
    python evaluation/generate_graph6.py --op-epsilon 5.0
    python evaluation/generate_graph6.py --out results/graphs/my_summary.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

sys.path.insert(0, os.path.join(_ROOT, "config"))
from constants import DEFAULT_EPSILON   # noqa: E402

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG   = '#0b0d14'
SURFACE   = '#13161f'
BORDER    = '#252a3a'
BLUE      = '#4f8ef7'
GREEN     = '#3ecf8e'
AMBER     = '#f5a623'
PURPLE    = '#7c5cbf'
TEXT      = '#e2e8f0'
MUTED     = '#64748b'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    SURFACE,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT,
    'axes.titlecolor':   TEXT,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'text.color':        TEXT,
    'grid.color':        BORDER,
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
})


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_json(rel_path: str) -> dict | list:
    full = os.path.join(_ROOT, rel_path)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"Required results file not found: {full}\n"
            "Run the corresponding experiment script first."
        )
    with open(full) as f:
        return json.load(f)


def _client_label(client_entry: dict, idx: int) -> str:
    """
    Build a human-readable client label from the eval JSON entry.
    Uses 'client_name' if present, otherwise falls back to 'client_id'
    or a generic 'Client N' label.
    """
    name = client_entry.get("client_name") or client_entry.get("client_id")
    if name:
        return str(name)
    return f"Client {idx + 1}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(op_epsilon: float = DEFAULT_EPSILON,
         out_path: str = None):

    sweep  = load_json("results/epsilon_sweep.json")
    rounds = load_json("results/epsilon_rounds.json")
    cv     = load_json("results/cross_validation.json")
    eval_  = load_json("results/fl_rounds/global_model_eval.json")

    if out_path is None:
        out_path = os.path.join(_ROOT, "results", "graphs", "graph6_final_results.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('IntelliClave — Final Results Summary', fontsize=14,
                 fontweight='bold', color=TEXT, y=0.98)
    fig.patch.set_facecolor(DARK_BG)

    # ── Panel A: Privacy-Utility Tradeoff ─────────────────────────────────────
    ax = axes[0, 0]
    eps_vals  = [r['actual_epsilon'] for r in sweep]
    test_acc  = [r['test_accuracy']  * 100 for r in sweep]
    train_acc = [r['train_accuracy'] * 100 for r in sweep]

    # Derive no-DP baseline: highest epsilon in the sweep approximates no-DP
    valid_test = [r['test_accuracy'] for r in sweep if r.get('test_accuracy') is not None]
    baseline_acc = max(valid_test) * 100 if valid_test else None

    ax.plot(eps_vals, test_acc,  'o-', color=BLUE,  linewidth=2, markersize=7,
            label='Test accuracy')
    ax.plot(eps_vals, train_acc, 's--', color=GREEN, linewidth=1.5, markersize=6,
            label='Train accuracy', alpha=0.8)
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color=AMBER, linestyle=':', linewidth=1.5,
                   label=f'Best sweep acc ({baseline_acc:.1f}%)')

    # Annotate the operating-point epsilon
    op_candidates = [
        (i, r) for i, r in enumerate(sweep)
        if r.get('target_epsilon') == op_epsilon
    ]
    if op_candidates:
        op_idx, _ = op_candidates[0]
        ax.annotate(
            f'  ε={op_epsilon}\n  {test_acc[op_idx]:.1f}%',
            xy=(eps_vals[op_idx], test_acc[op_idx]),
            color=BLUE, fontsize=9,
        )

    ax.set_xlabel('Privacy budget ε')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('A — Privacy-Utility Tradeoff')
    ax.set_ylim(40, 102)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
    ax.grid(True)

    # ── Panel B: FL+DP Training Curve ─────────────────────────────────────────
    ax = axes[0, 1]
    fl_rounds_x = [r['fl_round']         for r in rounds]
    fl_acc      = [r['test_accuracy'] * 100 for r in rounds]
    fl_eps      = [r['epsilon_consumed']  for r in rounds]

    ax2 = ax.twinx()
    ax2.set_facecolor(SURFACE)

    line1, = ax.plot(fl_rounds_x, fl_acc, 'o-', color=GREEN, linewidth=2,
                     markersize=7, label='Test accuracy (%)')
    line2, = ax2.plot(fl_rounds_x, fl_eps, 's--', color=AMBER, linewidth=2,
                      markersize=6, label='ε consumed')
    ax2.axhline(op_epsilon, color=AMBER, linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_ylabel('ε consumed', color=AMBER)
    ax2.tick_params(axis='y', colors=AMBER)
    eps_max = max(fl_eps) if fl_eps else op_epsilon
    ax2.set_ylim(0, max(eps_max, op_epsilon) * 1.2)

    ax.set_xlabel('FL Round')
    ax.set_ylabel('Test Accuracy (%)', color=GREEN)
    ax.tick_params(axis='y', colors=GREEN)
    ax.set_title(f'B — FL+DP Training Curve (ε={op_epsilon})')
    ax.set_ylim(50, 100)
    ax.set_xticks(fl_rounds_x)
    ax.grid(True)

    lines  = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, facecolor=SURFACE,
              edgecolor=BORDER, labelcolor=TEXT)

    # ── Panel C: Per-Client Accuracy — Baseline vs FL+DP ─────────────────────
    ax = axes[1, 0]

    per_client = eval_.get('per_client', [])
    n_clients  = len(per_client)

    # Client labels: prefer names from eval JSON, fall back to generic
    client_labels = [
        _client_label(c, i).replace(" ", "\n") for i, c in enumerate(per_client)
    ]

    # Baseline: read from eval JSON if present, otherwise use FL+DP as proxy
    baseline_key = 'baseline_accuracy'
    if all(baseline_key in c for c in per_client):
        baseline_acc_list = [c[baseline_key] * 100 for c in per_client]
        baseline_label    = 'FL Baseline (no DP)'
    else:
        # Fall back: use the best sweep accuracy as a flat reference
        flat_base = max(valid_test) * 100 if valid_test else None
        baseline_acc_list = [flat_base] * n_clients if flat_base else None
        baseline_label    = 'Best sweep acc (proxy)'

    dp_acc = [c['accuracy'] * 100 for c in per_client]
    x = np.arange(n_clients)
    w = 0.35

    if baseline_acc_list:
        bars1 = ax.bar(x - w/2, baseline_acc_list, w, label=baseline_label,
                       color=GREEN, alpha=0.85, edgecolor=BORDER)
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{bar.get_height():.1f}%', ha='center', va='bottom',
                    fontsize=8.5, color=TEXT)

    bars2 = ax.bar(x + (w/2 if baseline_acc_list else 0), dp_acc, w,
                   label=f'FL + DP (ε={op_epsilon})',
                   color=BLUE, alpha=0.85, edgecolor=BORDER)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(client_labels, fontsize=9)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('C — Per-Client Accuracy: Baseline vs FL+DP')
    y_min = min((baseline_acc_list or dp_acc) + dp_acc)
    ax.set_ylim(max(0, y_min - 10), 102)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
    ax.grid(True, axis='y')

    # ── Panel D: Cross-Validation Summary ─────────────────────────────────────
    ax = axes[1, 1]

    # Derive CV keys and labels dynamically from the JSON
    # Keys that are not metadata fields are treated as client/split entries
    _meta_keys = {'config', 'summary', 'timestamp'}
    cv_keys = [k for k in cv.keys() if k not in _meta_keys
               and isinstance(cv[k], dict) and 'summary' in cv[k]]

    cv_labels = []
    for k in cv_keys:
        entry = cv[k]
        label = entry.get('client_name') or entry.get('client_id') or k
        cv_labels.append(str(label).replace(" ", "\n"))

    cv_means = [cv[k]['summary']['accuracy']['mean'] * 100 for k in cv_keys]
    cv_stds  = [cv[k]['summary']['accuracy']['std']  * 100 for k in cv_keys]
    cv_auc   = [cv[k]['summary']['auc_roc']['mean']  * 100 for k in cv_keys]

    x = np.arange(len(cv_keys))
    w = 0.35

    bars_acc = ax.bar(x - w/2, cv_means, w, yerr=cv_stds, capsize=4,
                      label='Accuracy (mean ± std)', color=PURPLE, alpha=0.85,
                      edgecolor=BORDER,
                      error_kw={'ecolor': TEXT, 'elinewidth': 1.2})
    bars_auc = ax.bar(x + w/2, cv_auc, w,
                      label='AUC-ROC (mean)', color=AMBER, alpha=0.85,
                      edgecolor=BORDER)

    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8, color=TEXT)
    for bar in bars_auc:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(cv_labels, fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title('D — 5-Fold Cross-Validation (no DP)')
    all_scores = cv_means + cv_auc
    ax.set_ylim(max(0, min(all_scores) - 5), 102)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
    ax.grid(True, axis='y')

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"✓ Graph 6 saved → {out_path}")

    # Also copy to report/figures if that directory exists
    report_dir = os.path.join(_ROOT, "report", "figures")
    if os.path.isdir(report_dir):
        import shutil
        report_path = os.path.join(report_dir, os.path.basename(out_path))
        shutil.copy(out_path, report_path)
        print(f"✓ Copied  → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the 4-panel final results summary graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--op-epsilon", type=float, default=DEFAULT_EPSILON,
                        help="Operating-point epsilon to annotate on Panel A and label Panel B.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path. Defaults to results/graphs/graph6_final_results.png.")
    args = parser.parse_args()
    main(op_epsilon=args.op_epsilon, out_path=args.out)
