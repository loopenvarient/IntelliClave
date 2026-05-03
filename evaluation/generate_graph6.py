"""
evaluation/generate_graph6.py

Graph 6 — Final Results Summary (4-panel figure)

Panels:
  A. Privacy-Utility Tradeoff: Test accuracy vs epsilon (sweep)
  B. FL+DP Training Curve: Accuracy & epsilon per round
  C. Per-Client Accuracy: Baseline vs FL+DP (bar chart)
  D. Cross-Validation Summary: Mean accuracy ± std per client

Output:
  results/graphs/graph6_final_results.png

Run:
    python evaluation/generate_graph6.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# ── Load data ─────────────────────────────────────────────────────────────────

def load(rel):
    with open(os.path.join(_ROOT, rel)) as f:
        return json.load(f)

sweep   = load("results/epsilon_sweep.json")
rounds  = load("results/epsilon_rounds.json")
cv      = load("results/cross_validation.json")
eval_   = load("results/fl_rounds/global_model_eval.json")

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

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('IntelliClave — Final Results Summary', fontsize=14, fontweight='bold',
             color=TEXT, y=0.98)
fig.patch.set_facecolor(DARK_BG)

# ── Panel A: Privacy-Utility Tradeoff ─────────────────────────────────────────
ax = axes[0, 0]
eps_vals  = [r['actual_epsilon'] for r in sweep]
test_acc  = [r['test_accuracy'] * 100 for r in sweep]
train_acc = [r['train_accuracy'] * 100 for r in sweep]

# Baseline (no DP) reference line
BASELINE = 96.99

ax.plot(eps_vals, test_acc,  'o-', color=BLUE,  linewidth=2, markersize=7,
        label='Test accuracy')
ax.plot(eps_vals, train_acc, 's--', color=GREEN, linewidth=1.5, markersize=6,
        label='Train accuracy', alpha=0.8)
ax.axhline(BASELINE, color=AMBER, linestyle=':', linewidth=1.5,
           label=f'No-DP baseline ({BASELINE}%)')

# Annotate the ε=10 operating point
op_idx = next(i for i, r in enumerate(sweep) if r['target_epsilon'] == 10.0)
ax.annotate(
    f'  ε=10\n  {test_acc[op_idx]:.1f}%',
    xy=(eps_vals[op_idx], test_acc[op_idx]),
    color=BLUE, fontsize=9,
)

ax.set_xlabel('Privacy budget ε')
ax.set_ylabel('Accuracy (%)')
ax.set_title('A — Privacy-Utility Tradeoff')
ax.set_ylim(40, 102)
ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True)

# ── Panel B: FL+DP Training Curve ─────────────────────────────────────────────
ax = axes[0, 1]
fl_rounds_x = [r['fl_round'] for r in rounds]
fl_acc      = [r['test_accuracy'] * 100 for r in rounds]
fl_eps      = [r['epsilon_consumed'] for r in rounds]

ax2 = ax.twinx()
ax2.set_facecolor(SURFACE)

line1, = ax.plot(fl_rounds_x, fl_acc, 'o-', color=GREEN, linewidth=2,
                 markersize=7, label='Test accuracy (%)')
line2, = ax2.plot(fl_rounds_x, fl_eps, 's--', color=AMBER, linewidth=2,
                  markersize=6, label='ε consumed')
ax2.axhline(10.0, color=AMBER, linestyle=':', linewidth=1, alpha=0.5)
ax2.set_ylabel('ε consumed', color=AMBER)
ax2.tick_params(axis='y', colors=AMBER)
ax2.set_ylim(0, 12)

ax.set_xlabel('FL Round')
ax.set_ylabel('Test Accuracy (%)', color=GREEN)
ax.tick_params(axis='y', colors=GREEN)
ax.set_title('B — FL+DP Training Curve (ε=10)')
ax.set_ylim(50, 100)
ax.set_xticks(fl_rounds_x)
ax.grid(True)

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=9, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# ── Panel C: Per-Client Accuracy — Baseline vs FL+DP ─────────────────────────
ax = axes[1, 0]

clients     = ['FitLife\n(Client 1)', 'MediTrack\n(Client 2)', 'CareWatch\n(Client 3)']
# Baseline from fl_history (10-round no-DP run)
baseline_acc = [95.81, 97.23, 97.75]
# FL+DP from global_model_eval.json
dp_acc = [c['accuracy'] * 100 for c in eval_['per_client']]

x = np.arange(len(clients))
w = 0.35

bars1 = ax.bar(x - w/2, baseline_acc, w, label='FL Baseline (no DP)',
               color=GREEN, alpha=0.85, edgecolor=BORDER)
bars2 = ax.bar(x + w/2, dp_acc,       w, label='FL + DP (ε=10)',
               color=BLUE,  alpha=0.85, edgecolor=BORDER)

# Value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=8.5, color=TEXT)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=8.5, color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels(clients, fontsize=9)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('C — Per-Client Accuracy: Baseline vs FL+DP')
ax.set_ylim(80, 102)
ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, axis='y')

# ── Panel D: Cross-Validation Summary ─────────────────────────────────────────
ax = axes[1, 1]

cv_clients = ['FitLife\n(Client 1)', 'MediTrack\n(Client 2)', 'CareWatch\n(Client 3)', 'Combined']
cv_keys    = ['1', '2', '3', 'combined']
cv_means   = [cv[k]['summary']['accuracy']['mean'] * 100 for k in cv_keys]
cv_stds    = [cv[k]['summary']['accuracy']['std']  * 100 for k in cv_keys]
cv_auc     = [cv[k]['summary']['auc_roc']['mean']  * 100 for k in cv_keys]

x = np.arange(len(cv_clients))
w = 0.35

bars_acc = ax.bar(x - w/2, cv_means, w, yerr=cv_stds, capsize=4,
                  label='Accuracy (mean ± std)', color=PURPLE, alpha=0.85,
                  edgecolor=BORDER, error_kw={'ecolor': TEXT, 'elinewidth': 1.2})
bars_auc = ax.bar(x + w/2, cv_auc,   w,
                  label='AUC-ROC (mean)', color=AMBER, alpha=0.85, edgecolor=BORDER)

for bar in bars_acc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=8, color=TEXT)
for bar in bars_auc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=8, color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels(cv_clients, fontsize=9)
ax.set_ylabel('Score (%)')
ax.set_title('D — 5-Fold Cross-Validation (no DP)')
ax.set_ylim(93, 102)
ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, axis='y')

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dir  = os.path.join(_ROOT, "results", "graphs")
out_path = os.path.join(out_dir, "graph6_final_results.png")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

print(f"✓ Graph 6 saved → {out_path}")

# Also copy to report/figures
report_path = os.path.join(_ROOT, "report", "figures", "graph6_final_results.png")
import shutil
shutil.copy(out_path, report_path)
print(f"✓ Copied  → {report_path}")
