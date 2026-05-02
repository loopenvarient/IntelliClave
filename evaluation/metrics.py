"""
evaluation/metrics.py
Evaluation utilities for IntelliClave FL+DP training rounds.
"""
import json
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score
)

CLASS_NAMES = [
    'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
    'SITTING', 'STANDING', 'LAYING'
]
N_CLASSES = len(CLASS_NAMES)


def evaluate(y_true, y_pred, y_prob, epsilon, round_num):
    """
    Compute evaluation metrics for one FL round.

    Args:
        y_true    : array-like (N,)   ground truth labels 0-5
        y_pred    : array-like (N,)   predicted labels 0-5
        y_prob    : array-like (N, 6) predicted probabilities
        epsilon   : float             privacy budget spent so far
        round_num : int               FL round number

    Returns:
        dict with macro_f1, per_class_f1, accuracy, auc_roc, epsilon, round_num
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    macro_f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class   = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(N_CLASSES)))
    accuracy    = accuracy_score(y_true, y_pred)

    # AUC-ROC: needs all classes present; fall back gracefully
    try:
        auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc_roc = float('nan')

    return {
        'round':        round_num,
        'epsilon':      round(float(epsilon), 6),
        'accuracy':     round(float(accuracy), 6),
        'macro_f1':     round(float(macro_f1), 6),
        'per_class_f1': {CLASS_NAMES[i]: round(float(v), 6) for i, v in enumerate(per_class)},
        'auc_roc':      round(float(auc_roc), 6) if not np.isnan(auc_roc) else None,
    }


def save_results(results_dict, path):
    """Save results dict to a JSON file."""
    with open(path, 'w') as f:
        json.dump(results_dict, f, indent=2)


def print_summary(results):
    """Print a formatted summary of evaluation results."""
    print(f"\n{'='*50}")
    print(f"Round {results['round']}  |  epsilon={results['epsilon']}  |  delta=1e-5")
    print(f"{'='*50}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Macro F1  : {results['macro_f1']:.4f}")
    print(f"  AUC-ROC   : {results['auc_roc']}")
    print(f"  Per-class F1:")
    for cls, val in results['per_class_f1'].items():
        print(f"    {cls:<25} {val:.4f}")
    print(f"{'='*50}\n")


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    N = 600  # 100 per class
    y_true = np.repeat(np.arange(6), 100)

    # 1. Perfect predictions
    y_pred_perfect = y_true.copy()
    y_prob_perfect = np.eye(6)[y_pred_perfect]
    r1 = evaluate(y_true, y_pred_perfect, y_prob_perfect, epsilon=0.5, round_num=1)
    assert r1['macro_f1'] == 1.0, f"Expected 1.0, got {r1['macro_f1']}"
    print(f"Perfect predictions  → macro F1 = {r1['macro_f1']:.3f}  (expected 1.0) ✓")

    # 2. Random predictions
    y_pred_random = np.random.randint(0, 6, N)
    y_prob_random = np.random.dirichlet(np.ones(6), N)
    r2 = evaluate(y_true, y_pred_random, y_prob_random, epsilon=1.0, round_num=2)
    assert r2['macro_f1'] < 0.35, f"Expected ~0.167, got {r2['macro_f1']}"
    print(f"Random predictions   → macro F1 = {r2['macro_f1']:.3f}  (expected ~0.167) ✓")

    # 3. All-same class
    y_pred_same = np.zeros(N, dtype=int)
    y_prob_same = np.zeros((N, 6)); y_prob_same[:, 0] = 1.0
    r3 = evaluate(y_true, y_pred_same, y_prob_same, epsilon=1.5, round_num=3)
    assert r3['macro_f1'] < 0.25, f"Expected very low, got {r3['macro_f1']}"
    print(f"All-same class       → macro F1 = {r3['macro_f1']:.3f}  (expected very low) ✓")

    print("\nAll metric tests passed ✓")
    print_summary(r1)
