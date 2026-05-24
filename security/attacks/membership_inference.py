"""
security/attacks/membership_inference.py

Attack 2: Membership Inference

Goal: Determine whether a given sample was part of the training set.
      If the model is significantly more confident on training samples than
      on held-out samples, an adversary can infer membership.

Method (shadow model / threshold attack):
  - Split each client's data into "member" (training) and "non-member" (held-out).
  - Query the model with both sets and record the max softmax confidence.
  - Train a simple threshold classifier: if confidence > threshold → member.
  - Measure attack accuracy, precision, recall, and AUC.

Threat model (STRIDE: Information Disclosure):
  A hospital or company participating in FL could query the global model with
  their own patients' data to infer whether a competitor's patient data was
  used in training — violating data privacy agreements.

Output:
  results/attacks/membership_inference.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "fl"))

from model import get_model          # noqa: E402
from data_utils import infer_csv_schema  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "config"))
from constants import (              # noqa: E402
    DEFAULT_EPSILON, LABEL_COL, RANDOM_SEED,
)

MODEL_PATH     = os.path.join(_ROOT, "results", "fl_rounds", "global_model_latest.pth")
META_PATH      = os.path.join(_ROOT, "results", "fl_rounds", "model_meta.json")
PROCESSED_DIR  = os.path.join(_ROOT, "data", "processed")
CLIENT_CSVS    = sorted(
    os.path.join(PROCESSED_DIR, f)
    for f in os.listdir(PROCESSED_DIR)
    if f.endswith(".csv")
) if os.path.isdir(PROCESSED_DIR) else []
OUT_PATH       = os.path.join(_ROOT, "results", "attacks", "membership_inference.json")
OUT_PATH_NODP  = os.path.join(_ROOT, "results", "attacks", "membership_inference_nodp.json")
TEST_SPLIT  = 0.3


def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    if not CLIENT_CSVS:
        raise FileNotFoundError("No model_meta.json and no CSVs in data/processed/")
    input_dim, _ = infer_csv_schema(CLIENT_CSVS[0], LABEL_COL)
    df = pd.read_csv(CLIENT_CSVS[0], usecols=[LABEL_COL])
    num_classes = int(df[LABEL_COL].nunique())
    return {"input_dim": input_dim, "num_classes": num_classes,
            "class_names": [f"class_{i}" for i in range(num_classes)]}


def load_model(model_path: str = MODEL_PATH):
    meta = load_meta()
    model = get_model(meta["input_dim"], meta["num_classes"])
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_confidences(model, X: np.ndarray) -> np.ndarray:
    """Return max softmax confidence for each sample."""
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        probs  = torch.softmax(logits, dim=1).numpy()
    return probs.max(axis=1)   # shape (N,)


def threshold_attack(member_conf, nonmember_conf):
    """
    Find the optimal confidence threshold that maximises attack accuracy.
    Returns metrics dict.
    """
    scores = np.concatenate([member_conf, nonmember_conf])
    labels = np.concatenate([
        np.ones(len(member_conf)),
        np.zeros(len(nonmember_conf)),
    ])

    # AUC — measures how well confidence separates members from non-members
    auc = roc_auc_score(labels, scores)

    # Find threshold that maximises accuracy
    fpr, tpr, thresholds = roc_curve(labels, scores)
    accs = [accuracy_score(labels, (scores >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(accs))
    best_threshold = float(thresholds[best_idx])

    preds = (scores >= best_threshold).astype(int)
    return {
        "auc":           round(float(auc), 6),
        "best_threshold": round(best_threshold, 6),
        "accuracy":      round(float(accuracy_score(labels, preds)), 6),
        "precision":     round(float(precision_score(labels, preds, zero_division=0)), 6),
        "recall":        round(float(recall_score(labels, preds, zero_division=0)), 6),
        "member_conf_mean":    round(float(member_conf.mean()), 6),
        "nonmember_conf_mean": round(float(nonmember_conf.mean()), 6),
        "conf_gap":      round(float(member_conf.mean() - nonmember_conf.mean()), 6),
    }


def main(out_path: str = OUT_PATH, dp_mode: bool = False,
         model_path: str = MODEL_PATH, label_col: str = LABEL_COL,
         test_split: float = TEST_SPLIT, seed: int = RANDOM_SEED):
    print("=" * 55)
    print("Attack 2: Membership Inference")
    if dp_mode:
        print("Mode: DP-trained model")
    else:
        print("Mode: No-DP model (baseline)")
    print("=" * 55)

    model = load_model(model_path=model_path)
    all_results = []

    for i, csv_path in enumerate(CLIENT_CSVS, 1):
        client_name = f"client{i}"
        print(f"\n  Client {i} ({client_name})...")

        df = pd.read_csv(csv_path).dropna()
        feat_cols = [c for c in df.columns if c != label_col]
        X = df[feat_cols].values.astype(np.float32)
        raw_y = df[label_col].values

        # Auto-detect label offset — works for 0-based, 1-based, or any range
        unique = sorted(np.unique(raw_y).tolist())
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique):
            offset = int(min(unique))
            y = raw_y.astype(np.int64) - offset
        else:
            label_map = {v: i for i, v in enumerate(unique)}
            y = np.array([label_map[v] for v in raw_y], dtype=np.int64)

        # Split: members = training set, non-members = held-out
        X_mem, X_nonmem, _, _ = train_test_split(
            X, y, test_size=test_split, random_state=seed, stratify=y
        )

        # Scale using member data only (simulates attacker knowing train distribution)
        scaler = StandardScaler()
        X_mem_s    = scaler.fit_transform(X_mem)
        X_nonmem_s = scaler.transform(X_nonmem)

        mem_conf    = get_confidences(model, X_mem_s)
        nonmem_conf = get_confidences(model, X_nonmem_s)

        metrics = threshold_attack(mem_conf, nonmem_conf)

        # Risk: AUC > 0.7 is a meaningful attack; random = 0.5
        risk = "HIGH" if metrics["auc"] > 0.7 else \
               "MEDIUM" if metrics["auc"] > 0.6 else "LOW"

        print(f"    Members    : {len(X_mem)} samples  "
              f"avg_conf={metrics['member_conf_mean']:.4f}")
        print(f"    Non-members: {len(X_nonmem)} samples  "
              f"avg_conf={metrics['nonmember_conf_mean']:.4f}")
        print(f"    Conf gap   : {metrics['conf_gap']:.4f}")
        print(f"    AUC        : {metrics['auc']:.4f}")
        print(f"    Accuracy   : {metrics['accuracy']:.4f}")
        print(f"    Risk       : {risk}")

        all_results.append({
            "client_id":   i,
            "client_name": client_name,
            "n_members":   len(X_mem),
            "n_nonmembers": len(X_nonmem),
            "metrics":     metrics,
            "risk_level":  risk,
        })

    # Overall summary
    avg_auc  = np.mean([r["metrics"]["auc"] for r in all_results])
    avg_acc  = np.mean([r["metrics"]["accuracy"] for r in all_results])
    avg_gap  = np.mean([r["metrics"]["conf_gap"] for r in all_results])
    high_risk = sum(1 for r in all_results if r["risk_level"] == "HIGH")

    summary = {
        "avg_auc":      round(float(avg_auc), 6),
        "avg_accuracy": round(float(avg_acc), 6),
        "avg_conf_gap": round(float(avg_gap), 6),
        "high_risk_clients": high_risk,
        "verdict": (
            "VULNERABLE — model significantly overfits, membership is detectable"
            if avg_auc > 0.7 else
            "MODERATE — slight overfitting, limited membership signal"
            if avg_auc > 0.6 else
            "RESISTANT — confidence gap is small, attack near random (AUC ≈ 0.5)"
        ),
        "dp_note": (
            "Re-run with DP-trained model (--dp flag) to compare. "
            "DP-SGD should reduce the confidence gap and lower attack AUC."
        ),
    }

    output = {
        "attack":  "membership_inference",
        "mode":    "dp" if dp_mode else "no_dp",
        "results": all_results,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Avg AUC      : {avg_auc:.4f}  (random = 0.5, perfect = 1.0)")
    print(f"  Avg accuracy : {avg_acc:.4f}")
    print(f"  Avg conf gap : {avg_gap:.4f}")
    print(f"  Verdict : {summary['verdict']}")
    print(f"{'='*55}")
    print(f"\n✅ Saved → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Membership Inference Attack against a trained FL model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dp", action="store_true",
                        help="Tag results as DP-trained model run (saves to _nodp.json when not set).")
    parser.add_argument("--model-path", default=MODEL_PATH,
                        help="Path to the trained model .pth file.")
    parser.add_argument("--label-col", default=LABEL_COL,
                        help="Name of the label column in client CSVs.")
    parser.add_argument("--test-split", type=float, default=TEST_SPLIT,
                        help="Fraction of each client's data held out as non-members.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--out", default=None,
                        help="Override output JSON path (default: auto-selected by --dp flag).")
    args = parser.parse_args()

    # Route output: no --dp flag → _nodp.json, --dp flag → .json (canonical DP result)
    out_path = args.out or (OUT_PATH if args.dp else OUT_PATH_NODP)
    main(
        out_path=out_path,
        dp_mode=args.dp,
        model_path=args.model_path,
        label_col=args.label_col,
        test_split=args.test_split,
        seed=args.seed,
    )
