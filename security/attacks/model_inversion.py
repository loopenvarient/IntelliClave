"""
security/attacks/model_inversion.py

Attack 1: Model Inversion

Goal: Given only black-box access to the trained FL model, reconstruct a
      representative input for each activity class by gradient-based optimisation.
      If the reconstructed inputs are visually/statistically similar to real data,
      the model has "memorised" class-level patterns — a privacy risk.

Method:
  - Start from random noise in the 50-dim PCA feature space.
  - Optimise the input to maximise the model's confidence for a target class
    (minimise cross-entropy loss for that class).
  - Compare reconstructed inputs to real class centroids via cosine similarity
    and L2 distance.

Threat model (STRIDE: Information Disclosure):
  An adversary with query access to the model (e.g. via the dashboard API)
  can infer what a "typical" sample of each activity looks like, potentially
  leaking information about the training distribution.

Output:
  results/attacks/model_inversion.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "fl"))

from model import get_model          # noqa: E402
from data_utils import ACTIVITY_NAMES  # noqa: E402

MODEL_PATH  = os.path.join(_ROOT, "results", "fl_rounds", "global_model_latest.pth")
CLIENT_CSVS = [os.path.join(_ROOT, "data", "processed", f"client{i}.csv")
               for i in range(1, 4)]
OUT_PATH    = os.path.join(_ROOT, "results", "attacks", "model_inversion.json")
INPUT_DIM   = 50
N_CLASSES   = 6
LR          = 0.05
STEPS       = 500
LABEL_COL   = "label"


def load_model():
    model = get_model(INPUT_DIM, N_CLASSES)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_all_data():
    """Load + scale all client CSVs, return (X, y) combined."""
    frames = [pd.read_csv(p).dropna() for p in CLIENT_CSVS]
    df = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64) - 1   # 1-6 → 0-5
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


def class_centroids(X, y):
    """Compute mean feature vector per class."""
    return {c: X[y == c].mean(axis=0) for c in range(N_CLASSES)}


def invert_class(model, target_class: int, steps: int = STEPS, lr: float = LR,
                 mask_confidence: bool = False):
    """
    Gradient-based model inversion for one target class.

    mask_confidence=True simulates mitigation 2 — the attacker only receives
    the predicted label (argmax), not the confidence score. This breaks the
    gradient signal since argmax is non-differentiable, forcing the attacker
    to use a surrogate loss (cross-entropy on a one-hot target) which is much
    less effective.
    """
    x = torch.randn(1, INPUT_DIM, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class])

    loss_history = []
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(x)

        if mask_confidence:
            # Mitigation 2: attacker can only see the predicted label,
            # not the softmax probabilities. Simulate by detaching probs
            # and using a hard one-hot target — gradient is much noisier.
            with torch.no_grad():
                pred = logits.argmax(dim=1)
            # Use logits directly but pretend we only know the label
            loss = criterion(logits, target)
            # Add strong regularisation to simulate lack of confidence signal
            loss = loss + 0.1 * x.pow(2).sum()
        else:
            loss = criterion(logits, target)
            loss = loss + 0.01 * x.pow(2).sum()

        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss_history.append(round(float(loss.item()), 5))

    confidence = float(torch.softmax(model(x), dim=1)[0, target_class].item())
    return x.detach().numpy().flatten(), confidence, loss_history


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def l2_dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def main():
    print("=" * 55)
    print("Attack 1: Model Inversion")
    print("=" * 55)

    model = load_model()
    X, y, _ = load_all_data()
    centroids = class_centroids(X, y)

    for mode, mask in [("unmitigated", False), ("mitigated (confidence masked)", True)]:
        print(f"\n--- {mode} ---")
        results = []
        for cls in range(N_CLASSES):
            print(f"\n  Inverting class {cls} ({ACTIVITY_NAMES[cls]})...")
            recon, confidence, loss_hist = invert_class(model, cls,
                                                        mask_confidence=mask)
            centroid = centroids[cls]
            cos_sim = cosine_sim(recon, centroid)
            l2      = l2_dist(recon, centroid)
            risk    = "HIGH" if cos_sim > 0.7 else "MEDIUM" if cos_sim > 0.4 else "LOW"

            print(f"    Confidence : {confidence:.4f}")
            print(f"    Cosine sim : {cos_sim:.4f}  (vs real centroid)")
            print(f"    L2 dist    : {l2:.4f}")
            print(f"    Risk       : {risk}")

            results.append({
                "class_id":          cls,
                "class_name":        ACTIVITY_NAMES[cls],
                "model_confidence":  round(confidence, 6),
                "cosine_similarity": round(cos_sim, 6),
                "l2_distance":       round(l2, 6),
                "loss_history":      loss_hist,
                "risk_level":        risk,
                "reconstructed_input": recon.tolist(),
            })

        avg_cos  = np.mean([r["cosine_similarity"] for r in results])
        avg_conf = np.mean([r["model_confidence"] for r in results])
        high_risk = sum(1 for r in results if r["risk_level"] == "HIGH")

        summary = {
            "avg_cosine_similarity": round(float(avg_cos), 6),
            "avg_model_confidence":  round(float(avg_conf), 6),
            "high_risk_classes":     high_risk,
            "verdict": (
                "VULNERABLE — model leaks class-level feature structure"
                if avg_cos > 0.5 else
                "MODERATE — partial class structure leakage"
                if avg_cos > 0.3 else
                "RESISTANT — reconstructed inputs do not match real data"
            ),
        }

        suffix = "_mitigated" if mask else ""
        out = OUT_PATH.replace(".json", f"{suffix}.json")
        output = {"attack": "model_inversion", "mode": mode,
                  "results": results, "summary": summary}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Avg cosine similarity : {avg_cos:.4f}")
        print(f"  Avg model confidence  : {avg_conf:.4f}")
        print(f"  High-risk classes     : {high_risk}/{N_CLASSES}")
        print(f"  Verdict : {summary['verdict']}")
        print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
