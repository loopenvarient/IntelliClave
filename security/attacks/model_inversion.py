"""
security/attacks/model_inversion.py

Attack 1: Model Inversion

Goal: Given only black-box access to the trained FL model, reconstruct a
      representative input for each class by gradient-based optimisation.

Method:
  - Start from random noise in the feature space.
  - Optimise the input to maximise the model's confidence for a target class.
  - Compare reconstructed inputs to real class centroids via cosine similarity
    and L2 distance.

Output:
  results/attacks/model_inversion.json
  results/attacks/model_inversion_mitigated.json
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
from data_utils import infer_csv_schema  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "config"))
from constants import LABEL_COL      # noqa: E402

MODEL_PATH   = os.path.join(_ROOT, "results", "fl_rounds", "global_model_latest.pth")
META_PATH    = os.path.join(_ROOT, "results", "fl_rounds", "model_meta.json")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
OUT_PATH     = os.path.join(_ROOT, "results", "attacks", "model_inversion.json")
DEFAULT_LR    = 0.05
DEFAULT_STEPS = 500


def load_meta(label_col: str = LABEL_COL):
    """Load model metadata (input_dim, num_classes, class_names)."""
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    # Fallback: infer from first CSV
    csvs = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv"))
    if not csvs:
        raise FileNotFoundError("No model_meta.json and no CSVs in data/processed/")
    first_csv = os.path.join(PROCESSED_DIR, csvs[0])
    input_dim, _ = infer_csv_schema(first_csv, label_col)
    df = pd.read_csv(first_csv, usecols=[label_col])
    num_classes = int(df[label_col].nunique())
    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": [f"class_{i}" for i in range(num_classes)],
    }


def load_model(input_dim, num_classes, model_path: str = MODEL_PATH):
    model = get_model(input_dim, num_classes)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_all_data(input_dim, label_col: str = LABEL_COL):
    """Load + scale all client CSVs, return (X, y) combined."""
    csvs = sorted(
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    )
    frames = [pd.read_csv(p).dropna() for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols].values.astype(np.float32)
    raw_y = df[label_col].values
    unique = sorted(np.unique(raw_y).tolist())
    offset = int(min(unique)) if all(isinstance(v, (int, float)) for v in unique) else 0
    y = raw_y.astype(np.int64) - offset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def class_centroids(X, y, num_classes):
    return {c: X[y == c].mean(axis=0) for c in range(num_classes)}


def invert_class(model, target_class, input_dim, steps=DEFAULT_STEPS, lr=DEFAULT_LR,
                 mask_confidence=False):
    x = torch.randn(1, input_dim, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class])
    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(x)
        if mask_confidence:
            loss = criterion(logits, target) + 0.1 * x.pow(2).sum()
        else:
            loss = criterion(logits, target) + 0.01 * x.pow(2).sum()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss_history.append(round(float(loss.item()), 5))

    confidence = float(torch.softmax(model(x), dim=1)[0, target_class].item())
    return x.detach().numpy().flatten(), confidence, loss_history


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def l2_dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def main(model_path: str = MODEL_PATH, label_col: str = LABEL_COL,
         lr: float = DEFAULT_LR, steps: int = DEFAULT_STEPS,
         out_path: str = OUT_PATH):
    print("=" * 55)
    print("Attack 1: Model Inversion")
    print("=" * 55)

    meta = load_meta(label_col=label_col)
    input_dim   = meta["input_dim"]
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    model = load_model(input_dim, num_classes, model_path=model_path)
    X, y  = load_all_data(input_dim, label_col=label_col)
    centroids = class_centroids(X, y, num_classes)

    for mode, mask in [("unmitigated", False), ("mitigated (confidence masked)", True)]:
        print(f"\n--- {mode} ---")
        results = []
        for cls in range(num_classes):
            name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            print(f"\n  Inverting class {cls} ({name})...")
            recon, confidence, loss_hist = invert_class(
                model, cls, input_dim, steps=steps, lr=lr,
                mask_confidence=mask
            )
            centroid = centroids.get(cls, np.zeros(input_dim))
            cos_sim  = cosine_sim(recon, centroid)
            l2       = l2_dist(recon, centroid)
            risk     = "HIGH" if cos_sim > 0.7 else "MEDIUM" if cos_sim > 0.4 else "LOW"

            print(f"    Confidence : {confidence:.4f}")
            print(f"    Cosine sim : {cos_sim:.4f}")
            print(f"    L2 dist    : {l2:.4f}")
            print(f"    Risk       : {risk}")

            results.append({
                "class_id":           cls,
                "class_name":         name,
                "model_confidence":   round(confidence, 6),
                "cosine_similarity":  round(cos_sim, 6),
                "l2_distance":        round(l2, 6),
                "loss_history":       loss_hist,
                "risk_level":         risk,
                "reconstructed_input": recon.tolist(),
            })

        avg_cos   = np.mean([r["cosine_similarity"] for r in results])
        avg_conf  = np.mean([r["model_confidence"] for r in results])
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
        out = out_path.replace(".json", f"{suffix}.json")
        output = {"attack": "model_inversion", "mode": mode,
                  "results": results, "summary": summary}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Avg cosine similarity : {avg_cos:.4f}")
        print(f"  Avg model confidence  : {avg_conf:.4f}")
        print(f"  High-risk classes     : {high_risk}/{num_classes}")
        print(f"  Verdict : {summary['verdict']}")
        print(f"  Saved → {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Model Inversion Attack against a trained FL model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=MODEL_PATH,
                        help="Path to the trained model .pth file.")
    parser.add_argument("--label-col",  default=LABEL_COL,
                        help="Name of the label column in client CSVs.")
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR,
                        help="Optimisation learning rate for input reconstruction.")
    parser.add_argument("--steps",      type=int,   default=DEFAULT_STEPS,
                        help="Number of gradient steps per class inversion.")
    parser.add_argument("--out",        default=OUT_PATH,
                        help="Output JSON path (suffix _mitigated.json also written).")
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        label_col=args.label_col,
        lr=args.lr,
        steps=args.steps,
        out_path=args.out,
    )
