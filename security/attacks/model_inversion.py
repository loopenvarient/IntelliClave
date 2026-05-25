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

Three modes are tested:
  1. unmitigated          — bare model, no defences
  2. confidence masked    — legacy mitigation (masking only), kept for comparison
  3. privacy wrapper      — new PrivacyWrapper (Laplace noise + temperature scaling)
                            uses MI_NOISE_SCALE / MI_TEMPERATURE from constants.py

Output:
  results/attacks/model_inversion.json           (unmitigated)
  results/attacks/model_inversion_mitigated.json (confidence masked)
  results/attacks/model_inversion_defended.json  (PrivacyWrapper — new)
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
sys.path.insert(0, os.path.join(_ROOT, "config"))

from model import get_model, get_defended_model   # noqa: E402
from data_utils import infer_csv_schema            # noqa: E402
from constants import LABEL_COL, MI_NOISE_SCALE, MI_TEMPERATURE  # noqa: E402

MODEL_PATH    = os.path.join(_ROOT, "results", "fl_rounds", "global_model_latest.pth")
META_PATH     = os.path.join(_ROOT, "results", "fl_rounds", "model_meta.json")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
OUT_PATH      = os.path.join(_ROOT, "results", "attacks", "model_inversion.json")
DEFAULT_LR    = 0.05
DEFAULT_STEPS = 500


def _strip_state_prefixes(state):
    if not isinstance(state, dict):
        return state
    if any(key.startswith("_module.") for key in state):
        return {key.removeprefix("_module."): value for key, value in state.items()}
    if any(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _load_checkpoint(model_path: str):
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return _strip_state_prefixes(state)


def _infer_hidden_dims(state: dict, model_type: str):
    if model_type == "mlp":
        layers = []
        layer_idx = 0
        while True:
            weight_key = f"feature_extractor.{layer_idx}.weight"
            if weight_key not in state:
                break
            layers.append(int(state[weight_key].shape[0]))
            layer_idx += 3
        return tuple(layers) if layers else None
    if model_type == "resnet-tabular" and "input_proj.0.weight" in state:
        return (int(state["input_proj.0.weight"].shape[0]),)
    return None


def load_meta(label_col: str = LABEL_COL):
    return load_meta_for_path(MODEL_PATH, label_col=label_col)


def load_meta_for_path(model_path: str, label_col: str = LABEL_COL):
    model_dir = os.path.dirname(os.path.abspath(model_path))
    sibling_meta = os.path.join(model_dir, "model_meta.json")
    if os.path.exists(sibling_meta):
        with open(sibling_meta, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    csvs = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv"))
    if not csvs:
        raise FileNotFoundError("No model_meta.json and no CSVs in data/processed/")
    first_csv = os.path.join(PROCESSED_DIR, csvs[0])
    input_dim, _ = infer_csv_schema(first_csv, label_col)
    df = pd.read_csv(first_csv, usecols=[label_col])
    num_classes = int(df[label_col].nunique())
    return {
        "input_dim":   input_dim,
        "num_classes": num_classes,
        "class_names": [f"class_{i}" for i in range(num_classes)],
    }


def load_model(input_dim, num_classes, model_path: str = MODEL_PATH):
    """Load the bare (undefended) model for unmitigated and masking modes."""
    state = _load_checkpoint(model_path)
    meta = load_meta_for_path(model_path)
    model_type = meta.get("model_type", "mlp")
    hidden_dims = _infer_hidden_dims(state, model_type)
    model = get_model(input_dim, num_classes, model_type=model_type, hidden_dims=hidden_dims)
    model.load_state_dict(state)
    model.eval()
    return model


def load_defended(input_dim, num_classes, model_path: str = MODEL_PATH,
                  noise_scale: float = MI_NOISE_SCALE,
                  temperature: float = MI_TEMPERATURE):
    """
    Load the PrivacyWrapper-defended model.
    Weights are loaded into wrapper.base_model so Opacus checkpoints
    (which only contain the inner model's state_dict) load correctly.
    """
    wrapper = get_defended_model(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type=load_meta_for_path(model_path).get("model_type", "mlp"),
        hidden_dims=_infer_hidden_dims(_load_checkpoint(model_path), load_meta_for_path(model_path).get("model_type", "mlp")),
        noise_scale=noise_scale,
        temperature=temperature,
        enabled=True,
    )
    state = _load_checkpoint(model_path)
    wrapper.base_model.load_state_dict(state)
    wrapper.eval()
    return wrapper


def load_all_data(_input_dim, label_col: str = LABEL_COL):
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
    """
    Gradient-based input reconstruction for the target class.

    For the PrivacyWrapper-defended model the inversion loop still runs, but
    the noise injected at each forward pass corrupts the gradient signal —
    the optimizer cannot climb toward a clean class prototype.
    """
    x = torch.randn(1, input_dim, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class])
    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()
        out = model(x)

        # For PrivacyWrapper the output is already softmax probabilities.
        # CrossEntropyLoss expects logits, so we convert back via log.
        # For bare models out is raw logits — no conversion needed.
        if out.min() >= 0 and out.max() <= 1 and abs(out.sum().item() - 1.0) < 0.05:
            # Looks like probabilities — convert to pseudo-logits for CE loss
            logit_like = torch.log(out + 1e-9)
            if mask_confidence:
                loss = criterion(logit_like, target) + 0.1 * x.pow(2).sum()
            else:
                loss = criterion(logit_like, target) + 0.01 * x.pow(2).sum()
        else:
            if mask_confidence:
                loss = criterion(out, target) + 0.1 * x.pow(2).sum()
            else:
                loss = criterion(out, target) + 0.01 * x.pow(2).sum()

        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss_history.append(round(float(loss.item()), 5))

    # Measure confidence from the model's actual output
    with torch.no_grad():
        final_out = model(x)
    if final_out.min() >= 0 and final_out.max() <= 1 and abs(final_out.sum().item() - 1.0) < 0.05:
        confidence = float(final_out[0, target_class].item())
    else:
        confidence = float(torch.softmax(final_out, dim=1)[0, target_class].item())

    return x.detach().numpy().flatten(), confidence, loss_history


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def l2_dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _prediction_metrics(final_out: torch.Tensor):
    if final_out.min() >= 0 and final_out.max() <= 1 and abs(final_out.sum().item() - 1.0) < 0.05:
        probs = final_out[0]
    else:
        probs = torch.softmax(final_out, dim=1)[0]
    probs = torch.clamp(probs, min=1e-9)
    entropy = float(-(probs * torch.log(probs)).sum().item())
    top2 = torch.topk(probs, k=min(2, probs.numel())).values
    sharpness = float((top2[0] - top2[1]).item()) if top2.numel() > 1 else float(top2[0].item())
    confidence = float(probs.max().item())
    return confidence, entropy, sharpness


def run_mode(model, mode_label, num_classes, class_names, input_dim, centroids,
             steps, lr, mask_confidence, out_path):
    print(f"\n--- {mode_label} ---")
    results = []
    for cls in range(num_classes):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"\n  Inverting class {cls} ({name})...")
        recon, confidence, loss_hist = invert_class(
            model, cls, input_dim, steps=steps, lr=lr,
            mask_confidence=mask_confidence,
        )
        centroid = centroids.get(cls, np.zeros(input_dim))
        cos_sim  = cosine_sim(recon, centroid)
        l2       = l2_dist(recon, centroid)
        risk     = "HIGH" if cos_sim > 0.6 else "MEDIUM" if cos_sim > 0.35 else "LOW"

        with torch.no_grad():
            final_out = model(torch.tensor(recon, dtype=torch.float32).unsqueeze(0))
        confidence, entropy, sharpness = _prediction_metrics(final_out)
        recon_variance = float(np.var(recon))

        print(f"    Confidence : {confidence:.4f}")
        print(f"    Entropy    : {entropy:.4f}")
        print(f"    Sharpness  : {sharpness:.4f}")
        print(f"    Recon var  : {recon_variance:.4f}")
        print(f"    Cosine sim : {cos_sim:.4f}")
        print(f"    L2 dist    : {l2:.4f}")
        print(f"    Risk       : {risk}")

        results.append({
            "class_id":            cls,
            "class_name":          name,
            "model_confidence":    round(confidence, 6),
            "prediction_entropy":  round(entropy, 6),
            "prediction_sharpness": round(sharpness, 6),
            "reconstruction_variance": round(recon_variance, 6),
            "cosine_similarity":   round(cos_sim, 6),
            "l2_distance":         round(l2, 6),
            "loss_history":        loss_hist,
            "risk_level":          risk,
            "reconstructed_input": recon.tolist(),
        })

    avg_cos   = np.mean([r["cosine_similarity"] for r in results])
    avg_conf  = np.mean([r["model_confidence"]  for r in results])
    avg_entropy = np.mean([r["prediction_entropy"] for r in results])
    avg_sharpness = np.mean([r["prediction_sharpness"] for r in results])
    avg_recon_var = np.mean([r["reconstruction_variance"] for r in results])
    high_risk = sum(1 for r in results if r["risk_level"] == "HIGH")

    summary = {
        "avg_cosine_similarity": round(float(avg_cos), 6),
        "avg_model_confidence":  round(float(avg_conf), 6),
        "avg_prediction_entropy": round(float(avg_entropy), 6),
        "avg_prediction_sharpness": round(float(avg_sharpness), 6),
        "avg_reconstruction_variance": round(float(avg_recon_var), 6),
        "high_risk_classes":     high_risk,
        "verdict": (
            "VULNERABLE — model leaks class-level feature structure"
            if avg_cos > 0.4 else
            "MODERATE — partial class structure leakage"
            if avg_cos > 0.25 else
            "RESISTANT — reconstructed inputs do not match real data"
        ),
    }

    output = {"attack": "model_inversion", "mode": mode_label,
              "results": results, "summary": summary}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Avg cosine similarity : {avg_cos:.4f}")
    print(f"  Avg model confidence  : {avg_conf:.4f}")
    print(f"  High-risk classes     : {high_risk}/{num_classes}")
    print(f"  Verdict : {summary['verdict']}")
    print(f"  Saved → {out_path}")
    return summary


def main(model_path: str = MODEL_PATH, label_col: str = LABEL_COL,
         lr: float = DEFAULT_LR, steps: int = DEFAULT_STEPS,
         out_path: str = OUT_PATH,
         noise_scale: float = MI_NOISE_SCALE,
         temperature: float = MI_TEMPERATURE):

    print("=" * 55)
    print("Attack 1: Model Inversion")
    print("=" * 55)

    meta        = load_meta_for_path(model_path, label_col=label_col)
    input_dim   = meta["input_dim"]
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    bare_model = load_model(input_dim, num_classes, model_path=model_path)
    X, y       = load_all_data(input_dim, label_col=label_col)
    centroids  = class_centroids(X, y, num_classes)

    # ── Mode 1: unmitigated ───────────────────────────────────────────────────
    run_mode(
        model=bare_model,
        mode_label="unmitigated",
        num_classes=num_classes,
        class_names=class_names,
        input_dim=input_dim,
        centroids=centroids,
        steps=steps, lr=lr,
        mask_confidence=False,
        out_path=out_path,
    )

    # ── Mode 2: legacy confidence masking (kept for comparison) ───────────────
    run_mode(
        model=bare_model,
        mode_label="mitigated (confidence masked)",
        num_classes=num_classes,
        class_names=class_names,
        input_dim=input_dim,
        centroids=centroids,
        steps=steps, lr=lr,
        mask_confidence=True,
        out_path=out_path.replace(".json", "_mitigated.json"),
    )

    # ── Mode 3: PrivacyWrapper (new defence) ──────────────────────────────────
    print(f"\n  [Defence params] noise_scale={noise_scale}  temperature={temperature}")
    defended_model = load_defended(
        input_dim, num_classes, model_path=model_path,
        noise_scale=noise_scale, temperature=temperature,
    )
    run_mode(
        model=defended_model,
        mode_label=f"defended (noise={noise_scale}, temp={temperature})",
        num_classes=num_classes,
        class_names=class_names,
        input_dim=input_dim,
        centroids=centroids,
        steps=steps, lr=lr,
        mask_confidence=False,
        out_path=out_path.replace(".json", "_defended.json"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Model Inversion Attack against a trained FL model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path",   default=MODEL_PATH)
    parser.add_argument("--label-col",    default=LABEL_COL)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--steps",        type=int,   default=DEFAULT_STEPS)
    parser.add_argument("--out",          default=OUT_PATH)
    parser.add_argument("--noise-scale",  type=float, default=MI_NOISE_SCALE,
                        help="Laplace noise scale for PrivacyWrapper defence.")
    parser.add_argument("--temperature",  type=float, default=MI_TEMPERATURE,
                        help="Softmax temperature for PrivacyWrapper defence.")
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        label_col=args.label_col,
        lr=args.lr,
        steps=args.steps,
        out_path=args.out,
        noise_scale=args.noise_scale,
        temperature=args.temperature,
    )