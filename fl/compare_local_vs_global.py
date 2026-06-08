"""
fl/compare_local_vs_global.py

Train (or load) per-client local-only baselines and compare them against the
federated global model on each client's held-out test split.

Writes: results/local_vs_global.json  (read by dashboard GET /comparison)

Usage:
    python fl/compare_local_vs_global.py --dp --epsilon 10 --retrain
    python fl/compare_local_vs_global.py --epochs 20 --retrain
    python fl/compare_local_vs_global.py --checkpoint results/fl_rounds/run_*/global_model_latest.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import (  # noqa: E402
    get_default_client_csvs,
    get_global_normalization_path,
    load_class_weights,
    load_csv_data,
    load_global_normalization_file,
)
from evaluate_global_model import _load_checkpoint, evaluate_checkpoint  # noqa: E402
from model import build_model_from_state  # noqa: E402
from train_local import evaluate, train_local  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(ROOT, "results", "local_vs_global.json")
LOCAL_DIR = os.path.join(ROOT, "results", "local_baselines")
PCA_MODEL_PATH = os.path.join(ROOT, "data", "samples", "pca_model.pkl")


def _find_latest_model_path() -> str:
    fl_rounds_dir = os.path.join(ROOT, "results", "fl_rounds")
    if not os.path.isdir(fl_rounds_dir):
        return os.path.join(fl_rounds_dir, "global_model_latest.pth")

    candidates: List[str] = []
    for entry in os.scandir(fl_rounds_dir):
        if entry.is_dir() and entry.name.startswith("run_"):
            p = os.path.join(entry.path, "global_model_latest.pth")
            if os.path.exists(p):
                candidates.append(p)
    flat = os.path.join(fl_rounds_dir, "global_model_latest.pth")
    if os.path.exists(flat):
        candidates.append(flat)
    if not candidates:
        raise FileNotFoundError("No global_model_latest.pth found under results/fl_rounds/")
    return max(candidates, key=os.path.getmtime)


def _load_global_norm(checkpoint_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    run_dir = os.path.dirname(checkpoint_path)
    norm_path = get_global_normalization_path(run_dir)
    if os.path.exists(norm_path):
        return load_global_normalization_file(norm_path)
    return None, None


def _local_checkpoint(client_id: int) -> str:
    return os.path.join(LOCAL_DIR, f"client{client_id}_local.pth")


def _eval_local_checkpoint(
    checkpoint_path: str,
    csv_path: str,
    batch_size: int,
    model_type: str,
) -> Tuple[float, float, float, int]:
    """Evaluate a saved local model on its client test split (per-client normalization)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, meta = load_csv_data(
        csv_path,
        batch_size=batch_size,
        pca_model_path=PCA_MODEL_PATH,
    )

    state = _load_checkpoint(checkpoint_path)
    model = build_model_from_state(
        meta.input_dim,
        meta.num_classes,
        model_type=model_type,
        state=state,
    ).to(device)
    model.load_state_dict(state, strict=True)

    criterion = nn.CrossEntropyLoss(
        weight=load_class_weights(num_classes=meta.num_classes, device=device)
    )
    total_loss, n_examples = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(x_batch.to(device))
            loss = criterion(logits, y_batch.to(device))
            total_loss += loss.item() * len(y_batch)
            n_examples += len(y_batch)

    accuracy, macro_f1 = evaluate(model, test_loader, device)
    return round(accuracy, 5), round(macro_f1, 5), round(total_loss / max(n_examples, 1), 5), meta.test_size


def _train_or_load_local(
    client_id: int,
    csv_path: str,
    epochs: int,
    batch_size: int,
    model_type: str,
    use_dp: bool,
    target_epsilon: float,
    retrain: bool,
) -> Tuple[float, float, float, int, str]:
    save_path = _local_checkpoint(client_id)
    if os.path.exists(save_path) and not retrain:
        print(f"[compare] Client {client_id}: loading cached local model")
        acc, f1, loss, test_size = _eval_local_checkpoint(
            save_path, csv_path, batch_size, model_type
        )
        return acc, f1, loss, test_size, save_path

    print(f"[compare] Client {client_id}: training local baseline ({epochs} epochs)")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    model, history, meta = train_local(
        csv_path=csv_path,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        model_type=model_type,
        use_dp=use_dp,
        target_epsilon=target_epsilon,
        early_stopping_patience=3,
        early_stopping_metric="macro_f1",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _ = load_csv_data(csv_path, batch_size=batch_size, pca_model_path=PCA_MODEL_PATH)
    accuracy, macro_f1 = evaluate(model.to(device), test_loader, device)
    last = history[-1] if history else {}
    return (
        round(float(accuracy), 5),
        round(float(macro_f1), 5),
        round(float(last.get("loss", 0)), 5),
        meta.test_size,
        save_path,
    )


def compare_local_vs_global(
    checkpoint_path: Optional[str] = None,
    csv_paths: Optional[List[str]] = None,
    epochs: int = 20,
    batch_size: int = 32,
    model_type: str = "mlp",
    use_dp: bool = False,
    target_epsilon: float = 10.0,
    retrain: bool = False,
    out_path: str = DEFAULT_OUT,
) -> Dict:
    checkpoint_path = checkpoint_path or _find_latest_model_path()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Global checkpoint not found: {checkpoint_path}")

    csv_paths = csv_paths or get_default_client_csvs()
    global_mean, global_std = _load_global_norm(checkpoint_path)

    meta_path = os.path.join(os.path.dirname(checkpoint_path), "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            model_type = json.load(f).get("model_type", model_type)

    global_eval = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        csv_paths=csv_paths,
        batch_size=batch_size,
        model_type=model_type,
        global_mean=global_mean,
        global_std=global_std,
    )
    global_by_csv = {row["client_csv"]: row for row in global_eval["per_client"]}

    clients_out = []
    total_test = 0
    w_local_acc = w_local_f1 = 0.0
    w_global_acc = w_global_f1 = 0.0
    wins = 0

    for i, csv_path in enumerate(csv_paths):
        client_id = i + 1
        local_acc, local_f1, local_loss, test_size, local_ckpt = _train_or_load_local(
            client_id=client_id,
            csv_path=csv_path,
            epochs=epochs,
            batch_size=batch_size,
            model_type=model_type,
            use_dp=use_dp,
            target_epsilon=target_epsilon,
            retrain=retrain,
        )
        g_row = global_by_csv.get(csv_path, {})
        global_acc = float(g_row.get("accuracy", 0))
        global_f1 = float(g_row.get("macro_f1", 0))
        gain_acc = round(global_acc - local_acc, 5)
        gain_f1 = round(global_f1 - local_f1, 5)
        if gain_f1 > 0:
            wins += 1

        clients_out.append({
            "client_id": client_id,
            "client_name": f"Client {client_id}",
            "client_csv": os.path.basename(csv_path),
            "test_size": test_size,
            "local": {
                "accuracy": local_acc,
                "macro_f1": local_f1,
                "loss": local_loss,
                "checkpoint": os.path.relpath(local_ckpt, ROOT).replace("\\", "/"),
                "normalization": "per_client",
            },
            "global": {
                "accuracy": round(global_acc, 5),
                "macro_f1": round(global_f1, 5),
                "loss": g_row.get("loss"),
                "checkpoint": os.path.relpath(checkpoint_path, ROOT).replace("\\", "/"),
                "normalization": "global" if global_mean is not None else "per_client",
            },
            "gain": {
                "accuracy": gain_acc,
                "macro_f1": gain_f1,
                "accuracy_pct": round(gain_acc * 100, 2),
                "macro_f1_pct": round(gain_f1 * 100, 2),
            },
        })

        total_test += test_size
        w_local_acc += local_acc * test_size
        w_local_f1 += local_f1 * test_size
        w_global_acc += global_acc * test_size
        w_global_f1 += global_f1 * test_size

    summary = {
        "local_weighted": {
            "accuracy": round(w_local_acc / total_test, 5),
            "macro_f1": round(w_local_f1 / total_test, 5),
        },
        "global_weighted": {
            "accuracy": round(w_global_acc / total_test, 5),
            "macro_f1": round(w_global_f1 / total_test, 5),
        },
        "avg_gain": {
            "accuracy": round((w_global_acc - w_local_acc) / total_test, 5),
            "macro_f1": round((w_global_f1 - w_local_f1) / total_test, 5),
            "accuracy_pct": round((w_global_acc - w_local_acc) / total_test * 100, 2),
            "macro_f1_pct": round((w_global_f1 - w_local_f1) / total_test * 100, 2),
        },
        "clients_where_global_wins_f1": wins,
        "total_clients": len(csv_paths),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "global_checkpoint": os.path.relpath(checkpoint_path, ROOT).replace("\\", "/"),
        "local_training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "model_type": model_type,
            "use_dp": use_dp,
            "target_epsilon": target_epsilon if use_dp else None,
            "baseline_dir": os.path.relpath(LOCAL_DIR, ROOT).replace("\\", "/"),
        },
        "clients": clients_out,
        "summary": summary,
        "note": (
            "Local models train only on each hospital's data (per-client normalization)"
            + (" with DP-SGD." if use_dp else " without DP.")
            + " Global model is the federated FL+DP checkpoint (global normalization). "
            "Positive delta F1 means collaboration beat solo training under comparable privacy."
        ),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}")
    print("Local vs Global FL Comparison")
    print(f"{'='*60}")
    for row in clients_out:
        print(
            f"  {row['client_name']}: "
            f"local F1={row['local']['macro_f1']:.3f}  "
            f"global F1={row['global']['macro_f1']:.3f}  "
            f"gain={row['gain']['macro_f1_pct']:+.1f}pp"
        )
    print(
        f"\n  Weighted: local F1={summary['local_weighted']['macro_f1']:.3f}  "
        f"global F1={summary['global_weighted']['macro_f1']:.3f}  "
        f"avg gain={summary['avg_gain']['macro_f1_pct']:+.2f}pp"
    )
    print(f"\nSaved -> {out_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare local-only vs federated global models.")
    parser.add_argument("--checkpoint", default=None, help="Path to global_model_latest.pth")
    parser.add_argument("--epochs", type=int, default=20, help="Local baseline training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "resnet-tabular", "transformer-tabular"])
    parser.add_argument("--dp", action="store_true", help="Train local baselines with DP-SGD")
    parser.add_argument("--epsilon", type=float, default=10.0, help="DP epsilon for local baselines")
    parser.add_argument("--retrain", action="store_true", help="Retrain local baselines even if cached")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    compare_local_vs_global(
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        retrain=args.retrain,
        out_path=args.out,
    )
