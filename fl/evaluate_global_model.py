"""
fl/evaluate_global_model.py

Evaluate a saved global FL model checkpoint against all client test sets.

Scaler correctness
------------------
Each client's test data is scaled using statistics computed from that client's
training split only — matching exactly what happened during FL training.
This avoids the data leakage that would occur if the scaler were fitted on
the full client dataset (train + test combined).
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import get_default_client_csvs, load_class_weights, load_csv_data  # noqa: E402
from model import get_model  # noqa: E402
from train_local import evaluate  # noqa: E402


def _strip_state_prefixes(state):
    if not isinstance(state, dict):
        return state
    if any(key.startswith("_module.") for key in state):
        return {key.removeprefix("_module."): value for key, value in state.items()}
    if any(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _load_checkpoint(checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location="cpu")
    # common wrappers: {'state_dict': {...}} or raw state dict
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return _strip_state_prefixes(state)


def _infer_hidden_dims_from_state(state: dict, model_type: str):
    if not isinstance(state, dict):
        return None
    if model_type == "mlp":
        layers = []
        idx = 0
        # feature_extractor layer blocks are Linear, ReLU, Dropout → weights at indices 0,3,6...
        while True:
            key = f"feature_extractor.{idx}.weight"
            if key not in state:
                break
            layers.append(int(state[key].shape[0]))
            idx += 3
        return tuple(layers) if layers else None
    if model_type == "resnet-tabular":
        if "input_proj.0.weight" in state:
            return (int(state["input_proj.0.weight"].shape[0]),)
    return None


def evaluate_checkpoint(
    checkpoint_path: str,
    csv_paths: List[str],
    batch_size: int = 64,
    model_type: str = "mlp",
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
) -> Dict:
    """
    Evaluate a global model checkpoint on each client's test split.

    Parameters
    ----------
    checkpoint_path : path to the .pth model file
    csv_paths       : list of client CSV paths
    batch_size      : DataLoader batch size
    model_type      : architecture used during training
    global_mean     : if global normalization was used during training,
                      pass the same global_mean here for consistent scaling
    global_std      : same as global_mean — must match training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load schema and model_type from first CSV + model_meta.json
    meta_path = os.path.join(os.path.dirname(checkpoint_path), "model_meta.json")
    saved_model_type = model_type  # use caller's value as default
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            saved_meta = json.load(f)
        saved_model_type = saved_meta.get("model_type", model_type)
        if saved_model_type != model_type:
            print(f"[evaluate] NOTE: model_meta.json says model_type='{saved_model_type}', "
                  f"overriding --model-type='{model_type}'")
            model_type = saved_model_type

    _, _, metadata = load_csv_data(
        csv_paths[0],
        batch_size=batch_size,
        global_mean=global_mean,
        global_std=global_std,
    )

    # Load checkpoint and infer hidden dims (for compatibility with older checkpoints)
    state_dict = _load_checkpoint(checkpoint_path)
    hidden_dims = _infer_hidden_dims_from_state(state_dict, model_type)
    model = get_model(
        metadata.input_dim,
        metadata.num_classes,
        model_type=model_type,
        hidden_dims=hidden_dims,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    criterion = nn.CrossEntropyLoss(
        weight=load_class_weights(num_classes=metadata.num_classes, device=device)
    )

    results        = []
    total_examples = 0
    weighted_loss  = 0.0
    weighted_acc   = 0.0
    weighted_f1    = 0.0

    for csv_path in csv_paths:
        # load_csv_data fits the scaler on the training split only (no leakage)
        # The test_loader uses the scaler fitted on train — same as during FL training
        _, test_loader, client_meta = load_csv_data(
            csv_path,
            batch_size=batch_size,
            global_mean=global_mean,
            global_std=global_std,
        )

        total_loss = 0.0
        n_examples = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch.to(device))
                loss   = criterion(logits, y_batch.to(device))
                total_loss += loss.item() * len(y_batch)
                n_examples += len(y_batch)

        accuracy, macro_f1 = evaluate(model, test_loader, device)
        avg_loss = total_loss / n_examples

        results.append({
            "client_csv": csv_path,
            "test_size":  client_meta.test_size,
            "loss":       round(avg_loss, 5),
            "accuracy":   round(accuracy, 5),
            "macro_f1":   round(macro_f1, 5),
        })

        total_examples += n_examples
        weighted_loss  += avg_loss * n_examples
        weighted_acc   += accuracy * n_examples
        weighted_f1    += macro_f1 * n_examples

    summary = {
        "checkpoint": checkpoint_path,
        "per_client": results,
        "overall_weighted": {
            "loss":      round(weighted_loss / total_examples, 5),
            "accuracy":  round(weighted_acc   / total_examples, 5),
            "macro_f1":  round(weighted_f1    / total_examples, 5),
        },
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the saved global FL model on client CSVs."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("results", "fl_rounds", "global_model_latest.pth"),
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output",
        default=os.path.join("results", "fl_rounds", "global_model_eval.json"),
        help="Path to write the evaluation JSON.",
    )
    parser.add_argument(
        "--model-type", default="mlp",
        choices=["mlp", "resnet-tabular", "transformer-tabular"],
        help="Model architecture used during training (default: mlp).",
    )
    parser.add_argument(
        "--csvs", nargs="*", default=None,
        help="Client CSV paths. Defaults to data/processed/client1.csv etc.",
    )
    args = parser.parse_args()

    csv_paths = args.csvs or get_default_client_csvs()
    summary   = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        csv_paths=csv_paths,
        batch_size=args.batch_size,
        model_type=args.model_type,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Checkpoint: {summary['checkpoint']}")
    for item in summary["per_client"]:
        print(
            f"  {os.path.basename(item['client_csv'])}: "
            f"loss={item['loss']:.5f}  "
            f"accuracy={item['accuracy']:.5f}  "
            f"macro_f1={item['macro_f1']:.5f}"
        )
    overall = summary["overall_weighted"]
    print(
        f"Overall weighted: loss={overall['loss']:.5f}  "
        f"accuracy={overall['accuracy']:.5f}  "
        f"macro_f1={overall['macro_f1']:.5f}"
    )
    print(f"Saved -> {args.output}")
