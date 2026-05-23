"""
fl/fl_checkpoint.py — Round-level FL server checkpointing for crash recovery.

After each completed round the server writes fl_server_checkpoint.json and
updates global_model_latest.pth. On restart with --resume, training continues
from the last completed round using the saved global weights.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

CHECKPOINT_FILENAME = "fl_server_checkpoint.json"


def checkpoint_path(save_dir: str) -> str:
    return os.path.join(save_dir, CHECKPOINT_FILENAME)


def save_round_checkpoint(
    save_dir: str,
    completed_round: int,
    target_rounds: int,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    strategy_name: str = "fedavg",
    model_type: str = "mlp",
) -> str:
    """Persist checkpoint after a round finishes (fit + evaluate)."""
    os.makedirs(save_dir, exist_ok=True)
    record = {
        "version":            1,
        "completed_round":    int(completed_round),
        "target_rounds":      int(target_rounds),
        "remaining_rounds":   max(0, int(target_rounds) - int(completed_round)),
        "global_model_path":  os.path.join(save_dir, "global_model_latest.pth"),
        "updated_at":                 datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_training_completed_at":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy_name":      strategy_name,
        "model_type":         model_type,
        "last_round_metrics": metrics or {},
        "resume_hint": (
            f"Restart server with --resume --save-dir {save_dir} "
            f"and matching --rounds {target_rounds}"
        ),
    }
    path = checkpoint_path(save_dir)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def load_checkpoint(save_dir: str) -> Optional[Dict[str, Any]]:
    path = checkpoint_path(save_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "completed_round" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def load_initial_parameters(
    save_dir: str,
    input_dim: int,
    num_classes: int,
    model_type: str = "mlp",
):
    """
    Load global_model_latest.pth as Flower Parameters for strategy resume.
    Returns None if no checkpoint model exists.
    """
    import flwr as fl

    pth = os.path.join(save_dir, "global_model_latest.pth")
    sealed = pth + ".sealed"
    if not os.path.isfile(pth) and not os.path.isfile(sealed):
        return None

    _fl_dir = os.path.dirname(os.path.abspath(__file__))
    if _fl_dir not in sys.path:
        sys.path.insert(0, _fl_dir)
    from model import get_model  # noqa: E402

    _sealed_dir = os.path.abspath(os.path.join(_fl_dir, "..", "tee", "sealed_storage"))
    if _sealed_dir not in sys.path:
        sys.path.insert(0, _sealed_dir)
    from sealed_storage import load_checkpoint_state_dict  # noqa: E402

    model = get_model(input_dim, num_classes, model_type=model_type)
    state = load_checkpoint_state_dict(pth, map_location="cpu")
    model.load_state_dict(state, strict=True)
    arrays = [v.cpu().numpy() for v in model.state_dict().values()]
    return fl.common.ndarrays_to_parameters(arrays)


def resolve_resume_plan(
    save_dir: str,
    num_rounds: int,
    resume: bool,
) -> Dict[str, Any]:
    """
    Compute how many rounds to run and optional initial parameters.

    Returns dict with keys:
      run_rounds, initial_parameters, round_offset, checkpoint, resumed
    """
    plan = {
        "run_rounds":           num_rounds,
        "initial_parameters":   None,
        "round_offset":         0,
        "checkpoint":           None,
        "resumed":              False,
    }
    if not resume:
        return plan

    ckpt = load_checkpoint(save_dir)
    if ckpt is None:
        print(f"[Checkpoint] --resume set but no {CHECKPOINT_FILENAME} in {save_dir} — starting fresh.")
        return plan

    completed = int(ckpt.get("completed_round", 0))
    target    = int(ckpt.get("target_rounds", num_rounds))
    if num_rounds != target:
        print(
            f"[Checkpoint] WARNING: --rounds={num_rounds} differs from "
            f"checkpoint target_rounds={target}; using checkpoint target."
        )
        target = target

    remaining = max(0, target - completed)
    if remaining == 0:
        print(f"[Checkpoint] Training already complete ({completed}/{target} rounds).")
        plan["run_rounds"] = 0
        plan["checkpoint"] = ckpt
        plan["resumed"] = True
        plan["round_offset"] = completed
        return plan

    meta_path = os.path.join(save_dir, "model_meta.json")
    model_type = ckpt.get("model_type", "mlp")
    input_dim = num_classes = None
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model_type = meta.get("model_type", model_type)
        input_dim = meta.get("input_dim")
        num_classes = meta.get("num_classes")

    if input_dim is not None and num_classes is not None:
        plan["initial_parameters"] = load_initial_parameters(
            save_dir, input_dim, num_classes, model_type=model_type
        )

    plan["run_rounds"] = remaining
    plan["round_offset"] = completed
    plan["checkpoint"] = ckpt
    plan["resumed"] = True
    print(
        f"[Checkpoint] Resuming after round {completed}/{target} — "
        f"{remaining} round(s) remaining."
    )
    return plan
