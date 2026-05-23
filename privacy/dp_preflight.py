"""
privacy/dp_preflight.py

Pre-flight checks before DP federated training.

At ε < 5 a low max_grad_norm (e.g. 1.0) often yields a very high noise
multiplier and ~50% accuracy unless clipping is tuned first.

At ε = 10 (the production default) with max_grad_norm=2.0 the noise
multiplier is typically moderate. The preflight
now surfaces this at ALL epsilon values so the user can decide whether
to tune the clipping norm before training.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional, Tuple

# Warn at ALL epsilon values — even ε=10 has a measurable accuracy cost
# that can be reduced by tuning max_grad_norm.
_EPSILON_WARN_THRESHOLD = float("inf")   # always run the check when --dp is set

# Noise multiplier thresholds for different warning levels:
#   > 3.0 → accuracy collapse likely (exit unless --skip-dp-preflight)
#   > 1.5 → moderate accuracy cost (warn, continue)
#   > 0.8 → light noise (info only)
_NOISE_MULTIPLIER_HARD_FAIL = 3.0
_NOISE_MULTIPLIER_WARN      = 1.5
_NOISE_MULTIPLIER_INFO      = 0.8

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


def sweep_results_path(epsilon: float) -> str:
    return os.path.join(_RESULTS_DIR, f"clipping_norm_sweep_eps{epsilon}.json")


def load_recommended_clipping_norm(epsilon: float) -> Optional[float]:
    """Return best max_grad_norm from a prior clipping_norm_sweep run, if any."""
    path = sweep_results_path(epsilon)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        best = data.get("best") or {}
        norm = best.get("max_grad_norm")
        return float(norm) if norm is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def estimate_noise_multiplier(
    csv_path: str,
    target_epsilon: float,
    max_grad_norm: float,
    local_epochs: int,
    num_fl_rounds: int,
    batch_size: int = 32,
    model_type: str = "mlp",
) -> Tuple[float, float]:
    """
    Attach Opacus once and return (noise_multiplier, target_delta).
    Does not run training.
    """
    _fl_dir = os.path.join(_PROJECT_ROOT, "fl")
    if _fl_dir not in sys.path:
        sys.path.insert(0, _fl_dir)

    from data_utils import load_csv_data  # noqa: E402
    from model import get_model  # noqa: E402
    from opacus import PrivacyEngine  # noqa: E402
    from opacus.validators import ModuleValidator  # noqa: E402
    import torch.optim as optim  # noqa: E402

    train_loader, _, metadata = load_csv_data(csv_path, batch_size=batch_size)
    model = get_model(
        metadata.input_dim, metadata.num_classes, model_type=model_type
    )
    model = ModuleValidator.fix(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    delta = 1.0 / metadata.train_size
    total_epochs = local_epochs * num_fl_rounds

    pe = PrivacyEngine()
    _, opt, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=total_epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )
    return float(opt.noise_multiplier), delta


def run_dp_preflight(
    csv_path: str,
    target_epsilon: float,
    max_grad_norm: float,
    local_epochs: int,
    num_fl_rounds: int,
    *,
    batch_size: int = 32,
    model_type: str = "mlp",
    auto_clipping_sweep: bool = False,
    skip: bool = False,
) -> float:
    """
    Surface the DP noise level before training starts and warn/abort when
    accuracy collapse is likely.

    Always runs (regardless of epsilon) so the user sees the noise multiplier
    at the production epsilon (ε=10) as well as at low epsilon.

    Returns the max_grad_norm to use (may differ after auto-sweep).
    Raises SystemExit(1) on hard failure (noise_multiplier > 3.0) unless
    --skip-dp-preflight is passed.
    """
    if skip:
        print("[DP Preflight] Skipped (--skip-dp-preflight).")
        return max_grad_norm

    recommended = load_recommended_clipping_norm(target_epsilon)
    if recommended is not None and recommended != max_grad_norm:
        print(
            f"[DP Preflight] Prior sweep for ε={target_epsilon} recommends "
            f"max_grad_norm={recommended} (current: {max_grad_norm})."
        )

    try:
        nm, delta = estimate_noise_multiplier(
            csv_path,
            target_epsilon,
            max_grad_norm,
            local_epochs,
            num_fl_rounds,
            batch_size=batch_size,
            model_type=model_type,
        )
    except ImportError:
        print(
            "[DP Preflight] Opacus not installed — skipping preflight. "
            "Install opacus before enabling --dp."
        )
        return max_grad_norm
    except Exception as exc:
        print(f"[DP Preflight] WARNING: calibration failed ({exc}) — continuing.")
        return max_grad_norm

    noise_std = nm * max_grad_norm
    print(
        f"[DP Preflight] ε={target_epsilon}, δ≈{delta:.2e}, "
        f"C={max_grad_norm} → noise_multiplier={nm:.2f}, noise_std≈{noise_std:.2f}"
    )

    # ── Tier 1: hard fail — accuracy collapse certain ─────────────────────────
    if nm > _NOISE_MULTIPLIER_HARD_FAIL:
        print(
            f"\n[DP Preflight] *** ACCURACY COLLAPSE RISK at ε={target_epsilon} ***\n"
            f"  noise_multiplier={nm:.2f} > {_NOISE_MULTIPLIER_HARD_FAIL} with C={max_grad_norm}.\n"
            f"  Typical symptom: test accuracy ~50% (near random for 6 classes).\n"
            f"  Fix: run privacy/clipping_norm_sweep.py --epsilon {target_epsilon} "
            f"--csv {csv_path}\n"
            f"  Then restart with --max-grad-norm <optimal>.\n"
        )

        if auto_clipping_sweep:
            print("[DP Preflight] Running quick clipping norm sweep (reduced norms/epochs)...")
            _privacy_dir = os.path.dirname(__file__)
            if _privacy_dir not in sys.path:
                sys.path.insert(0, _privacy_dir)
            from clipping_norm_sweep import run_sweep  # noqa: E402

            quick_norms = [0.5, 1.0, 2.0, 5.0, 10.0]
            out_path = sweep_results_path(target_epsilon)
            run_sweep(
                csv_path=csv_path,
                epsilon=target_epsilon,
                clipping_norms=quick_norms,
                epochs=3,
                batch_size=batch_size,
                out_path=out_path,
            )
            best_norm = load_recommended_clipping_norm(target_epsilon)
            if best_norm is not None:
                print(f"[DP Preflight] Auto-sweep selected max_grad_norm={best_norm}")
                return best_norm

        if recommended is not None:
            print(
                f"[DP Preflight] Applying saved sweep recommendation: "
                f"max_grad_norm={recommended}"
            )
            return recommended

        raise SystemExit(
            1,
            "DP preflight failed: tune clipping norm before training at this epsilon, "
            "or pass --skip-dp-preflight to override (not recommended).",
        )

    # ── Tier 2: moderate noise — warn but continue ────────────────────────────
    if nm > _NOISE_MULTIPLIER_WARN:
        print(
            f"[DP Preflight] WARNING: noise_multiplier={nm:.2f} is moderate.\n"
            f"  Expected accuracy cost: ~3–8% vs no-DP baseline.\n"
            f"  To reduce: run privacy/clipping_norm_sweep.py --epsilon {target_epsilon} "
            f"--csv {csv_path}\n"
            f"  and restart with --max-grad-norm <optimal>."
        )
        if recommended is not None and recommended != max_grad_norm:
            print(
                f"[DP Preflight] Consider --max-grad-norm {recommended} from prior sweep."
            )
        return max_grad_norm

    # ── Tier 3: light noise — info only ──────────────────────────────────────
    if nm > _NOISE_MULTIPLIER_INFO:
        print(
            f"[DP Preflight] INFO: noise_multiplier={nm:.2f} — "
            f"expect ~1–3% accuracy cost vs no-DP baseline. "
            f"Run clipping_norm_sweep.py to verify this is optimal."
        )
    else:
        print(
            f"[DP Preflight] noise_multiplier={nm:.2f} — "
            f"very light noise; privacy guarantee may be weak at this ε."
        )

    if recommended is not None and recommended != max_grad_norm:
        print(
            f"[DP Preflight] Consider --max-grad-norm {recommended} from prior sweep."
        )
    return max_grad_norm
