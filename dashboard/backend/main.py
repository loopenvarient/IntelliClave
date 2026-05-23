"""
IntelliClave Dashboard — FastAPI backend

Fixes applied:
  - Model cache is invalidated when global_model_latest.pth changes on disk
    (mtime check on every request — no stale predictions after retraining).
  - /status and /results endpoints now work: they read from status.json and
    results/results.json which are written by run_fl_simulation.py / fl_server.py.
  - Model path discovery scans all timestamped run subdirectories and picks
    the most recently modified global_model_latest.pth.
  - CORS allowed origins are configurable via the CORS_ORIGINS env variable
    (comma-separated). Defaults to http://localhost:3000 for development.
"""
import json
import os
import sys
import time
from typing import List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── model + inference protection imports ─────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'fl'))
sys.path.insert(0, os.path.join(ROOT, 'security'))
from config.runtime_paths import read_json_runtime, resolve_runtime_file  # noqa: E402
from model import get_model  # noqa: E402
from inference_protection import (  # noqa: E402
    OutputPerturbation,
    BudgetExhaustedError,
    RateLimitExceeded,
    LIFETIME_BUDGET_DEFAULT,
    NOISE_SCALE_DEFAULT,
    create_budget_store,
    create_rate_limiter,
)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="IntelliClave Dashboard API", version="1.0.0")

# ── CORS — configurable via environment variable ──────────────────────────────
# Set CORS_ORIGINS="https://app.example.com,https://admin.example.com" in prod.
_cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# ─────────────────────────────────────────────────────────────────────────────

# ── Inference-time protections (Issue 1 fix) ──────────────────────────────────
# OutputPerturbation: Laplace noise on post-softmax probabilities.
#   noise_scale = L1_sensitivity / epsilon_inf = 2.0 / 4.0 = 0.5 (default)
#   Override via INFERENCE_NOISE_SCALE or INFERENCE_EPSILON env vars.
_output_perturbation = OutputPerturbation(noise_scale=NOISE_SCALE_DEFAULT)

# Lifetime query budget + rate limiter — Redis when REDIS_URL is set (required
# in docker-compose / K8s via REQUIRE_REDIS=true).
RATE_LIMIT_MAX    = int(os.environ.get("RATE_LIMIT_MAX", "100"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
_require_redis    = os.environ.get("REQUIRE_REDIS", "").lower() in ("1", "true", "yes")
_redis_url        = os.environ.get("REDIS_URL", "").strip()

_budget_store = create_budget_store(require_redis=_require_redis)
_rate_limiter = create_rate_limiter(
    max_requests=RATE_LIMIT_MAX,
    window_seconds=RATE_LIMIT_WINDOW,
    require_redis=_require_redis,
)
_store_backend = "redis" if _redis_url else "in-memory"

# P1 — refuse plaintext .pth when only sealed blobs are allowed (production).
_REQUIRE_SEALED = os.environ.get(
    "REQUIRE_SEALED_CHECKPOINTS", ""
).lower() in ("1", "true", "yes")

print(
    f"[Dashboard] Inference protection active — "
    f"noise_scale={_output_perturbation.noise_scale:.3f} "
    f"(ε_inf={_output_perturbation.epsilon_inf:.2f}), "
    f"lifetime_budget={LIFETIME_BUDGET_DEFAULT} predictions/key, "
    f"stores={_store_backend}"
)
if _store_backend == "in-memory" and not _require_redis:
    print(
        "[Dashboard] WARNING: REDIS_URL not set — budget and rate limits are "
        "per-process and reset on restart. Set REDIS_URL for production."
    )
# ─────────────────────────────────────────────────────────────────────────────


def _check_rate_limit(client_ip: str):
    try:
        _rate_limiter.check(client_ip)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
# ─────────────────────────────────────────────────────────────────────────────

# ── Model loader with staleness detection ─────────────────────────────────────
_model_cache: dict = {}


def _resolve_checkpoint_path(base_pth: str) -> str | None:
    """
    Resolve global_model_latest.pth to a loadable path.

    Prefers .pth.sealed over plaintext. When REQUIRE_SEALED_CHECKPOINTS is set,
    returns None if only plaintext exists (P1 white-box containment).
    """
    sealed = base_pth + ".sealed"
    has_plain = os.path.isfile(base_pth)
    has_sealed = os.path.isfile(sealed)

    if _REQUIRE_SEALED:
        if has_sealed:
            return sealed
        if has_plain:
            print(
                f"[Dashboard] WARNING: Plaintext checkpoint rejected "
                f"({os.path.basename(base_pth)}). "
                "Run: python scripts/harden_production.py --seal-checkpoints"
            )
        return None

    if has_sealed:
        return sealed
    if has_plain:
        return base_pth
    return None


def _find_latest_model_path() -> str:
    """
    Find the most recently modified global_model_latest checkpoint under
    results/fl_rounds/ (prefers sealed blobs when available).
    """
    fl_rounds_dir = os.path.join(ROOT, "results", "fl_rounds")
    flat = os.path.join(fl_rounds_dir, "global_model_latest.pth")

    if not os.path.isdir(fl_rounds_dir):
        resolved = _resolve_checkpoint_path(flat)
        return resolved if resolved else flat

    candidates: list[str] = []
    for entry in os.scandir(fl_rounds_dir):
        if entry.is_dir() and entry.name.startswith("run_"):
            resolved = _resolve_checkpoint_path(
                os.path.join(entry.path, "global_model_latest.pth")
            )
            if resolved:
                candidates.append(resolved)

    resolved_flat = _resolve_checkpoint_path(flat)
    if resolved_flat:
        candidates.append(resolved_flat)

    if not candidates:
        return flat  # triggers 503 in _get_model

    return max(candidates, key=os.path.getmtime)


def _find_meta_path(model_path: str) -> str:
    """Return the model_meta.json path co-located with the checkpoint."""
    return os.path.join(os.path.dirname(model_path), "model_meta.json")


def _load_model_meta(model_path: str) -> dict:
    """Load model_meta.json. Falls back to inferring from processed CSVs."""
    meta_path = _find_meta_path(model_path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    # Fallback: infer from first processed CSV
    processed_dir = os.path.join(ROOT, "data", "processed")
    csv_files = sorted(
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir)
        if f.endswith(".csv")
    ) if os.path.isdir(processed_dir) else []

    if csv_files:
        import pandas as pd
        df_head = pd.read_csv(csv_files[0], nrows=1)
        input_dim = len([c for c in df_head.columns if c != "label"])
        df_full   = pd.read_csv(csv_files[0], usecols=["label"])
        num_classes = int(df_full["label"].nunique())
        return {
            "input_dim":   input_dim,
            "num_classes": num_classes,
            "class_names": [f"class_{i}" for i in range(num_classes)],
            "model_type":  "mlp",
        }

    raise HTTPException(
        status_code=503,
        detail="Cannot determine model shape — no model_meta.json or CSVs found.",
    )


def _get_model():
    """
    Return (model, meta). Reloads from disk if the checkpoint has been
    modified since the last load (staleness detection via mtime).
    """
    model_path = _find_latest_model_path()

    if not model_path or not os.path.exists(model_path):
        if _REQUIRE_SEALED:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No sealed checkpoint found. Train with FL server "
                    "(SEAL_REMOVE_PLAINTEXT=true) or run "
                    "scripts/harden_production.py --seal-checkpoints."
                ),
            )
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    current_mtime = os.path.getmtime(model_path)
    cached_mtime  = _model_cache.get("mtime", -1)

    if "model" not in _model_cache or current_mtime != cached_mtime:
        meta  = _load_model_meta(model_path)
        model = get_model(
            input_dim=meta["input_dim"],
            num_classes=meta["num_classes"],
            model_type=meta.get("model_type", "mlp"),
        )
        sys.path.insert(0, os.path.join(ROOT, "tee", "sealed_storage"))
        try:
            from sealed_storage import load_checkpoint_state_dict  # noqa: E402
            state = load_checkpoint_state_dict(model_path, map_location="cpu")
        except ImportError:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model_cache["model"] = model
        _model_cache["meta"]  = meta
        _model_cache["mtime"] = current_mtime
        _model_cache["path"]  = model_path
        print(f"[Dashboard] Model loaded from {model_path} "
              f"(type={meta.get('model_type','mlp')})")

    return _model_cache["model"], _model_cache["meta"]
# ─────────────────────────────────────────────────────────────────────────────

# ── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]
    return_confidence: bool = False


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float | None = None
# ─────────────────────────────────────────────────────────────────────────────


def _read_json(rel_path: str) -> dict:
    if rel_path in ("status.json", "attestation.json"):
        try:
            return read_json_runtime(rel_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"{rel_path} not found")
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{rel_path} not found")
    with open(path) as f:
        return json.load(f)


def _annotate_data_source(data: dict, rel_path: str) -> dict:
    """
    Add training_freshness / stale_hours so the UI can show LIVE vs STALE
  (not a brittle binary on file mtime alone).
    """
    sys.path.insert(0, os.path.join(ROOT, "dashboard", "backend"))
    from training_freshness import compute_training_freshness  # noqa: E402

    return compute_training_freshness(data, rel_path, ROOT)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    """
    Return the latest training run summary.

    Includes training_freshness (live | stale | training | pre-recorded),
    last_training_completed_at, stale_hours, and expected_training_cadence_hours.
    """
    data = _read_json("status.json")
    rel = "status.json"
    resolved = resolve_runtime_file("status.json")
    if resolved:
        rel = os.path.relpath(resolved, ROOT)
    return _annotate_data_source(data, rel)


@app.get("/results")
def results():
    """
    Return the full round-by-round metrics from the latest training run.
    Includes data_source: 'live' | 'pre-recorded'.
    """
    data = _read_json("results/results.json")
    return _annotate_data_source(data, "results/results.json")


@app.get("/attestation")
def attestation():
    """
    Return the TEE attestation record.

    When simulation_mode is True the attestation is software-computed
    (gramine-direct, no hardware enclave). The dashboard surfaces this
    as a visible warning. Production deployments use gramine-sgx.
    """
    sys.path.insert(0, os.path.join(ROOT, "tee"))
    try:
        from tee_mode import enrich_attestation_record  # noqa: E402
    except ImportError:
        enrich_attestation_record = lambda r: r  # type: ignore

    data = _read_json("attestation.json")
    if isinstance(data, dict):
        data = enrich_attestation_record(data)
    return data


@app.get("/benchmarks")
def benchmarks():
    return _read_json("results/benchmarks_baseline.json")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    """
    Run inference on a feature vector.

    Mitigations (Issue 1 — model inversion):
      1. Output perturbation — calibrated Laplace noise is added to the
         post-softmax probability vector before the label is derived.
         noise_scale = L1_sensitivity / ε_inf (default ε_inf = 4.0).
         The label itself is drawn from the noisy distribution, so the
         adversary cannot reconstruct the clean decision boundary.
      2. Lifetime query budget — each IP is capped at LIFETIME_BUDGET_DEFAULT
         total predictions. Budget counts completed predictions, not requests.
         Returns HTTP 429 with a permanent exhaustion message on cap.
      3. Per-window rate limiting — max 100 req / 60 s per IP (existing).
      4. Confidence masking — full probability vector is never returned;
         only the top-1 label and optionally the top-1 noisy score.
    """
    client_ip = request.client.host

    # Rate limit (per-window)
    _check_rate_limit(client_ip)

    # Lifetime budget check — consume BEFORE inference so a failed request
    # (wrong feature count) does not consume budget.
    model, meta  = _get_model()
    expected_dim = meta["input_dim"]
    class_names  = meta["class_names"]

    if len(payload.features) != expected_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_dim} features, got {len(payload.features)}",
        )

    # Consume one unit of lifetime budget (raises BudgetExhaustedError if at 0)
    try:
        remaining = _budget_store.consume(client_ip, LIFETIME_BUDGET_DEFAULT)
    except BudgetExhaustedError:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Lifetime prediction budget exhausted for this IP "
                f"(limit={LIFETIME_BUDGET_DEFAULT}). "
                "Contact the API administrator to request a new key."
            ),
        )

    x = torch.tensor(payload.features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()

    # Apply output perturbation — Laplace noise on the probability simplex
    noisy_probs = _output_perturbation.perturb(probs)

    pred_class = int(noisy_probs.argmax().item())
    pred_label = (class_names[pred_class]
                  if pred_class < len(class_names) else str(pred_class))

    # Return top-1 noisy confidence only if explicitly requested
    confidence = None
    if payload.return_confidence:
        confidence = round(float(noisy_probs[pred_class].item()), 4)

    return PredictResponse(
        predicted_class=pred_class,
        predicted_label=pred_label,
        confidence=confidence,
    )


@app.get("/attacks")
def attacks():
    """Return security attack simulation results for the dashboard."""
    out = {}
    attack_files = {
        "model_inversion":      "results/attacks/model_inversion.json",
        "membership_inference": "results/attacks/membership_inference.json",
        "gradient_poisoning":   "results/attacks/gradient_poisoning.json",
    }
    for key, rel_path in attack_files.items():
        path = os.path.join(ROOT, rel_path)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            out[key] = data.get("summary", {})
        else:
            out[key] = None
    return out


@app.get("/privacy_log")
def privacy_log():
    """
    Return the privacy audit log.

    The log contains per-round per-client epsilon values from Opacus.
    These are CUMULATIVE values (Renyi DP accountant), not per-round
    increments. The cumulative_summary field shows the worst-case
    cumulative epsilon across all clients and rounds.
    """
    path = os.path.join(ROOT, "results", "privacy_log.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="results/privacy_log.json not found")
    with open(path) as f:
        data = json.load(f)

    # Handle both old format (plain list) and new format (dict with log + summary)
    if isinstance(data, list):
        # Legacy format — wrap it and add a note
        return {
            "log": data,
            "cumulative_summary": {
                "note": (
                    "Legacy format detected. Re-run FL training to generate "
                    "cumulative epsilon summary. The epsilon values in this log "
                    "are cumulative (Opacus Renyi DP accountant) but the summary "
                    "was not computed at generation time."
                ),
                "max_epsilon_seen": max(
                    (e.get("epsilon") or e.get("epsilon_cumulative") or 0)
                    for e in data
                ) if data else 0,
            },
        }
    return data


@app.get("/query_stats")
def query_stats(request: Request):
    """Show rate limit and lifetime budget status for the calling IP."""
    client_ip = request.client.host
    budget_remaining = _budget_store.get_remaining(client_ip)
    return {
        "client_ip":              client_ip,
        "store_backend":          _store_backend,
        "redis_url_configured":   bool(_redis_url),
        "rate_limit_max":         RATE_LIMIT_MAX,
        "rate_limit_window_secs": RATE_LIMIT_WINDOW,
        "lifetime_budget_total":  LIFETIME_BUDGET_DEFAULT,
        "lifetime_budget_remaining": budget_remaining,
        "lifetime_budget_exhausted": budget_remaining <= 0,
        "noise_scale":            _output_perturbation.noise_scale,
        "epsilon_inf":            round(_output_perturbation.epsilon_inf, 4),
        "note": (
            "Budget and rate limits persist across restarts via Redis."
            if _store_backend == "redis"
            else "In-memory stores — set REDIS_URL (and REQUIRE_REDIS=true in prod) "
                 "for shared, restart-safe limits."
        ),
    }


@app.get("/model_info")
def model_info():
    """Return metadata about the currently loaded model."""
    model_path = _find_latest_model_path()
    if not model_path or not os.path.exists(model_path):
        if _REQUIRE_SEALED:
            raise HTTPException(
                status_code=503,
                detail="No sealed checkpoint found.",
            )
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    meta = _load_model_meta(model_path)
    return {
        "model_path":  model_path,
        "input_dim":   meta.get("input_dim"),
        "num_classes": meta.get("num_classes"),
        "class_names": meta.get("class_names"),
        "model_type":  meta.get("model_type", "mlp"),
        "checkpoint_mtime": os.path.getmtime(model_path),
    }
