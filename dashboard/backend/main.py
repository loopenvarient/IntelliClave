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
from collections import defaultdict
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
from pydantic import BaseModel

# ── model imports ─────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'fl'))
from data_utils import infer_default_preprocessing, load_preprocessing_metadata  # noqa: E402
from model import get_model  # noqa: E402
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="IntelliClave Dashboard API", version="1.0.0")

# ── Simple token-based auth (in-memory). Configure via DASHBOARD_USERS env var
# DASHBOARD_USERS expected JSON: {"username": {"password": "pw", "token": "tok", "role": "admin"}, ...}
import os as _os
import json as _json

_default_users = {
    "admin": {"password": "adminpass", "token": "admin-token-123", "role": "admin"},
    "viewer": {"password": "viewerpass", "token": "viewer-token-abc", "role": "viewer"},
}

_users = _default_users
try:
    env_users = _os.environ.get("DASHBOARD_USERS")
    if env_users:
        parsed = _json.loads(env_users)
        if isinstance(parsed, dict):
            _users = parsed
except Exception:
    # ignore parsing errors and fall back to defaults
    pass

# Build token->user map for fast lookup
TOKEN_STORE = {u["token"]: {"username": name, "role": u.get("role", "viewer")} for name, u in _users.items()}

security = HTTPBearer(auto_error=False)
ISSUED_TOKEN_STORE = {}


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    if credentials is None:
        return {"username": "anonymous", "role": "viewer", "is_anonymous": True}

    token = credentials.credentials
    # 1) Legacy token check
    user = TOKEN_STORE.get(token)
    if user:
        return user

    # 2) Issued bearer token check
    user = ISSUED_TOKEN_STORE.get(token)
    if user:
        return user

    raise HTTPException(status_code=401, detail="Invalid or missing authentication token")


def require_authenticated(user=Depends(get_current_user)):
    return user


def require_admin(user=Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

# ── CORS — configurable via environment variable ──────────────────────────────
# Set CORS_ORIGINS="https://app.example.com,https://admin.example.com" in prod.
_cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# ─────────────────────────────────────────────────────────────────────────────

# ── Rate limiter ──────────────────────────────────────────────────────────────
# NOTE: This is an in-memory rate limiter. In a multi-worker deployment
# (uvicorn --workers N) each worker has its own counter, so the effective
# limit is N × RATE_LIMIT_MAX. For production, replace with a Redis-backed
# solution (e.g. slowapi with Redis storage).
RATE_LIMIT_MAX    = 100
RATE_LIMIT_WINDOW = 60   # seconds
_query_log: dict  = defaultdict(list)


def _check_rate_limit(client_ip: str):
    now    = time.time()
    window = now - RATE_LIMIT_WINDOW
    _query_log[client_ip] = [t for t in _query_log[client_ip] if t > window]
    if len(_query_log[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX} queries "
                   f"per {RATE_LIMIT_WINDOW}s. Try again later.",
        )
    _query_log[client_ip].append(now)
# ─────────────────────────────────────────────────────────────────────────────

# ── Model loader with staleness detection ─────────────────────────────────────
_model_cache: dict = {}


def _find_latest_model_path() -> str:
    """
    Find the most recently modified global_model_latest.pth across all
    timestamped run subdirectories under results/fl_rounds/.

    Falls back to the legacy flat path results/fl_rounds/global_model_latest.pth
    if no run subdirectories exist.
    """
    fl_rounds_dir = os.path.join(ROOT, "results", "fl_rounds")
    if not os.path.isdir(fl_rounds_dir):
        return os.path.join(fl_rounds_dir, "global_model_latest.pth")

    candidates = []
    # Check run_* subdirectories (timestamped)
    for entry in os.scandir(fl_rounds_dir):
        if entry.is_dir() and entry.name.startswith("run_"):
            p = os.path.join(entry.path, "global_model_latest.pth")
            if os.path.exists(p):
                candidates.append(p)
    # Also check the legacy flat location
    flat = os.path.join(fl_rounds_dir, "global_model_latest.pth")
    if os.path.exists(flat):
        candidates.append(flat)

    if not candidates:
        return flat  # will trigger 503 in _get_model

    # Return the most recently modified checkpoint
    return max(candidates, key=os.path.getmtime)


def _find_meta_path(model_path: str) -> str:
    """Return the model_meta.json path co-located with the checkpoint."""
    return os.path.join(os.path.dirname(model_path), "model_meta.json")


def _find_privacy_log_path() -> str:
    """
    Return the privacy log path for the currently active model run.

    The FL server writes run-scoped privacy telemetry beside the checkpoint as
    fl_privacy.json, so the dashboard should read that same artifact.
    """
    model_path = _find_latest_model_path()
    return os.path.join(os.path.dirname(model_path), "fl_privacy.json")


def _load_model_meta(model_path: str) -> dict:
    """Load model_meta.json. Falls back to inferring from processed CSVs."""
    meta_path = _find_meta_path(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
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

    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    current_mtime = os.path.getmtime(model_path)
    cached_mtime  = _model_cache.get("mtime", -1)

    if "model" not in _model_cache or current_mtime != cached_mtime:
        meta  = _load_model_meta(model_path)
        preprocessing = load_preprocessing_metadata(model_path)
        if preprocessing is None:
            preprocessing = infer_default_preprocessing()
        model = get_model(
            input_dim=meta["input_dim"],
            num_classes=meta["num_classes"],
            model_type=meta.get("model_type", "mlp"),
        )
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        _model_cache["model"] = model
        _model_cache["meta"]  = meta
        _model_cache["preprocessing"] = preprocessing
        if preprocessing is not None:
            _model_cache["mean_tensor"] = torch.tensor(preprocessing["mean"], dtype=torch.float32)
            _model_cache["std_tensor"] = torch.tensor(preprocessing["std"], dtype=torch.float32)
        else:
            _model_cache["mean_tensor"] = None
            _model_cache["std_tensor"] = None
        _model_cache["mtime"] = current_mtime
        _model_cache["path"]  = model_path
        print(f"[Dashboard] Model loaded from {model_path} "
              f"(type={meta.get('model_type','mlp')})")

    return (
        _model_cache["model"],
        _model_cache["meta"],
        _model_cache.get("preprocessing"),
    )
# ─────────────────────────────────────────────────────────────────────────────

# ── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]
    return_confidence: bool = False


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float | None = None
# Additional response models for stricter API schemas


class ClientInfo(BaseModel):
    client_id: Optional[str] = None
    id: Optional[str] = None
    status: Optional[str] = None
    samples: Optional[int] = None


class StatusResponse(BaseModel):
    round: Optional[int] = None
    total_rounds: Optional[int] = None
    clients: List[Dict[str, Any]] = []
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    save_dir: Optional[str] = None
    model_type: Optional[str] = None
    early_stopped: Optional[bool] = None
    epsilon: Optional[float] = None
    training_active: Optional[bool] = False


class ResultsResponse(BaseModel):
    rounds: List[Dict[str, Any]] = []
    save_dir: Optional[str] = None
    per_class_f1: Optional[Dict[str, float]] = None


class AttackSummary(BaseModel):
    verdict: Optional[str] = None
    avg_cosine_similarity: Optional[float] = None
    auc: Optional[float] = None
    accuracy_drop: Optional[float] = None


class AttacksResponse(BaseModel):
    model_inversion: Optional[Dict[str, Any]] = None
    membership_inference: Optional[Dict[str, Any]] = None
    gradient_poisoning: Optional[Dict[str, Any]] = None


class PrivacyLogEntry(BaseModel):
    round: Optional[int] = None
    epsilon: Optional[float] = None
    avg_epsilon: Optional[float] = None
    clients: Optional[List[Dict[str, Any]]] = None
# ─────────────────────────────────────────────────────────────────────────────


def _read_json(rel_path: str) -> dict:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{rel_path} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/token")
async def token(request: Request):
    """Exchange username/password for a bearer token (development only).

    Supply `DASHBOARD_USERS` env var in production instead of this ad-hoc store.
    """
    form_data = await request.form()
    username = str(form_data.get("username", ""))
    password = str(form_data.get("password", ""))
    user = _users.get(username)
    if not user or user.get("password") != password:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    # Issue opaque bearer token for local dashboard auth.
    token = secrets.token_urlsafe(32)
    ISSUED_TOKEN_STORE[token] = {"username": username, "role": user.get("role", "viewer")}
    return {"access_token": token, "token_type": "bearer", "role": user.get("role")}


@app.get("/status", response_model=StatusResponse)
def status(user=Depends(get_current_user)):
    """
    Return the latest training run summary.
    Written by run_fl_simulation.py / fl_server.py after training completes.
    """
    # Load raw status.json then normalize for frontend expectations.
    status_obj = _read_json("status.json")

    # Normalize `clients`: if a simple integer is present, convert to an array
    # of placeholder client descriptors so the frontend can iterate safely.
    clients = status_obj.get("clients")
    if isinstance(clients, int):
        status_obj["clients"] = [{"client_id": i + 1} for i in range(clients)]

    # If epsilon is missing, attempt to extract it from the privacy log
    if "epsilon" not in status_obj:
        try:
            ppath = _find_privacy_log_path()
            if os.path.exists(ppath):
                with open(ppath, encoding="utf-8") as pf:
                    pl = json.load(pf)
                # Heuristic: look for common epsilon keys in privacy log
                def _find_eps(obj):
                    if isinstance(obj, dict):
                        for k in ("epsilon", "actual_epsilon", "cumulative_epsilon", "spent_epsilon", "final_epsilon", "total_epsilon"):
                            if k in obj and isinstance(obj[k], (int, float)):
                                return float(obj[k])
                        # try nested
                        for v in obj.values():
                            res = _find_eps(v)
                            if res is not None:
                                return res
                    if isinstance(obj, list) and obj:
                        return _find_eps(obj[-1])
                    return None

                found = _find_eps(pl)
                if found is not None:
                    status_obj["epsilon"] = found
        except Exception:
            # Do not fail the status endpoint for parsing heuristics.
            pass

    return status_obj


@app.get("/results", response_model=ResultsResponse)
def results(user=Depends(get_current_user)):
    """
    Return the full round-by-round metrics from the latest training run.
    Written by run_fl_simulation.py / fl_server.py after training completes.
    """
    return _read_json("results/results.json")


@app.get("/attestation")
def attestation(user=Depends(get_current_user)):
    return _read_json("attestation.json")


@app.get("/benchmarks")
def benchmarks(user=Depends(get_current_user)):
    return _read_json("results/benchmarks_baseline.json")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request, user=Depends(require_authenticated)):
    """
    Run inference on a feature vector.

    The expected feature length is determined by the trained model at runtime —
    no hardcoded dimensions. Works with any dataset.

    Mitigations:
      2. Confidence masking — returns label only by default. Set
         return_confidence=true for top-1 score (no full probability vector).
      3. Rate limiting — max 100 queries per IP per 60 seconds → HTTP 429.
    """
    client_ip = request.client.host
    _check_rate_limit(client_ip)

    model, meta, preprocessing = _get_model()
    expected_dim = meta["input_dim"]
    class_names  = meta["class_names"]

    if len(payload.features) != expected_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_dim} features, got {len(payload.features)}",
        )

    if preprocessing is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model preprocessing metadata is missing. Retrain the model so "
                "the run produces preprocessing.json before serving predictions."
            ),
        )

    mean = preprocessing["mean"]
    std = preprocessing["std"]
    if len(mean) != expected_dim or len(std) != expected_dim:
        raise HTTPException(
            status_code=500,
            detail="Invalid preprocessing metadata: feature dimensions do not match the model.",
        )

    x = torch.tensor(payload.features, dtype=torch.float32)
    mean_tensor = _model_cache["mean_tensor"]
    std_tensor = _model_cache["std_tensor"]
    x = (x - mean_tensor) / std_tensor
    x = x.unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()

    pred_class = int(probs.argmax().item())
    pred_label = (class_names[pred_class]
                  if pred_class < len(class_names) else str(pred_class))

    confidence = None
    if payload.return_confidence:
        confidence = round(float(probs[pred_class].item()), 4)

    return PredictResponse(
        predicted_class=pred_class,
        predicted_label=pred_label,
        confidence=confidence,
    )


@app.get("/attacks", response_model=AttacksResponse)
def attacks(user=Depends(get_current_user)):
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
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out[key] = data.get("summary", {})
        else:
            out[key] = None
    return out


@app.get("/privacy_log", response_model=List[PrivacyLogEntry])
def privacy_log(user=Depends(get_current_user)):
    path = _find_privacy_log_path()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            pl = json.load(f)
        # Normalize entries to include top-level `epsilon` for frontend/tests
        if isinstance(pl, list):
            out = []
            for entry in pl:
                if isinstance(entry, dict):
                    if "epsilon" not in entry:
                        if "avg_epsilon" in entry:
                            entry["epsilon"] = entry.get("avg_epsilon")
                        else:
                            clients = entry.get("clients")
                            if isinstance(clients, list) and clients:
                                first = clients[0]
                                if isinstance(first, dict) and "epsilon" in first:
                                    entry["epsilon"] = first.get("epsilon")
                out.append(entry)
            return out
        return pl

    # Backward-compatible fallback for older demo artifacts.
    legacy_path = os.path.join(ROOT, "results", "privacy_log.json")
    if os.path.exists(legacy_path):
        with open(legacy_path, encoding="utf-8") as f:
            pl = json.load(f)
        # Normalize legacy privacy_log (list of dicts) as above
        if isinstance(pl, list):
            out = []
            for entry in pl:
                if isinstance(entry, dict) and "epsilon" not in entry:
                    if "avg_epsilon" in entry:
                        entry["epsilon"] = entry.get("avg_epsilon")
                out.append(entry)
            return out
        return pl

    # Additional fallback: if an epsilon-over-rounds result exists, synthesize
    # a privacy log from it so the dashboard and tests have per-round epsilon
    # telemetry to display.
    eps_rounds = os.path.join(ROOT, "results", "epsilon_rounds.json")
    if os.path.exists(eps_rounds):
        try:
            with open(eps_rounds, encoding="utf-8") as f:
                rounds = json.load(f)
            out = []
            for r in rounds:
                eps = r.get("epsilon_consumed") or r.get("epsilon") or r.get("actual_epsilon")
                round_num = r.get("fl_round") or r.get("round")
                entry = {"round": round_num, "epsilon": eps}
                out.append(entry)
            if out:
                return out
        except Exception:
            pass

    # Final fallback: return an empty list rather than 404 so the UI can
    # gracefully show "no privacy log available" instead of an error.
    return []


@app.get("/query_stats")
def query_stats(request: Request, user=Depends(get_current_user)):
    """Show rate limit status for the calling IP."""
    client_ip = request.client.host
    now    = time.time()
    window = now - RATE_LIMIT_WINDOW
    recent = [t for t in _query_log.get(client_ip, []) if t > window]
    return {
        "client_ip":         client_ip,
        "queries_in_window": len(recent),
        "limit":             RATE_LIMIT_MAX,
        "window_seconds":    RATE_LIMIT_WINDOW,
        "remaining":         max(0, RATE_LIMIT_MAX - len(recent)),
        "note": (
            "Rate limiter is in-memory. In multi-worker deployments each worker "
            "has its own counter. Use Redis-backed rate limiting for production."
        ),
    }


@app.get("/model_info")
def model_info(user=Depends(get_current_user)):
    """Return metadata about the currently loaded model."""
    model_path = _find_latest_model_path()
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    meta = _load_model_meta(model_path)
    preprocessing = load_preprocessing_metadata(model_path) or infer_default_preprocessing()
    return {
        "model_path":  model_path,
        "input_dim":   meta.get("input_dim"),
        "num_classes": meta.get("num_classes"),
        "class_names": meta.get("class_names"),
        "model_type":  meta.get("model_type", "mlp"),
        "preprocessing_available": preprocessing is not None,
        "preprocessing_mode": None if preprocessing is None else preprocessing.get("normalization"),
        "checkpoint_mtime": os.path.getmtime(model_path),
    }
