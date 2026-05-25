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
  - Model inversion defence: /predict now loads the model via get_defended_model()
    (PrivacyWrapper with Laplace noise + temperature scaling). Defence parameters
    are read from config/constants.py (MI_NOISE_SCALE, MI_TEMPERATURE,
    MI_DEFENCE_ENABLED) and can be overridden via environment variables
    MI_NOISE_SCALE, MI_TEMPERATURE, and MI_DEFENCE_ENABLED.
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
sys.path.insert(0, os.path.join(ROOT, 'config'))
from data_utils import infer_default_preprocessing, load_preprocessing_metadata  # noqa: E402
from model import get_defended_model  # noqa: E402  ← replaces bare get_model

# ── Defence constants — override via env vars in production ───────────────────
try:
    from constants import MI_NOISE_SCALE, MI_TEMPERATURE, MI_DEFENCE_ENABLED
except ImportError:
    MI_NOISE_SCALE    = 0.5
    MI_TEMPERATURE    = 4.0
    MI_DEFENCE_ENABLED = True

_noise_scale   = float(os.environ.get("MI_NOISE_SCALE",    MI_NOISE_SCALE))
_temperature   = float(os.environ.get("MI_TEMPERATURE",    MI_TEMPERATURE))
_defence_on    = os.environ.get("MI_DEFENCE_ENABLED", str(MI_DEFENCE_ENABLED)).lower() not in ("0", "false")
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
    pass

TOKEN_STORE = {u["token"]: {"username": name, "role": u.get("role", "viewer")} for name, u in _users.items()}

security = HTTPBearer(auto_error=False)
ISSUED_TOKEN_STORE = {}


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    if credentials is None:
        return {"username": "anonymous", "role": "viewer", "is_anonymous": True}
    token = credentials.credentials
    user = TOKEN_STORE.get(token)
    if user:
        return user
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

# ── CORS ──────────────────────────────────────────────────────────────────────
_cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
RATE_LIMIT_MAX    = 100
RATE_LIMIT_WINDOW = 60
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

# ── Model loader with staleness detection ─────────────────────────────────────
_model_cache: dict = {}


def _find_latest_model_path() -> str:
    fl_rounds_dir = os.path.join(ROOT, "results", "fl_rounds")
    if not os.path.isdir(fl_rounds_dir):
        return os.path.join(fl_rounds_dir, "global_model_latest.pth")

    candidates = []
    for entry in os.scandir(fl_rounds_dir):
        if entry.is_dir() and entry.name.startswith("run_"):
            p = os.path.join(entry.path, "global_model_latest.pth")
            if os.path.exists(p):
                candidates.append(p)
    flat = os.path.join(fl_rounds_dir, "global_model_latest.pth")
    if os.path.exists(flat):
        candidates.append(flat)

    if not candidates:
        return flat
    return max(candidates, key=os.path.getmtime)


def _find_meta_path(model_path: str) -> str:
    return os.path.join(os.path.dirname(model_path), "model_meta.json")


def _find_privacy_log_path() -> str:
    model_path = _find_latest_model_path()
    return os.path.join(os.path.dirname(model_path), "fl_privacy.json")


def _load_model_meta(model_path: str) -> dict:
    meta_path = _find_meta_path(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

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
    Return (model, meta, preprocessing).

    The model is a PrivacyWrapper (get_defended_model) with Laplace noise +
    temperature scaling active during eval(). Reloads from disk when the
    checkpoint mtime changes.
    """
    model_path = _find_latest_model_path()

    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    current_mtime = os.path.getmtime(model_path)
    cached_mtime  = _model_cache.get("mtime", -1)

    if "model" not in _model_cache or current_mtime != cached_mtime:
        meta         = _load_model_meta(model_path)
        preprocessing = load_preprocessing_metadata(model_path)
        if preprocessing is None:
            preprocessing = infer_default_preprocessing()

        # ── Build defended model ──────────────────────────────────────────────
        # get_defended_model returns PrivacyWrapper(base_model).
        # We load weights into wrapper.base_model so Opacus-trained checkpoints
        # (which only save the inner model's state_dict) load correctly.
        wrapper = get_defended_model(
            input_dim=meta["input_dim"],
            num_classes=meta["num_classes"],
            model_type=meta.get("model_type", "mlp"),
            noise_scale=_noise_scale,
            temperature=_temperature,
            enabled=_defence_on,
        )
        wrapper.base_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        wrapper.eval()
        # ─────────────────────────────────────────────────────────────────────

        _model_cache["model"]        = wrapper
        _model_cache["meta"]         = meta
        _model_cache["preprocessing"] = preprocessing
        if preprocessing is not None:
            _model_cache["mean_tensor"] = torch.tensor(preprocessing["mean"], dtype=torch.float32)
            _model_cache["std_tensor"]  = torch.tensor(preprocessing["std"],  dtype=torch.float32)
        else:
            _model_cache["mean_tensor"] = None
            _model_cache["std_tensor"]  = None
        _model_cache["mtime"] = current_mtime
        _model_cache["path"]  = model_path
        print(
            f"[Dashboard] Model loaded from {model_path} "
            f"(type={meta.get('model_type','mlp')}, "
            f"defence={'ON' if _defence_on else 'OFF'} "
            f"noise={_noise_scale} temp={_temperature})"
        )

    return (
        _model_cache["model"],
        _model_cache["meta"],
        _model_cache.get("preprocessing"),
    )

# ── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]
    return_confidence: bool = False


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float | None = None


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
    form_data = await request.form()
    username = str(form_data.get("username", ""))
    password = str(form_data.get("password", ""))
    user = _users.get(username)
    if not user or user.get("password") != password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    tok = secrets.token_urlsafe(32)
    ISSUED_TOKEN_STORE[tok] = {"username": username, "role": user.get("role", "viewer")}
    return {"access_token": tok, "token_type": "bearer", "role": user.get("role")}


@app.get("/status", response_model=StatusResponse)
def status(user=Depends(get_current_user)):
    status_obj = _read_json("status.json")
    clients = status_obj.get("clients")
    if isinstance(clients, int):
        status_obj["clients"] = [{"client_id": i + 1} for i in range(clients)]

    if "epsilon" not in status_obj:
        try:
            ppath = _find_privacy_log_path()
            if os.path.exists(ppath):
                with open(ppath, encoding="utf-8") as pf:
                    pl = json.load(pf)

                def _find_eps(obj):
                    if isinstance(obj, dict):
                        for k in ("epsilon", "actual_epsilon", "cumulative_epsilon",
                                  "spent_epsilon", "final_epsilon", "total_epsilon"):
                            if k in obj and isinstance(obj[k], (int, float)):
                                return float(obj[k])
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
            pass

    return status_obj


@app.get("/results", response_model=ResultsResponse)
def results(user=Depends(get_current_user)):
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

    Model inversion defences active (when MI_DEFENCE_ENABLED=True):
      1. Output perturbation — Laplace noise on logits (scale=MI_NOISE_SCALE).
      2. Temperature scaling — logits divided by MI_TEMPERATURE before softmax.
      3. Confidence masking — top-1 label returned by default; top-1 score only
         when return_confidence=true (no full probability vector ever exposed).
      4. Rate limiting — max 100 queries per IP per 60 seconds → HTTP 429.

    Defence parameters can be tuned without retraining:
      export MI_NOISE_SCALE=1.0    # increase if cosine sim still above 0.6
      export MI_TEMPERATURE=6.0    # increase to flatten probabilities further
      export MI_DEFENCE_ENABLED=false  # disable for ablation / testing
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
    std  = preprocessing["std"]
    if len(mean) != expected_dim or len(std) != expected_dim:
        raise HTTPException(
            status_code=500,
            detail="Invalid preprocessing metadata: feature dimensions do not match the model.",
        )

    x = torch.tensor(payload.features, dtype=torch.float32)
    mean_tensor = _model_cache["mean_tensor"]
    std_tensor  = _model_cache["std_tensor"]
    x = (x - mean_tensor) / std_tensor
    x = x.unsqueeze(0)

    # ── Defended inference ────────────────────────────────────────────────────
    # model is a PrivacyWrapper in eval() mode.
    # PrivacyWrapper.forward() applies Laplace noise + temperature scaling and
    # returns softmax probabilities (not raw logits) when in eval mode.
    # No torch.no_grad() is needed for the wrapper itself — it contains no
    # parameters beyond base_model — but we keep it for memory efficiency.
    with torch.no_grad():
        probs = model(x).squeeze()   # shape: (num_classes,)
    # ─────────────────────────────────────────────────────────────────────────

    pred_class = int(probs.argmax().item())
    pred_label = (class_names[pred_class]
                  if pred_class < len(class_names) else str(pred_class))

    # Return confidence only if explicitly requested (confidence masking).
    # Never expose the full probability vector.
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

    legacy_path = os.path.join(ROOT, "results", "privacy_log.json")
    if os.path.exists(legacy_path):
        with open(legacy_path, encoding="utf-8") as f:
            pl = json.load(f)
        if isinstance(pl, list):
            out = []
            for entry in pl:
                if isinstance(entry, dict) and "epsilon" not in entry:
                    if "avg_epsilon" in entry:
                        entry["epsilon"] = entry.get("avg_epsilon")
                out.append(entry)
            return out
        return pl

    eps_rounds = os.path.join(ROOT, "results", "epsilon_rounds.json")
    if os.path.exists(eps_rounds):
        try:
            with open(eps_rounds, encoding="utf-8") as f:
                rounds = json.load(f)
            out = []
            for r in rounds:
                eps = r.get("epsilon_consumed") or r.get("epsilon") or r.get("actual_epsilon")
                round_num = r.get("fl_round") or r.get("round")
                out.append({"round": round_num, "epsilon": eps})
            if out:
                return out
        except Exception:
            pass

    return []


@app.get("/query_stats")
def query_stats(request: Request, user=Depends(get_current_user)):
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
    model_path = _find_latest_model_path()
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    meta = _load_model_meta(model_path)
    preprocessing = load_preprocessing_metadata(model_path) or infer_default_preprocessing()
    return {
        "model_path":              model_path,
        "input_dim":               meta.get("input_dim"),
        "num_classes":             meta.get("num_classes"),
        "class_names":             meta.get("class_names"),
        "model_type":              meta.get("model_type", "mlp"),
        "preprocessing_available": preprocessing is not None,
        "preprocessing_mode":      None if preprocessing is None else preprocessing.get("normalization"),
        "checkpoint_mtime":        os.path.getmtime(model_path),
        "mi_defence": {
            "enabled":     _defence_on,
            "noise_scale": _noise_scale,
            "temperature": _temperature,
        },
    }