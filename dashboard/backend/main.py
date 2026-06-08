"""
IntelliClave Dashboard — FastAPI backend

Fixes applied:
  - Model cache is invalidated when global_model_latest.pth changes on disk.
  - /status and /results endpoints read from status.json and results/results.json.
  - Model path discovery scans all timestamped run subdirectories.
  - CORS allowed origins configurable via CORS_ORIGINS env var.
  - Model inversion defence via get_defended_model() (PrivacyWrapper).
  - /attacks endpoint prefers *_defended.json > *_mitigated.json > base file.
  - Batch CSV prediction routes registered via predict_routes.py.
"""
import json
import os
import sys
import time
import random
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
from data_utils import (  # noqa: E402
    get_default_client_csvs,
    infer_default_preprocessing,
    load_csv_data,
    load_preprocessing_metadata,
)
from model import get_defended_model  # noqa: E402

# ── Defence constants ─────────────────────────────────────────────────────────
try:
    from constants import (
        MI_NOISE_SCALE,
        MI_TEMPERATURE,
        MI_DEFENCE_ENABLED,
        OUTPUT_PROB_ROUNDING_STEP,
        OUTPUT_TOP_K,
        OUTPUT_RANDOM_RESPONSE_PROB,
    )
except ImportError:
    MI_NOISE_SCALE             = 0.5
    MI_TEMPERATURE             = 4.0
    MI_DEFENCE_ENABLED         = True
    OUTPUT_PROB_ROUNDING_STEP  = 0.1
    OUTPUT_TOP_K               = 1
    OUTPUT_RANDOM_RESPONSE_PROB = 0.05

_noise_scale        = float(os.environ.get("MI_NOISE_SCALE",           MI_NOISE_SCALE))
_temperature        = float(os.environ.get("MI_TEMPERATURE",           MI_TEMPERATURE))
_defence_on         = os.environ.get("MI_DEFENCE_ENABLED", str(MI_DEFENCE_ENABLED)).lower() not in ("0", "false")
_prob_rounding_step = float(os.environ.get("OUTPUT_PROB_ROUNDING_STEP", OUTPUT_PROB_ROUNDING_STEP))
_top_k              = max(1, int(os.environ.get("OUTPUT_TOP_K",          OUTPUT_TOP_K)))
_random_response_prob = float(os.environ.get("OUTPUT_RANDOM_RESPONSE_PROB", OUTPUT_RANDOM_RESPONSE_PROB))

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="IntelliClave Dashboard API", version="1.0.0")

# ── Batch CSV prediction routes ───────────────────────────────────────────────
from predict_routes import predict_router  # noqa: E402
app.include_router(predict_router)

# ── Auth ──────────────────────────────────────────────────────────────────────
import os as _os
import json as _json

_default_users = {
    "admin":  {"password": "adminpass",  "token": "admin-token-123",  "role": "admin"},
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

TOKEN_STORE        = {u["token"]: {"username": name, "role": u.get("role", "viewer")} for name, u in _users.items()}
security           = HTTPBearer(auto_error=False)
ISSUED_TOKEN_STORE = {}


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    if credentials is None:
        return {"username": "anonymous", "role": "viewer", "is_anonymous": True}
    auth_token = credentials.credentials
    user = TOKEN_STORE.get(auth_token) or ISSUED_TOKEN_STORE.get(auth_token)
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
_cors_env     = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
RATE_LIMIT_MAX          = 100
RATE_LIMIT_WINDOW       = 60
_query_log: dict        = defaultdict(list)
PREDICT_RATE_LIMIT_MAX    = int(os.environ.get("PREDICT_RATE_LIMIT_MAX",    "20"))
PREDICT_RATE_LIMIT_WINDOW = int(os.environ.get("PREDICT_RATE_LIMIT_WINDOW", "60"))
_predict_query_log: dict  = defaultdict(list)


def _check_rate_limit(client_ip: str):
    now    = time.time()
    window = now - RATE_LIMIT_WINDOW
    _query_log[client_ip] = [t for t in _query_log[client_ip] if t > window]
    if len(_query_log[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX} queries per {RATE_LIMIT_WINDOW}s.")
    _query_log[client_ip].append(now)


def _check_predict_rate_limit(client_ip: str):
    now    = time.time()
    window = now - PREDICT_RATE_LIMIT_WINDOW
    _predict_query_log[client_ip] = [t for t in _predict_query_log[client_ip] if t > window]
    if len(_predict_query_log[client_ip]) >= PREDICT_RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail=f"Prediction rate limit exceeded: max {PREDICT_RATE_LIMIT_MAX} queries per {PREDICT_RATE_LIMIT_WINDOW}s.")
    _predict_query_log[client_ip].append(now)

# ── Model loader ──────────────────────────────────────────────────────────────
_model_cache: dict = {}
_eval_cache:  dict = {}


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
    return max(candidates, key=os.path.getmtime) if candidates else flat


def _find_meta_path(model_path: str) -> str:
    return os.path.join(os.path.dirname(model_path), "model_meta.json")


def _find_privacy_log_path() -> str:
    return os.path.join(os.path.dirname(_find_latest_model_path()), "fl_privacy.json")


def _load_model_meta(model_path: str) -> dict:
    meta_path = _find_meta_path(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    processed_dir = os.path.join(ROOT, "data", "processed")
    csv_files = sorted(
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir) if f.endswith(".csv")
    ) if os.path.isdir(processed_dir) else []
    if csv_files:
        import pandas as pd
        df_head = pd.read_csv(csv_files[0], nrows=1)
        input_dim = len([c for c in df_head.columns if c != "label"])
        df_full   = pd.read_csv(csv_files[0], usecols=["label"])
        num_classes = int(df_full["label"].nunique())
        return {"input_dim": input_dim, "num_classes": num_classes,
                "class_names": [f"class_{i}" for i in range(num_classes)], "model_type": "mlp"}
    raise HTTPException(status_code=503, detail="Cannot determine model shape — no model_meta.json or CSVs found.")


def _get_model():
    model_path    = _find_latest_model_path()
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    current_mtime = os.path.getmtime(model_path)
    if "model" not in _model_cache or current_mtime != _model_cache.get("mtime", -1):
        meta          = _load_model_meta(model_path)
        preprocessing = load_preprocessing_metadata(model_path) or infer_default_preprocessing()
        wrapper = get_defended_model(
            input_dim=meta["input_dim"], num_classes=meta["num_classes"],
            model_type=meta.get("model_type", "mlp"),
            noise_scale=_noise_scale, temperature=_temperature, enabled=_defence_on,
        )
        wrapper.base_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        wrapper.eval()
        _model_cache.update({
            "model": wrapper, "meta": meta, "preprocessing": preprocessing,
            "mean_tensor": torch.tensor(preprocessing["mean"], dtype=torch.float32) if preprocessing else None,
            "std_tensor":  torch.tensor(preprocessing["std"],  dtype=torch.float32) if preprocessing else None,
            "mtime": current_mtime, "path": model_path,
        })
        print(f"[Dashboard] Model loaded ({meta.get('model_type','mlp')}, defence={'ON' if _defence_on else 'OFF'}, noise={_noise_scale}, temp={_temperature})")
    return _model_cache["model"], _model_cache["meta"], _model_cache.get("preprocessing")

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]
    return_confidence: bool = False
    randomized_response: bool = True

class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float | None = None

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
    noise_scale: Optional[float] = None
    temperature: Optional[float] = None
    client_distribution: Optional[Dict[str, Any]] = None

class ResultsResponse(BaseModel):
    rounds: List[Dict[str, Any]] = []
    save_dir: Optional[str] = None
    per_class_f1: Optional[Dict[str, float]] = None

class AttacksResponse(BaseModel):
    model_inversion: Optional[Dict[str, Any]] = None
    membership_inference: Optional[Dict[str, Any]] = None
    gradient_poisoning: Optional[Dict[str, Any]] = None

class PrivacyLogEntry(BaseModel):
    round: Optional[int] = None
    epsilon: Optional[float] = None
    avg_epsilon: Optional[float] = None
    clients: Optional[List[Dict[str, Any]]] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_json(rel_path: str) -> dict:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{rel_path} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _read_json_optional(rel_path: str) -> Optional[dict]:
    path = os.path.join(ROOT, rel_path)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None

def _get_save_dir(status_obj: Optional[dict] = None) -> str:
    if status_obj and status_obj.get("save_dir"):
        candidate = os.path.join(ROOT, status_obj["save_dir"])
        if os.path.isdir(candidate):
            return candidate
    return os.path.dirname(_find_latest_model_path())

def _short_verdict(verdict: Optional[str]) -> str:
    if not verdict:
        return "—"
    for sep in (" \u2014 ", " — ", " - "):
        if sep in verdict:
            return verdict.split(sep)[0].strip()
    return verdict.strip()

def _normalize_attack_summary(raw: Optional[dict]) -> Optional[dict]:
    if not raw:
        return None
    summary = dict(raw)
    if summary.get("avg_auc") is not None and summary.get("auc") is None:
        summary["auc"] = summary["avg_auc"]
    drop = summary.get("accuracy_drop")
    if drop is not None:
        drop_val = float(drop)
        summary["accuracy_drop_pct"] = round(drop_val * 100, 2) if drop_val <= 1.0 else round(drop_val, 2)
    summary["verdict_short"] = _short_verdict(summary.get("verdict"))
    return summary

def _best_attack_path(base_rel_path: str) -> str:
    """
    Walk through priority variants of an attack result file and return the
    most-defended one that exists on disk.

    Priority (highest defence first):
      1. <stem>_defended.json   — full noise + temperature defence (noise=3.0, temp=15.0)
      2. <stem>_mitigated.json  — confidence-masked only
      3. <stem>.json            — unmitigated baseline (fallback)

    This ensures /attacks always returns the RESISTANT defended verdict rather
    than the unmitigated VULNERABLE baseline.
    """
    stem = base_rel_path.replace(".json", "")
    for suffix in ("_defended.json", "_mitigated.json", ".json"):
        candidate = os.path.join(ROOT, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(ROOT, base_rel_path)

def _load_client_info(save_dir: str, n_clients: int) -> List[dict]:
    dist_path = os.path.join(save_dir, "distribution_report.json")
    if not os.path.exists(dist_path):
        return [{"id": f"Client {i+1}", "client_id": i+1, "status": "ready", "samples": 0} for i in range(n_clients)]
    with open(dist_path, encoding="utf-8") as f:
        dist = json.load(f)
    counts      = dist.get("counts_per_client", {})
    kl_map      = dist.get("kl_divergence", {})
    client_files = dist.get("clients") or list(counts.keys())
    clients = []
    for i, cf in enumerate(client_files):
        class_counts = counts.get(cf, {})
        samples = sum(int(v) for v in class_counts.values())
        kl_val  = kl_map.get(cf)
        clients.append({"id": f"Client {i+1}", "client_id": i+1, "status": "ready",
                         "samples": samples,
                         "kl_divergence": round(float(kl_val), 4) if kl_val is not None else None})
    return clients

def _load_client_distribution(save_dir: str) -> Optional[dict]:
    dist_path = os.path.join(save_dir, "distribution_report.json")
    if not os.path.exists(dist_path):
        return None
    with open(dist_path, encoding="utf-8") as f:
        dist = json.load(f)
    classes      = [str(c) for c in dist.get("classes", [])]
    counts       = dist.get("counts_per_client", {})
    client_files = dist.get("clients") or list(counts.keys())
    chart_data   = []
    for cls in classes:
        row = {"cls": f"C{cls}"}
        for i, cf in enumerate(client_files):
            cc = counts.get(cf, {})
            row[f"c{i+1}"] = int(cc.get(cls, cc.get(int(cls), 0)))
        chart_data.append(row)
    kl_map = dist.get("kl_divergence", {})
    return {
        "classes":      classes,
        "clients":      [f"Client {i+1}" for i in range(len(client_files))],
        "chart_data":   chart_data,
        "kl_divergence": {f"Client {i+1}": round(float(kl_map[cf]), 4)
                          for i, cf in enumerate(client_files) if cf in kl_map},
        "avg_kl": dist.get("avg_kl"),
    }

def _merge_privacy_into_rounds(rounds: List[dict], save_dir: str) -> List[dict]:
    eps_by_round: Dict[int, float] = {}
    privacy_path = os.path.join(save_dir, "fl_privacy.json")
    if os.path.exists(privacy_path):
        with open(privacy_path, encoding="utf-8") as f:
            pl = json.load(f)
        if isinstance(pl, list):
            for entry in pl:
                if not isinstance(entry, dict):
                    continue
                rnd = entry.get("round")
                eps = entry.get("avg_epsilon") or entry.get("epsilon")
                if rnd is not None and eps is not None:
                    eps_by_round[int(rnd)] = float(eps)
    if not eps_by_round:
        eps_rounds = _read_json_optional("results/epsilon_rounds.json")
        if isinstance(eps_rounds, list):
            for entry in eps_rounds:
                if not isinstance(entry, dict):
                    continue
                rnd = entry.get("fl_round") or entry.get("round")
                eps = entry.get("epsilon_consumed") or entry.get("epsilon") or entry.get("actual_epsilon")
                if rnd is not None and eps is not None:
                    eps_by_round[int(rnd)] = float(eps)
    enriched = []
    for row in rounds:
        merged = dict(row)
        rnd = merged.get("round")
        if rnd is not None and int(rnd) in eps_by_round and merged.get("epsilon") is None:
            merged["epsilon"] = eps_by_round[int(rnd)]
        enriched.append(merged)
    return enriched

def _compute_per_class_f1() -> Optional[Dict[str, float]]:
    try:
        from sklearn.metrics import f1_score
        model_path = _find_latest_model_path()
        if not os.path.exists(model_path):
            return None
        mtime = os.path.getmtime(model_path)
        if _eval_cache.get("mtime") == mtime and _eval_cache.get("per_class_f1"):
            return _eval_cache["per_class_f1"]
        wrapper, meta, preprocessing = _get_model()
        wrapper.eval()
        global_mean = preprocessing["mean"] if preprocessing else None
        global_std  = preprocessing["std"]  if preprocessing else None
        num_classes = meta["num_classes"]
        class_names = meta["class_names"]
        all_true, all_pred = [], []
        for csv_path in get_default_client_csvs():
            if not os.path.exists(csv_path):
                continue
            _, test_loader, _ = load_csv_data(
                csv_path, batch_size=64,
                global_mean=global_mean, global_std=global_std,
                num_classes=num_classes, class_names=class_names,
            )
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    preds = torch.argmax(wrapper(x_batch), dim=1).cpu().numpy()
                    all_pred.extend(preds.tolist())
                    all_true.extend(y_batch.numpy().tolist())
        if not all_true:
            return None
        per_class = f1_score(all_true, all_pred, average=None, zero_division=0, labels=list(range(num_classes)))
        result = {class_names[i]: round(float(v), 4) for i, v in enumerate(per_class)}
        _eval_cache.update({"mtime": mtime, "per_class_f1": result})
        return result
    except Exception as exc:
        print(f"[Dashboard] per_class_f1 evaluation failed: {exc}")
        return None

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/token")
async def token(request: Request):
    form_data = await request.form()
    username  = str(form_data.get("username", ""))
    password  = str(form_data.get("password", ""))
    user      = _users.get(username)
    if not user or user.get("password") != password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    tok = secrets.token_urlsafe(32)
    ISSUED_TOKEN_STORE[tok] = {"username": username, "role": user.get("role", "viewer")}
    return {"access_token": tok, "token_type": "bearer", "role": user.get("role")}

@app.get("/status", response_model=StatusResponse)
def status(user=Depends(get_current_user)):
    status_obj = _read_json("status.json")
    save_dir   = _get_save_dir(status_obj)
    clients    = status_obj.get("clients")
    if isinstance(clients, int):
        status_obj["clients"] = _load_client_info(save_dir, clients)
    elif isinstance(clients, list):
        needs_enrichment = not clients or all(not c.get("samples") for c in clients if isinstance(c, dict))
        if needs_enrichment:
            status_obj["clients"] = _load_client_info(save_dir, len(clients) or 3)
    status_obj["client_distribution"] = _load_client_distribution(save_dir)
    if "epsilon" not in status_obj:
        try:
            ppath = _find_privacy_log_path()
            if os.path.exists(ppath):
                with open(ppath, encoding="utf-8") as pf:
                    pl = json.load(pf)
                def _find_eps(obj):
                    if isinstance(obj, dict):
                        for k in ("epsilon","actual_epsilon","cumulative_epsilon","spent_epsilon","final_epsilon","total_epsilon"):
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
    data       = _read_json("results/results.json")
    status_obj = _read_json_optional("status.json") or {}
    save_dir   = _get_save_dir(status_obj)
    rounds     = data.get("rounds") or []
    if rounds:
        data["rounds"] = _merge_privacy_into_rounds(rounds, save_dir)
    if not data.get("per_class_f1"):
        per_class = _compute_per_class_f1()
        if per_class:
            data["per_class_f1"] = per_class
    return data

@app.get("/attestation")
def attestation(user=Depends(get_current_user)):
    return _read_json("attestation.json")

@app.get("/benchmarks")
def benchmarks(user=Depends(get_current_user)):
    return _read_json("results/benchmarks_baseline.json")

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request, user=Depends(require_authenticated)):
    client_ip = request.client.host
    _check_rate_limit(client_ip)
    _check_predict_rate_limit(client_ip)
    model, meta, preprocessing = _get_model()
    expected_dim = meta["input_dim"]
    class_names  = meta["class_names"]
    if len(payload.features) != expected_dim:
        raise HTTPException(status_code=422, detail=f"Expected {expected_dim} features, got {len(payload.features)}")
    if preprocessing is None:
        raise HTTPException(status_code=503, detail="Model preprocessing metadata is missing.")
    mean = preprocessing["mean"]
    std  = preprocessing["std"]
    if len(mean) != expected_dim or len(std) != expected_dim:
        raise HTTPException(status_code=500, detail="Invalid preprocessing metadata: feature dimensions mismatch.")
    x = torch.tensor(payload.features, dtype=torch.float32)
    # Bug 6 fix: use local tensors instead of _model_cache directly
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t  = torch.clamp(torch.tensor(std, dtype=torch.float32), min=1e-8)
    x = (x - mean_t) / std_t
    x = x.unsqueeze(0)

    def _round_probs(probs_tensor):
        if _prob_rounding_step <= 0:
            return probs_tensor
        rounded = torch.round(probs_tensor / _prob_rounding_step) * _prob_rounding_step
        rounded = torch.clamp(rounded, min=0.0)
        total = float(rounded.sum().item())
        return rounded / total if total > 0 else probs_tensor
    def _maybe_randomize(pred_idx, probs_tensor):
        if user.get("role") == "admin" and not payload.randomized_response:
            return pred_idx
        if _random_response_prob <= 0 or random.random() >= _random_response_prob:
            return pred_idx
        choices = [i for i in range(probs_tensor.numel()) if i != pred_idx]
        return random.choice(choices) if choices else pred_idx
    with torch.no_grad():
        probs = model(x).squeeze()
    probs      = _round_probs(probs)
    pred_class = int(probs.argmax().item())
    pred_class = _maybe_randomize(pred_class, probs)
    pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    confidence = None
    if user.get("role") == "admin" and payload.return_confidence:
        confidence = round(min(float(probs[pred_class].item()), 0.6), 2)
    return PredictResponse(predicted_class=pred_class, predicted_label=pred_label, confidence=confidence)

@app.get("/attacks", response_model=AttacksResponse)
def attacks(user=Depends(get_current_user)):
    """
    Return attack evaluation summaries.

    Uses _best_attack_path() to walk through defended → mitigated → unmitigated
    variants so the dashboard always shows the RESISTANT defended verdict.
    """
    out = {}
    attack_files = {
        "model_inversion":      "results/attacks/model_inversion.json",
        "membership_inference": "results/attacks/membership_inference.json",
        "gradient_poisoning":   "results/attacks/gradient_poisoning.json",
    }
    for key, rel_path in attack_files.items():
        best_path = _best_attack_path(rel_path)
        if os.path.exists(best_path):
            with open(best_path, encoding="utf-8") as f:
                data = json.load(f)
            summary = _normalize_attack_summary(data.get("summary", {}))
            if isinstance(summary, dict):
                variant = os.path.basename(best_path).replace(".json", "").split("_")[-1]
                if variant in ("defended", "mitigated"):
                    summary["_source"] = variant
            out[key] = summary
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
                            entry["epsilon"] = entry["avg_epsilon"]
                        else:
                            clients = entry.get("clients")
                            if isinstance(clients, list) and clients and isinstance(clients[0], dict):
                                entry["epsilon"] = clients[0].get("epsilon")
                out.append(entry)
            return out
        return pl
    legacy_path = os.path.join(ROOT, "results", "privacy_log.json")
    if os.path.exists(legacy_path):
        with open(legacy_path, encoding="utf-8") as f:
            pl = json.load(f)
        if isinstance(pl, list):
            return [dict(e, epsilon=e.get("epsilon") or e.get("avg_epsilon")) for e in pl if isinstance(e, dict)]
        return pl
    eps_rounds = os.path.join(ROOT, "results", "epsilon_rounds.json")
    if os.path.exists(eps_rounds):
        try:
            with open(eps_rounds, encoding="utf-8") as f:
                rounds = json.load(f)
            out = []
            for r in rounds:
                eps = r.get("epsilon_consumed") or r.get("epsilon") or r.get("actual_epsilon")
                out.append({"round": r.get("fl_round") or r.get("round"), "epsilon": eps})
            if out:
                return out
        except Exception:
            pass
    return []

@app.get("/query_stats")
def query_stats(request: Request, user=Depends(get_current_user)):
    client_ip = request.client.host
    now       = time.time()
    recent    = [t for t in _query_log.get(client_ip, []) if t > now - RATE_LIMIT_WINDOW]
    return {"client_ip": client_ip, "queries_in_window": len(recent),
            "limit": RATE_LIMIT_MAX, "window_seconds": RATE_LIMIT_WINDOW,
            "remaining": max(0, RATE_LIMIT_MAX - len(recent))}

@app.get("/model_info")
def model_info(user=Depends(get_current_user)):
    model_path    = _find_latest_model_path()
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    meta          = _load_model_meta(model_path)
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
        "mi_defence":              {"enabled": _defence_on, "noise_scale": _noise_scale, "temperature": _temperature},
    }

@app.get("/local_models")
def local_models(user=Depends(get_current_user)):
    """List which client local models exist on disk."""
    available = []
    for i in range(1, 4):
        path = os.path.join(ROOT, "results", "local_models", f"client{i}_local_model.pth")
        if os.path.exists(path):
            available.append({"client_id": i, "exists": True,
                               "size_kb": round(os.path.getsize(path) / 1024, 1),
                               "modified": os.path.getmtime(path)})
        else:
            available.append({"client_id": i, "exists": False})
    return {"local_models": available}