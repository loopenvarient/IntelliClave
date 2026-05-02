"""
IntelliClave Dashboard — FastAPI backend
"""
import json
import os
import sys
import time
from collections import defaultdict
from typing import List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── model imports ─────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'fl'))
from model import get_model  # noqa: E402
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="IntelliClave Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Rate limiter (mitigation 3) ───────────────────────────────────────────────
# Max queries per client IP per window
RATE_LIMIT_MAX     = 100   # max requests
RATE_LIMIT_WINDOW  = 60    # seconds
_query_log: dict   = defaultdict(list)   # ip → [timestamps]

def _check_rate_limit(client_ip: str):
    now    = time.time()
    window = now - RATE_LIMIT_WINDOW
    # Keep only timestamps within the current window
    _query_log[client_ip] = [t for t in _query_log[client_ip] if t > window]
    if len(_query_log[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX} queries "
                   f"per {RATE_LIMIT_WINDOW}s. Try again later."
        )
    _query_log[client_ip].append(now)
# ─────────────────────────────────────────────────────────────────────────────

# ── Model loader (cached) ─────────────────────────────────────────────────────
_model_cache = {}

ACTIVITY_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]

def _get_model():
    if "model" not in _model_cache:
        model_path = os.path.join(ROOT, "results", "fl_rounds", "global_model_latest.pth")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail="Model not trained yet.")
        model = get_model(input_dim=50, num_classes=6)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        _model_cache["model"] = model
    return _model_cache["model"]
# ─────────────────────────────────────────────────────────────────────────────

# ── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]          # 50-dim PCA feature vector
    return_confidence: bool = False  # if False → label only (mitigation 2)

class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float = None       # only populated if return_confidence=True
# ─────────────────────────────────────────────────────────────────────────────


def _read_json(rel_path: str) -> dict:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{rel_path} not found")
    with open(path) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return _read_json("status.json")


@app.get("/results")
def results():
    return _read_json("results/results.json")


@app.get("/attestation")
def attestation():
    return _read_json("attestation.json")


@app.get("/benchmarks")
def benchmarks():
    return _read_json("results/benchmarks_baseline.json")


# ── Inference endpoint with both mitigations ──────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    """
    Run inference on a 50-dim PCA feature vector.

    Mitigations against model inversion:
      2. Confidence masking — by default only returns the predicted class label,
         not the full softmax distribution. Set return_confidence=true to get
         the top-1 confidence score (still no full probability vector exposed).
      3. Rate limiting — max 100 queries per IP per 60 seconds.
         Exceeding this returns HTTP 429.
    """
    # Mitigation 3: rate limit by client IP
    client_ip = request.client.host
    _check_rate_limit(client_ip)

    if len(payload.features) != 50:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 50 features, got {len(payload.features)}"
        )

    model = _get_model()
    x = torch.tensor(payload.features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()

    pred_class = int(probs.argmax().item())
    pred_label = ACTIVITY_NAMES[pred_class]

    # Mitigation 2: only expose top-1 confidence if explicitly requested,
    # never expose the full probability vector
    confidence = None
    if payload.return_confidence:
        confidence = round(float(probs[pred_class].item()), 4)

    return PredictResponse(
        predicted_class=pred_class,
        predicted_label=pred_label,
        confidence=confidence,
    )


@app.get("/query_stats")
def query_stats(request: Request):
    """Show rate limit status for the calling IP."""
    client_ip = request.client.host
    now    = time.time()
    window = now - RATE_LIMIT_WINDOW
    recent = [t for t in _query_log.get(client_ip, []) if t > window]
    return {
        "client_ip":       client_ip,
        "queries_in_window": len(recent),
        "limit":           RATE_LIMIT_MAX,
        "window_seconds":  RATE_LIMIT_WINDOW,
        "remaining":       max(0, RATE_LIMIT_MAX - len(recent)),
    }

