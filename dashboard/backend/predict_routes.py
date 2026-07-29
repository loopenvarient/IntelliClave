"""
predict_routes.py  — fixed version
"""

import glob as _glob
import io, json, os, sys
from typing import List, Optional
from collections import Counter

import torch
import pandas as pd
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

_here = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "fl"))
sys.path.insert(0, os.path.join(ROOT, "config"))

try:
    from data_utils import load_preprocessing_metadata, infer_default_preprocessing
    from model import get_model, get_defended_model
except ImportError as e:
    raise RuntimeError(f"Cannot import FL modules — check sys.path: {e}")

# Bug 2 fix: prefix router to avoid /local_models collision with main.py
predict_router = APIRouter()

# Bug 1 fix: resolve defence constants without importing main.py
try:
    from constants import MI_NOISE_SCALE, MI_TEMPERATURE, MI_DEFENCE_ENABLED
    _def_noise, _def_temp, _def_defence = MI_NOISE_SCALE, MI_TEMPERATURE, MI_DEFENCE_ENABLED
except ImportError:
    _def_noise, _def_temp, _def_defence = 0.5, 4.0, True

_noise_scale = float(os.environ.get("MI_NOISE_SCALE",    _def_noise))
_temperature = float(os.environ.get("MI_TEMPERATURE",    _def_temp))
_defence_on  = os.environ.get("MI_DEFENCE_ENABLED", str(_def_defence)).lower() not in ("0", "false")


def _load_global_prep(model_path: str = None):
    candidates = []

    # Priority 1: same directory as model
    if model_path:
        p = os.path.join(os.path.dirname(model_path), "global_normalization.json")
        if os.path.exists(p):
            candidates.append(p)

    # Priority 2: scan run_* dirs and flat location
    candidates += (
        _glob.glob(os.path.join(ROOT, "results", "fl_rounds", "run_*", "global_normalization.json"))
        + [
            os.path.join(ROOT, "results", "fl_rounds", "global_normalization.json"),
            os.path.join(ROOT, "results", "preprocessing.json"),
        ]
    )

    existing = [p for p in candidates if os.path.exists(p)]
    if not existing:
        raise HTTPException(status_code=503, detail="Preprocessing metadata not found. Run FL training first.")

    best = existing[0] if model_path else max(existing, key=os.path.getmtime)
    with open(best) as f:
        d = json.load(f)
    mean = torch.tensor(d["mean"], dtype=torch.float32)
    std  = torch.clamp(torch.tensor(d["std"], dtype=torch.float32), min=1e-8)
    return mean, std, len(d["mean"])


def _load_model_meta(model_path: str = None):
    """
    Load model_meta.json from the same directory as model_path.
    Falls back to scanning run_* dirs by mtime, then CSV inference.
    """
    # Priority 1: same directory as the model file
    if model_path:
        meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)

    # Priority 2: most recently modified model_meta.json across all run dirs
    candidates = (
        _glob.glob(os.path.join(ROOT, "results", "fl_rounds", "run_*", "model_meta.json"))
        + _glob.glob(os.path.join(ROOT, "results", "fl_rounds", "model_meta.json"))
    )
    existing = [p for p in candidates if os.path.exists(p)]
    if existing:
        # sort by mtime, pick most recent — NOT alphabetical
        best = max(existing, key=os.path.getmtime)
        with open(best, encoding="utf-8") as f:
            return json.load(f)

    # Priority 3: infer from processed CSVs (last resort)
    processed = os.path.join(ROOT, "data", "processed")
    if os.path.isdir(processed):
        csvs = sorted(f for f in os.listdir(processed) if f.endswith(".csv"))
        if csvs:
            df = pd.read_csv(os.path.join(processed, csvs[0]), nrows=1)
            input_dim = len([c for c in df.columns if c != "label"])
            df_full = pd.read_csv(os.path.join(processed, csvs[0]), usecols=["label"])
            num_classes = int(df_full["label"].nunique())
            return {
                "input_dim":   input_dim,
                "num_classes": num_classes,
                "class_names": [f"class_{i}" for i in range(num_classes)],
                "model_type":  "mlp",
            }
    raise HTTPException(status_code=503, detail="Cannot determine model shape.")


def _find_global_model() -> str:
    fl_dir = os.path.join(ROOT, "results", "fl_rounds")
    candidates = sorted(
        _glob.glob(os.path.join(fl_dir, "run_*", "global_model_latest.pth")),
        key=os.path.getmtime
    )
    if candidates:
        return candidates[-1]
    flat = os.path.join(fl_dir, "global_model_latest.pth")
    if os.path.exists(flat):
        return flat
    raise HTTPException(status_code=503, detail="Global model not found. Run FL training first.")


def _find_local_model(client_id: int) -> str:
    candidates = [
        os.path.join(ROOT, "results", "local_models", f"client{client_id}_local_model.pth"),
        os.path.join(ROOT, f"client{client_id}_local_model.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise HTTPException(
        status_code=404,
        detail=f"Local model for client {client_id} not found. "
               "Run FL training and ensure run_client.py saves local model checkpoints."
    )


_cache: dict = {}

def _get_global_model():
    path  = _find_global_model()
    mtime = os.path.getmtime(path)
    key   = f"global_{path}"
    if key not in _cache or _cache[key]["mtime"] != mtime:
        # Pass model path so meta is read from the SAME directory
        meta = _load_model_meta(model_path=path)
        mean, std, _ = _load_global_prep(model_path=path)
        model = get_defended_model(
            input_dim=meta["input_dim"], num_classes=meta["num_classes"],
            model_type=meta.get("model_type", "mlp"),
            noise_scale=_noise_scale, temperature=_temperature, enabled=_defence_on,
        )
        model.base_model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        model.eval()
        _cache[key] = {"model": model, "meta": meta, "mean": mean, "std": std, "mtime": mtime}
    return _cache[key]


def _get_local_model(client_id: int):
    path  = _find_local_model(client_id)
    mtime = os.path.getmtime(path)
    key   = f"client{client_id}_{path}"
    if key not in _cache or _cache[key]["mtime"] != mtime:
        # For local models, find meta from the nearest run directory
        run_dir_meta = None
        run_dirs = sorted(
            _glob.glob(os.path.join(ROOT, "results", "fl_rounds", "run_*", "model_meta.json")),
            key=os.path.getmtime
        )
        if run_dirs:
            run_dir_meta = run_dirs[-1]  # most recent run's meta

        meta = _load_model_meta(model_path=run_dir_meta)
        mean, std, _ = _load_global_prep(model_path=run_dir_meta)
        model = get_model(
            input_dim=meta["input_dim"], num_classes=meta["num_classes"],
            model_type=meta.get("model_type", "mlp"),
        )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        _cache[key] = {"model": model, "meta": meta, "mean": mean, "std": std, "mtime": mtime}
    return _cache[key]

def _run_inference(cached: dict, df_features: pd.DataFrame, true_labels=None):
    meta        = cached["meta"]
    mean        = cached["mean"]
    std         = cached["std"]
    model       = cached["model"]
    input_dim   = meta["input_dim"]
    num_classes = meta["num_classes"]
    class_names = meta.get("class_names", [f"class_{i}" for i in range(num_classes)])

    if df_features.shape[1] != input_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Model expects {input_dim} feature columns, CSV has {df_features.shape[1]}."
        )

    X = torch.tensor(df_features.values, dtype=torch.float32)
    X = (X - mean) / std   # std already clamped in _load_global_prep

    with torch.no_grad():
        out = model(X)
        # Bug 5 fix: check wrapper type instead of fragile probability heuristic
        if hasattr(model, 'base_model'):
            probs = out   # PrivacyWrapper already returns probabilities
        else:
            probs = torch.softmax(out, dim=1)

    pred_classes = probs.argmax(dim=1).numpy().tolist()
    confidence   = probs.max(dim=1).values.numpy().tolist()
    probs_all    = probs.numpy().tolist()

    rows = []
    for i, (pred, conf, prob_row) in enumerate(zip(pred_classes, confidence, probs_all)):
        row = {
            "row":             i,
            "predicted_class": int(pred),
            "predicted_label": class_names[pred] if pred < len(class_names) else str(pred),
            "confidence":      round(float(conf), 4),
            "probabilities":   {
                class_names[j] if j < len(class_names) else f"class_{j}": round(float(p), 4)
                for j, p in enumerate(prob_row)
            },
        }
        if true_labels is not None:
            row["true_class"] = int(true_labels[i])
            row["true_label"] = class_names[int(true_labels[i])] if int(true_labels[i]) < len(class_names) else str(true_labels[i])
            row["correct"]    = int(pred) == int(true_labels[i])
        rows.append(row)

    result = {"predictions": rows, "num_rows": len(rows), "num_classes": num_classes, "class_names": class_names}

    count = Counter(pred_classes)
    result["prediction_distribution"] = [
        {"class": i, "label": class_names[i] if i < len(class_names) else f"class_{i}",
         "count": count.get(i, 0),
         "pct":   round(count.get(i, 0) / len(rows) * 100, 1)}
        for i in range(num_classes)
    ]

    conf_by_class = {i: [] for i in range(num_classes)}
    for pred, conf in zip(pred_classes, confidence):
        conf_by_class[pred].append(conf)
    result["mean_confidence_per_class"] = [
        {"class": i, "label": class_names[i] if i < len(class_names) else f"class_{i}",
         "mean_confidence": round(float(np.mean(v)), 4) if v else 0.0}
        for i, v in conf_by_class.items()
    ]

    if true_labels is not None:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        y_true = [int(x) for x in true_labels]
        y_pred = pred_classes
        result["accuracy"]    = round(float(accuracy_score(y_true, y_pred)), 4)
        result["macro_f1"]    = round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4)
        result["weighted_f1"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
        result["per_class_f1"] = [
            {"class": i, "label": class_names[i] if i < len(class_names) else f"class_{i}",
             "f1": round(float(per_class_f1[i]), 4)}
            for i in range(num_classes)
        ]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        result["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": [class_names[i] if i < len(class_names) else f"class_{i}" for i in range(num_classes)],
        }
        correct = sum(1 for r in rows if r.get("correct"))
        result["correct"]   = correct
        result["incorrect"] = len(rows) - correct

    return result


def _parse_csv(content: bytes):
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")
    true_labels = None
    if "label" in df.columns:
        true_labels = df["label"].values
        df_features = df.drop(columns=["label"])
    else:
        df_features = df
    df_features = df_features.select_dtypes(include=[np.number])
    if df_features.empty:
        raise HTTPException(status_code=400, detail="No numeric feature columns found in CSV.")
    return df_features, true_labels


@predict_router.get("/batch_local_models")
def list_local_models():
    available = []
    for i in range(1, 4):
        try:
            path = _find_local_model(i)
            available.append({"client_id": i, "path": path, "exists": True,
                               "size_kb": round(os.path.getsize(path) / 1024, 1),
                               "modified": os.path.getmtime(path)})
        except HTTPException:
            available.append({"client_id": i, "exists": False})
    return {"local_models": available}


@predict_router.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    df_features, true_labels = _parse_csv(content)
    cached = _get_global_model()
    result = _run_inference(cached, df_features, true_labels)
    result["model_type"] = "global"
    result["filename"]   = file.filename
    return JSONResponse(content=result)


@predict_router.post("/predict_csv_client")
async def predict_csv_client(
    file: UploadFile = File(...),
    client_id: int = Query(..., ge=1, le=3, description="Client ID: 1, 2, or 3"),
):
    content = await file.read()
    df_features, true_labels = _parse_csv(content)
    cached = _get_local_model(client_id)
    result = _run_inference(cached, df_features, true_labels)
    result["model_type"] = f"client_{client_id}_local"
    result["client_id"]  = client_id
    result["filename"]   = file.filename
    return JSONResponse(content=result)