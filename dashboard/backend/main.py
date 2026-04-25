"""
IntelliClave Dashboard — FastAPI backend
"""
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IntelliClave Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Works whether run from project root or backend folder
_here = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(_here, '..', '..'))


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
