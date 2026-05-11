"""
dashboard/backend/test_e2e.py

End-to-end test for the IntelliClave Dashboard API.

Tests every endpoint and confirms correct responses without needing
a running server — uses FastAPI's TestClient (no network required).

Endpoints tested:
    GET  /health          → {"status": "ok"}
    GET  /status          → round, epsilon, clients, etc.
    GET  /results         → rounds list with macro_f1, accuracy, epsilon
    GET  /attestation     → tee_verified, mrenclave, status
    GET  /benchmarks      → tee_overhead_ms list
    GET  /query_stats     → rate limit info
    POST /predict         → predicted_class, predicted_label (label only)
    POST /predict         → with return_confidence=true → confidence included
    POST /predict         → wrong feature count → 422
    POST /predict × 101   → rate limit → 429

Run:
    python3 dashboard/backend/test_e2e.py

Requirements:
    pip install fastapi httpx pytest
    (httpx is used by FastAPI TestClient)
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "fl"))

PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"
results = []

print("=" * 55)
print("IntelliClave — Dashboard API End-to-End Tests")
print("=" * 55)

# ── Import FastAPI test client ─────────────────────────────────────────────────
try:
    from fastapi.testclient import TestClient
except ImportError:
    print("ERROR: fastapi not installed. Run: pip install fastapi httpx")
    sys.exit(1)

# Import the app — this also imports torch and the model
try:
    from main import app, _query_log
except ImportError as e:
    print(f"ERROR: Could not import dashboard app: {e}")
    print("       Run this script from the dashboard/backend/ directory")
    print("       or ensure fl/model.py is on the Python path.")
    sys.exit(1)

client = TestClient(app, raise_server_exceptions=False)


# ── Helper ─────────────────────────────────────────────────────────────────────
def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"\n[Test] {name}")
    if detail:
        print(f"  {detail}")
    print(status)
    results.append((name, condition))
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: GET /health
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/health")
check(
    "GET /health → 200 + status ok",
    r.status_code == 200 and r.json().get("status") == "ok",
    f"status={r.status_code} body={r.json()}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: GET /status
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/status")
body = r.json() if r.status_code == 200 else {}
check(
    "GET /status → 200 + has round + clients",
    r.status_code == 200
    and "round" in body
    and "clients" in body,
    f"status={r.status_code} keys={list(body.keys())}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: GET /results
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/results")
body = r.json() if r.status_code == 200 else {}
check(
    "GET /results → 200 + has rounds list",
    r.status_code == 200 and "rounds" in body and isinstance(body["rounds"], list),
    f"status={r.status_code} keys={list(body.keys())}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: GET /attestation
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/attestation")
body = r.json() if r.status_code == 200 else {}
check(
    "GET /attestation → 200 + tee_verified + status VERIFIED",
    r.status_code == 200
    and body.get("tee_verified") is True
    and body.get("status") == "VERIFIED",
    f"status={r.status_code} tee_verified={body.get('tee_verified')} "
    f"att_status={body.get('status')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: GET /benchmarks
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/benchmarks")
body = r.json() if r.status_code == 200 else {}
check(
    "GET /benchmarks → 200 + tee_overhead_ms list",
    r.status_code == 200 and "tee_overhead_ms" in body,
    f"status={r.status_code} keys={list(body.keys())}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 6: GET /query_stats
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/query_stats")
body = r.json() if r.status_code == 200 else {}
check(
    "GET /query_stats → 200 + limit + remaining",
    r.status_code == 200
    and "limit" in body
    and "remaining" in body,
    f"status={r.status_code} body={body}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 7: POST /predict — label only (default, no confidence)
# ─────────────────────────────────────────────────────────────────────────────
# Determine the expected feature dimension from the model meta or processed CSVs
import json as _json
import pandas as _pd_module
_meta_path = os.path.join(_ROOT, "results", "fl_rounds", "model_meta.json")
if os.path.exists(_meta_path):
    with open(_meta_path) as _f:
        _meta = _json.load(_f)
    _expected_dim = _meta["input_dim"]
    _valid_labels = set(_meta.get("class_names", []))
else:
    # Fallback: infer from first processed CSV
    _processed = os.path.join(_ROOT, "data", "processed")
    _csvs = sorted(f for f in os.listdir(_processed) if f.endswith(".csv"))
    _df = _pd_module.read_csv(os.path.join(_processed, _csvs[0]), nrows=1)
    _expected_dim = len([c for c in _df.columns if c != "label"])
    _valid_labels = set()

features = [0.0] * _expected_dim
r = client.post("/predict", json={"features": features})
body = r.json() if r.status_code in (200, 422) else {}
check(
    "POST /predict (correct feature count) → 200 + predicted_label present",
    r.status_code == 200
    and "predicted_label" in body
    and "predicted_class" in body,
    f"status={r.status_code} label={body.get('predicted_label')} "
    f"confidence={body.get('confidence')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 8: POST /predict — with return_confidence=true
# ─────────────────────────────────────────────────────────────────────────────
r = client.post("/predict", json={"features": features, "return_confidence": True})
body = r.json() if r.status_code == 200 else {}
check(
    "POST /predict (return_confidence=true) → confidence is float",
    r.status_code == 200
    and isinstance(body.get("confidence"), float)
    and 0.0 <= body["confidence"] <= 1.0,
    f"status={r.status_code} confidence={body.get('confidence')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 9: POST /predict — wrong feature count → 422
# ─────────────────────────────────────────────────────────────────────────────
wrong_dim = max(1, _expected_dim - 10)
r = client.post("/predict", json={"features": [0.0] * wrong_dim})
check(
    "POST /predict (wrong feature count) → 422 Unprocessable",
    r.status_code == 422,
    f"status={r.status_code} (expected 422)",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 10: POST /predict — predicted label is a non-empty string
# ─────────────────────────────────────────────────────────────────────────────
r = client.post("/predict", json={"features": features, "return_confidence": True})
body = r.json() if r.status_code == 200 else {}
pred_label = body.get("predicted_label", "")
label_ok = isinstance(pred_label, str) and len(pred_label) > 0
if _valid_labels:
    label_ok = label_ok and pred_label in _valid_labels
check(
    "POST /predict → predicted_label is a valid non-empty string",
    r.status_code == 200 and label_ok,
    f"label={pred_label!r}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Rate limiter — 101 requests → 429
# ─────────────────────────────────────────────────────────────────────────────
# Reset the rate limit log for a clean test
_query_log.clear()
last_status = None
for i in range(101):
    r = client.post("/predict", json={"features": features})
    last_status = r.status_code
check(
    "POST /predict × 101 → rate limit triggers 429",
    last_status == 429,
    f"last response status={last_status} (expected 429 after 100 requests)",
)
_query_log.clear()  # reset after test

# ─────────────────────────────────────────────────────────────────────────────
# Test 12: GET /attacks → attack summaries
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/attacks")
body = r.json() if r.status_code == 200 else {}
check(
    "GET /attacks → 200 + model_inversion + membership_inference + gradient_poisoning",
    r.status_code == 200
    and "model_inversion" in body
    and "membership_inference" in body
    and "gradient_poisoning" in body,
    f"status={r.status_code} keys={list(body.keys())}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 13: GET /privacy_log → per-client epsilon log
# ─────────────────────────────────────────────────────────────────────────────
r = client.get("/privacy_log")
body = r.json() if r.status_code == 200 else {}
check(
    "GET /privacy_log → 200 + list with epsilon entries",
    r.status_code == 200
    and isinstance(body, list)
    and len(body) > 0
    and "epsilon" in body[0],
    f"status={r.status_code} entries={len(body) if isinstance(body, list) else 'N/A'}",
)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"Results: {passed}/{total} tests passed")
for name, ok in results:
    icon = "✓" if ok else "✗"
    print(f"  {icon}  {name}")

if passed == total:
    print()
    print("ALL E2E TESTS PASSED ✓")
else:
    print()
    print(f"FAILED: {total - passed} test(s)")
    sys.exit(1)

print("=" * 55)
