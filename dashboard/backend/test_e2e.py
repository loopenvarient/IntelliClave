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
features = [0.0] * 50
r = client.post("/predict", json={"features": features})
body = r.json() if r.status_code in (200, 422) else {}
# The API returns predicted_label and predicted_class; confidence is omitted
# when return_confidence=False (field is absent or null depending on Pydantic version)
check(
    "POST /predict (50 features) → 200 + predicted_label present",
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
r = client.post("/predict", json={"features": [0.0] * 10})
check(
    "POST /predict (10 features) → 422 Unprocessable",
    r.status_code == 422,
    f"status={r.status_code} (expected 422)",
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 10: POST /predict — predicted label is a valid activity name
# ─────────────────────────────────────────────────────────────────────────────
VALID_LABELS = {
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
}
r = client.post("/predict", json={"features": features, "return_confidence": True})
body = r.json() if r.status_code == 200 else {}
check(
    "POST /predict → predicted_label is a valid HAR activity",
    r.status_code == 200 and body.get("predicted_label") in VALID_LABELS,
    f"label={body.get('predicted_label')} valid_labels={VALID_LABELS}",
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
