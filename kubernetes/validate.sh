#!/bin/bash
# ── IntelliClave K8s YAML Validation ─────────────────────────────────────────
# Runs kubectl apply --dry-run=client on every YAML in the kubernetes/ folder.
# No cluster connection required — validates syntax and schema only.
#
# Usage:
#   bash kubernetes/validate.sh
#
# Expected output for each file:
#   namespace/intelliclave configured (dry run)
#   deployment.apps/fl-server configured (dry run)
#   ... etc
#
# Exit code 0 = all valid. Non-zero = at least one file failed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0
FAIL=0
ERRORS=()

echo "======================================================="
echo "IntelliClave — Kubernetes YAML Dry-Run Validation"
echo "======================================================="
echo ""

# Apply namespace first (other resources depend on it)
FILES=(
  "$SCRIPT_DIR/namespace.yaml"
  "$SCRIPT_DIR/volumes/server-pvc.yaml"
  "$SCRIPT_DIR/volumes/client-pvcs.yaml"
  "$SCRIPT_DIR/volumes/shared-storage.yaml"
  "$SCRIPT_DIR/services/fl-server-service.yaml"
  "$SCRIPT_DIR/policies/network-policy.yaml"
  "$SCRIPT_DIR/deployments/fl-server.yaml"
  "$SCRIPT_DIR/deployments/fl-client-1.yaml"
  "$SCRIPT_DIR/deployments/fl-client-2.yaml"
  "$SCRIPT_DIR/deployments/fl-client-3.yaml"
  "$SCRIPT_DIR/deployments/redis.yaml"
  "$SCRIPT_DIR/deployments/dashboard.yaml"
)

for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "  SKIP  $f (not found)"
    continue
  fi

  RESULT=$(kubectl apply --dry-run=client -f "$f" 2>&1)
  EXIT=$?

  BASENAME=$(basename "$f")
  if [ $EXIT -eq 0 ]; then
    echo "  ✓  $BASENAME"
    echo "     $RESULT"
    PASS=$((PASS + 1))
  else
    echo "  ✗  $BASENAME"
    echo "     $RESULT"
    FAIL=$((FAIL + 1))
    ERRORS+=("$BASENAME")
  fi
done

echo ""
echo "======================================================="
echo "Results: $PASS passed, $FAIL failed"

if [ $FAIL -eq 0 ]; then
  echo "ALL YAML FILES VALID ✓"
  echo "======================================================="
  exit 0
else
  echo "FAILED FILES:"
  for e in "${ERRORS[@]}"; do
    echo "  - $e"
  done
  echo "======================================================="
  exit 1
fi
