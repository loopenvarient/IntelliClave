#!/usr/bin/env bash
# Seal checkpoints, verify no plaintext exposure, run black-box inversion assert.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export INFERENCE_EPSILON="${INFERENCE_EPSILON:-4.0}"
export LIFETIME_QUERY_BUDGET="${LIFETIME_QUERY_BUDGET:-10000}"
export RATE_LIMIT_MAX="${RATE_LIMIT_MAX:-100}"
export RATE_LIMIT_WINDOW="${RATE_LIMIT_WINDOW:-60}"
exec python scripts/harden_production.py --all "$@"
