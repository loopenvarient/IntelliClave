# scripts/

Windows PowerShell orchestration helpers.

## Purpose

This folder chains the full IntelliClave pipeline into single commands: smoke tests, end-to-end experiment runs, distributed FL in separate terminals, and dashboard startup. Designed for **Windows demo day** and repeatable validation.

## Key files

| Script | What it does |
|--------|--------------|
| `test_pipeline.ps1` | Fast validation: data check, crypto, attestation, sealed storage, dashboard E2E, Opacus smoke, local training |
| `run_full_pipeline.ps1` | Full experiment chain: optional repartition → baseline FL → DP FL → eval → privacy monitor → CV → ε sweep → graph → attacks |
| `start_dashboard.ps1` | Opens backend (port 8001) + frontend (port 5173) in separate cmd windows |
| `run_distributed_fl.ps1` | Opens server + N client terminals with `-Dp`, `-Crypto`, `-Attest` flags |

## How it connects

Invokes modules across **`data/`**, **`fl/`**, **`privacy/`**, **`evaluation/`**, **`security/`**, **`dashboard/`**, **`tee/`**.

Writes logs to **`results/pipeline_tests/`** and **`results/pipeline_runs/`**.

## Commands

```powershell
# Quick smoke test
powershell -ExecutionPolicy Bypass -File scripts/test_pipeline.ps1

# Skip slow steps
powershell -ExecutionPolicy Bypass -File scripts/test_pipeline.ps1 -SkipSlow

# Full pipeline (5 rounds, ε=10)
powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1 -Rounds 5 -Epsilon 10.0

# Repartition data + skip attacks
powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1 -RepartitionData -SkipAttacks

# Dashboard only
powershell -ExecutionPolicy Bypass -File scripts/start_dashboard.ps1

# Distributed FL with full stack
powershell -ExecutionPolicy Bypass -File scripts/run_distributed_fl.ps1 -Dp -Crypto -Attest -Rounds 5
```
