# privacy/

Differential privacy layer (Opacus DP-SGD).

## Purpose

This folder wraps local client training with **DP-SGD**: gradient clipping, noise injection, and a tracked **privacy budget (ε)**. It also runs privacy–utility experiments (ε sweeps) to quantify accuracy cost vs privacy gain.

## Key files

| Path | Role |
|------|------|
| `dp_trainer.py` | `DPTrainer` — Opacus `PrivacyEngine` wrapper |
| `budget_monitor.py` | Per-client ε audit logic |
| `run_budget_monitor.py` | CLI to inspect ε spend from a run's `fl_privacy.json` |
| `epsilon_sweep.py` | Train/evaluate across multiple ε values |
| `dp_flower_client.py` | Standalone DP Flower client example |
| `opacus_smoke_test.py` | Verify Opacus compatibility |
| `validate_model.py` | Model/dataset validation helpers |

## How it connects

- Activated from **`fl/run_client.py --dp --epsilon 10`** and **`fl/run_fl_simulation.py --dp`**
- Clients report ε per round in fit metrics → server writes **`results/fl_rounds/run_*/fl_privacy.json`**
- Outputs **`results/epsilon_sweep.json`**, **`results/epsilon_rounds.json`**, **`results/privacy_log.json`**
- Dashboard **`/privacy_log`** and privacy budget charts read these files

## Commands

```bash
python privacy/epsilon_sweep.py --epsilons 1 2 5 10 20
python privacy/run_budget_monitor.py --max-epsilon 10.0 \
    --privacy-json results/fl_rounds/run_YYYYMMDD_HHMMSS/fl_privacy.json
python privacy/opacus_smoke_test.py
```

## Typical result

Reference config **ε = 10** trades roughly **~5–6% accuracy** vs no-DP baseline while keeping membership-inference attacks near random (AUC ≈ 0.5).
