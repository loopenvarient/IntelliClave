# security/

Adversarial security testing and attack simulations.

## Purpose

This folder runs **offline attack experiments** against trained models and FL behaviour. Results quantify how well IntelliClave resists model inversion, membership inference, and gradient poisoning. Output JSON files feed the dashboard evaluation panel and project reports.

These scripts are **lab tests**, not part of the live training loop.

## Key files

| Path | Role |
|------|------|
| `attacks/model_inversion.py` | Reconstruct class features from model outputs (cosine similarity metric) |
| `attacks/membership_inference.py` | Infer train vs test membership from confidence gap (AUC metric) |
| `attacks/gradient_poisoning.py` | Label-flip Byzantine sweep (FedAvg baseline + `--robust` trimmed mean) |
| `attacks/build_surrogate.py` | Train surrogate model from public features |
| `attacks/summarize_attack_results.py` | Compact comparison table across attack JSONs |

## Attack result files (written to `results/attacks/`)

| File | Meaning |
|------|---------|
| `model_inversion.json` | Unmitigated model — shows threat is real |
| `model_inversion_defended.json` | With PrivacyWrapper (noise + temperature) — RESISTANT |
| `membership_inference.json` | DP-trained model — near-random AUC |
| `gradient_poisoning.json` | FedAvg — VULNERABLE at 100% label-flip |
| `gradient_poisoning_robust.json` | Trimmed mean — RESISTANT |

## How it connects

- Loads checkpoints from **`results/fl_rounds/`**
- Uses defence constants from **`config/constants.py`**
- Dashboard **`/attacks`** reads summaries from `results/attacks/`
- Documented in **`report/security_report.md`** and **`report/stride_*.md`**

## Commands

```bash
python security/attacks/model_inversion.py
python security/attacks/membership_inference.py
python security/attacks/gradient_poisoning.py              # FedAvg baseline
python security/attacks/gradient_poisoning.py --robust   # trimmed mean defence
python security/attacks/summarize_attack_results.py
```
