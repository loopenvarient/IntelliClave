# fl/

Core federated learning engine (Flower + PyTorch).

## Purpose

This is the **main training stack**. It runs FedAvg or FedProx across multiple clients, saves global model checkpoints, writes dashboard status files, and optionally integrates DP, crypto, and TEE attestation.

## Key files

| Path | Role |
|------|------|
| `model.py` | MLP, ResNet-tabular, Transformer-tabular, `PrivacyWrapper` (inference defence) |
| `data_utils.py` | CSV loading, schema inference, global normalization, preprocessing metadata |
| `fl_client.py` | Flower client — local training, optional DP (Opacus), defended eval |
| `fl_server.py` | Flower server — aggregation, crypto decrypt, early stopping, checkpoint save |
| `run_server.py` | Launch distributed server (`--crypto`, `--attest`, `--strategy fedprox`) |
| `run_client.py` | Launch a client (`--dp`, `--epsilon`, `--crypto`, `--attest`) |
| `run_fl_simulation.py` | Single-process FL simulation (no separate server process) |
| `train_local.py` | Standalone single-client baseline / DP training |
| `evaluate_global_model.py` | Evaluate saved checkpoint on all client test splits |

## How it connects

- Reads **`data/processed/`**, **`config/constants.py`**
- DP via **`privacy/dp_trainer.py`** when `--dp` is set on clients
- Crypto via **`crypto/certs/`** when `--crypto` is set
- Attestation via **`tee/attestation/`** when `--attest` is set
- Writes **`status.json`**, **`results/results.json`**, **`results/fl_rounds/run_*/`**
- Checkpoints consumed by **`dashboard/backend/`** and **`security/attacks/`**

## Commands

```bash
# Simulation (easiest)
python fl/run_fl_simulation.py --rounds 35 --clients 3 --strategy fedprox --dp --epsilon 10

# Distributed full stack
python fl/run_server.py --rounds 5 --min-clients 3 --strategy fedprox --crypto --attest
python fl/run_client.py --id 1 --dp --epsilon 10 --rounds 5 --crypto --attest
python fl/run_client.py --id 2 --dp --epsilon 10 --rounds 5 --crypto --attest
python fl/run_client.py --id 3 --dp --epsilon 10 --rounds 5 --crypto --attest

# Evaluate saved model
python fl/evaluate_global_model.py --checkpoint results/fl_rounds/run_*/global_model_latest.pth

# Local vs global FL utility comparison (dashboard /comparison)
python fl/compare_local_vs_global.py --dp --epsilon 10 --retrain
```

## Note on aggregation vs attacks

Live training here uses **FedAvg/FedProx weighted averaging** in `fl_server.py`. Byzantine defences such as trimmed mean are tested separately in **`security/attacks/gradient_poisoning.py`**, not in this server aggregation path yet.
