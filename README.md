# IntelliClave
## Confidential Computing Secure Processing System
> Privacy-preserving federated learning pipeline — TEE · Differential Privacy · AES-256-GCM

---

## What Is This

IntelliClave is a dataset-agnostic, privacy-preserving federated learning pipeline. The reference
implementation uses UCI HAR sensor data with three non-IID clients (FitLife, MediTrack, CareWatch),
but the entire pipeline — FL training, DP, evaluation, security attacks, and dashboard — works with
any tabular classification dataset by dropping CSVs into `data/processed/`.

Three security layers stack on top of each other:

- **Federated Learning** — raw data never leaves each client; only model updates are shared
- **Differential Privacy** — Opacus DP-SGD ensures updates cannot reveal training data (ε=10)
- **Trusted Execution Environment** — model aggregation runs inside a hardware-sealed SGX enclave

---

## Key Results (UCI HAR reference run)

| Metric | Value |
|--------|-------|
| FL baseline accuracy (10 rounds, no DP) | 96.99% |
| FL + DP accuracy (ε=10, 5 rounds) | 91.27% |
| Privacy cost | −5.72% |
| Membership inference AUC | 0.503 (near-random — resistant) |
| Gradient poisoning drop (100% flip) | −3.78% |
| TEE overhead | 35.2% avg, 0.14% of FL round time |
| Cross-validation accuracy (combined) | 96.98% ± 0.34% |
| AUC-ROC (combined) | 99.85% |

---

## Team

| Member | Responsibility |
|--------|----------------|
| Member 1 | Data pipeline · Federated Learning · Deployment |
| Member 2 | Differential Privacy · Evaluation · Dashboard |
| Member 3 | TEE · SGX · Attestation · Kubernetes · Security |

---

## Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| ML Model | PyTorch | 2.1.0 |
| Federated Learning | Flower (flwr) | 1.6.0 |
| Differential Privacy | Opacus | 1.4.0 |
| TEE | Gramine (gramine-direct) | 1.9 |
| Cryptography | AES-256-GCM + RSA-2048 + TLS 1.3 | — |
| Infrastructure | Docker + Kubernetes (minikube) | — |
| Dashboard Backend | FastAPI | 0.104.1 |
| Dashboard Frontend | React + Recharts + Vite | 18.x |

---

## Project Structure

```
intelliclave/
├── config/
│   └── constants.py        # Shared defaults (ε, n_clients, lr, batch_size, etc.)
│
├── fl/                     # Federated Learning core
│   ├── model.py            # FLClassifier (MLP) + ResNetTabular + TransformerTabular
│   ├── data_utils.py       # Generic CSV loader — any dataset, auto schema inference
│   ├── fl_client.py        # Flower client + Opacus DP + AES-256-GCM crypto
│   ├── fl_server.py        # Flower server + FedAvg/FedProx + early stopping
│   ├── run_server.py       # Server launcher (--crypto, --attest, --strategy flags)
│   ├── run_client.py       # Client launcher (--dp, --epsilon, --crypto, --attest)
│   ├── run_fl_simulation.py # Single-process simulation (no Flower server needed)
│   ├── train_local.py      # Standalone local training (baseline + DP mode)
│   └── evaluate_global_model.py  # Evaluate saved checkpoint on all client CSVs
│
├── privacy/                # Differential Privacy
│   ├── dp_trainer.py       # DPTrainer — Opacus PrivacyEngine wrapper (lr configurable)
│   ├── epsilon_sweep.py    # Privacy-utility tradeoff experiments (argparse)
│   ├── budget_monitor.py   # Per-client epsilon tracking
│   └── run_budget_monitor.py  # Budget monitor CLI (--max-epsilon, --privacy-json)
│
├── tee/                    # Trusted Execution Environment
│   ├── attestation/        # SGX attestation simulator + FL integration
│   ├── sealed_storage/     # MRENCLAVE-bound AES-256-GCM sealed storage
│   ├── fl_enclave/         # Gramine manifests (server + 3 clients)
│   ├── full_stack_test/    # PyTorch + Flower + Opacus inside Gramine
│   └── benchmarks/         # TEE overhead measurement (35.2% avg)
│
├── crypto/certs/           # Cryptographic layer
│   ├── crypto_layer.py     # AES-256-GCM + RSA-2048 weight encryption
│   ├── crypto_context.py   # Server keypair lifecycle
│   ├── generate_tls_certs.py  # TLS certificate generator
│   └── test_crypto.py      # 4 crypto tests (all pass)
│
├── security/attacks/       # Security analysis (all fully argparse-configurable)
│   ├── model_inversion.py  # Gradient-based class reconstruction
│   ├── membership_inference.py  # Threshold attack (AUC ≈ 0.5 — resistant)
│   └── gradient_poisoning.py    # Label-flip Byzantine attack sweep
│
├── evaluation/
│   ├── cross_validation.py # 5-fold stratified CV — any dataset
│   ├── metrics.py          # F1, accuracy, AUC-ROC helpers
│   └── generate_graph6.py  # 4-panel final results figure (data-driven, no hardcoding)
│
├── dashboard/
│   ├── backend/main.py     # FastAPI — 9 endpoints, 13/13 E2E tests pass
│   └── frontend/           # React + Recharts + Vite live dashboard
│
├── data/
│   ├── processed/          # Client CSVs (client1.csv, client2.csv, ...)
│   ├── datascripts/        # pipeline.py (3 input modes), weights.py, har_analysis.py
│   └── class_weights.json  # Optional per-class loss weights
│
├── kubernetes/             # K8s deployment (minikube)
│   ├── cold_start.sh       # Full cluster cold start
│   ├── deployments/        # Server + clients + SGX variants + dashboard
│   ├── policies/           # NetworkPolicy (ports 8080 + 8001)
│   ├── secrets/            # fl-crypto-keys Secret template
│   └── volumes/            # PVCs (server 1Gi, clients 100Mi each)
│
├── docker/
│   ├── Dockerfile.server   # FL server image
│   ├── Dockerfile.client   # FL client image (+ opacus)
│   ├── docker-compose.yml  # Full stack: server + 3 clients (DP + crypto)
│   └── generate_compose.py # Generate compose for N clients (--clients N)
│
├── results/                # All experiment outputs (pre-populated for demo)
│   ├── fl_rounds/          # Model checkpoints + metrics + model_meta.json
│   ├── attacks/            # Attack results (3 attacks × DP/no-DP)
│   ├── benchmarks/         # TEE overhead measurements
│   └── graphs/             # graph6_final_results.png
│
├── attestation.json        # Live attestation record
├── status.json             # Live training status (read by dashboard)
├── contracts.md            # Data contracts and interface specifications
└── requirements.txt        # Pinned Python dependencies
```

---

## Quick Start

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python (conda) | 3.10 | `conda activate intelliclave` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Docker | 24+ | `docker --version` |

### 1 — Set up the environment

```bash
git clone <repo-url>
cd IntelliClave

# Create conda environment
conda create -n intelliclave python=3.10
conda activate intelliclave
pip install -r requirements.txt

# Frontend dependencies
cd dashboard/frontend/intelliclave-ui
npm install
cd ../../..
```

> **All Python commands require `conda activate intelliclave` first.**

### 2 — Verify everything works

```bash
conda activate intelliclave

# Crypto (4/4)
python crypto/certs/test_crypto.py

# Attestation
python tee/attestation/attestation_integration.py

# Sealed storage
python tee/sealed_storage/sealed_storage.py

# Dashboard E2E (13/13) — no server needed, uses TestClient
python dashboard/backend/test_e2e.py
```

### 3 — Start the dashboard

The dashboard loads pre-recorded results immediately — no training run required.

```bash
# Terminal A — backend
conda activate intelliclave
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Terminal B — frontend
cd dashboard/frontend/intelliclave-ui
npm run dev
# Opens at http://localhost:5173
```

---

## Running FL Training

### Simulation (single process, easiest)

```bash
conda activate intelliclave

# Baseline (no DP)
python fl/run_fl_simulation.py --rounds 10 --clients 3

# With DP
python fl/run_fl_simulation.py --rounds 5 --clients 3

# FedProx + ResNet model
python fl/run_fl_simulation.py --strategy fedprox --model-type resnet-tabular
```

### Distributed (Flower server + clients)

```bash
# Terminal 1 — server
python fl/run_server.py --rounds 5 --min-clients 3

# Terminals 2, 3, 4 — clients (no DP)
python fl/run_client.py --id 1
python fl/run_client.py --id 2
python fl/run_client.py --id 3
```

### Distributed with DP

```bash
# Server
python fl/run_server.py --rounds 5 --min-clients 3

# Clients — --rounds must match server
python fl/run_client.py --id 1 --dp --epsilon 10.0 --rounds 5
python fl/run_client.py --id 2 --dp --epsilon 10.0 --rounds 5
python fl/run_client.py --id 3 --dp --epsilon 10.0 --rounds 5
```

### Full stack (DP + Crypto + Attestation)

```bash
# Server
python fl/run_server.py --rounds 5 --min-clients 3 --crypto --attest

# Clients
python fl/run_client.py --id 1 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 2 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 3 --dp --epsilon 10.0 --rounds 5 --crypto --attest
```

### Docker

```bash
# Default: 3 clients, 10 rounds, DP + crypto
docker compose -f docker/docker-compose.yml up --build

# N clients
python docker/generate_compose.py --clients 5
docker compose -f docker/docker-compose.generated.yml up --build
```

### Kubernetes (minikube)

```bash
minikube start --driver=docker --cpus=4 --memory=6144
bash kubernetes/cold_start.sh

# SGX production mode
SGX=true bash kubernetes/cold_start.sh
```

---

## Using Your Own Dataset

IntelliClave works with any tabular classification dataset. Three input modes:

```bash
conda activate intelliclave

# Mode A — one CSV per client (recommended)
python data/datascripts/pipeline.py --mode per-client \
    --client-csvs hospital_a.csv hospital_b.csv hospital_c.csv \
    --label-col diagnosis

# Mode B — one combined CSV, split by a column
python data/datascripts/pipeline.py --mode split \
    --csv combined_data.csv --label-col outcome --split-col site_id

# Mode B — random split into N clients
python data/datascripts/pipeline.py --mode split \
    --csv combined_data.csv --label-col target --n-clients 4

# Recompute class weights after pipeline
python data/datascripts/weights.py
```

Then run training as normal — schema, class count, and feature dimensions are all inferred automatically.

---

## Experiments

```bash
conda activate intelliclave

# Privacy-utility tradeoff (ε sweep)
python privacy/epsilon_sweep.py --epsilons 1 2 5 10 20

# Cross-validation
python evaluation/cross_validation.py --folds 5 --epochs 10

# Security attacks
python security/attacks/model_inversion.py
python security/attacks/membership_inference.py
python security/attacks/gradient_poisoning.py --fl-rounds 5

# Generate final results figure
python evaluation/generate_graph6.py

# Evaluate saved global model
python fl/evaluate_global_model.py --checkpoint results/fl_rounds/global_model_latest.pth
```

---

## Dashboard API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/status` | Live training status (round, ε, accuracy) |
| GET | `/results` | Per-round FL metrics + per-class F1 |
| GET | `/attestation` | TEE attestation record (MRENCLAVE, status) |
| GET | `/benchmarks` | TEE overhead measurements |
| GET | `/attacks` | Security attack simulation summaries |
| GET | `/privacy_log` | Per-client per-round epsilon audit log |
| GET | `/model_info` | Loaded model metadata (input_dim, classes, type) |
| GET | `/query_stats` | Rate limit status for calling IP |
| POST | `/predict` | Inference — returns label + optional confidence |

---

## Security Mitigations

| Threat (STRIDE) | Mitigation | Result |
|-----------------|-----------|--------|
| Data leakage | Federated Learning — raw data never leaves client | ✓ |
| Gradient leakage | DP-SGD (ε=10, δ=1/n) | ε ≤ 10 across all rounds |
| Weight tampering | AES-256-GCM + RSA-2048 + HMAC-SHA256 | Tamper detected ✓ |
| Rogue server | SGX attestation — MRENCLAVE verified before connecting | Blocked ✓ |
| Model inversion | Confidence masking + rate limiting (100 req/60s) | Partial mitigation |
| Membership inference | DP-SGD reduces confidence gap | AUC = 0.503 (resistant) |
| Gradient poisoning | FedAvg dilution (1 of N clients) | −3.78% at 100% flip |

---

## Known Limitations

1. **gramine-direct** — WSL2 has no SGX hardware. Production uses `gramine-sgx` with zero code changes.
2. **FedAvg Byzantine robustness** — 3.78% accuracy drop at 100% poison rate. Production fix: Krum or Trimmed Mean aggregation.
3. **Model inversion** — cosine similarity 0.90 in PCA space even with confidence masking. PCA vectors cannot be mapped back to raw sensor data without `pca_model.pkl`.
4. **DP accuracy cost** — ε=10 costs 5.72% vs no-DP baseline. ε=1 → 50.89% accuracy (unusable).
5. **In-memory rate limiter** — effective limit is `N × 100` in multi-worker deployments. Replace with Redis-backed rate limiting for production.

---

## Interface Contracts

All shared data formats, CSV schemas, and API contracts are documented in `contracts.md`.

---

## Branch Rules

```
main     → stable, working code only — never commit directly
dev      → integration branch
feature/ → individual feature branches
```

Always open a PR to merge into `dev`. Never push directly to `main`.
