# IntelliClave
## Confidential Computing Secure Processing System
> Privacy-preserving federated learning pipeline using TEE · Differential Privacy · AES-256-GCM

---

## What Is This

IntelliClave is a privacy-preserving AI pipeline, in this case it is for Human Activity Recognition (HAR).
Three organisations (FitLife, MediTrack, CareWatch) collaboratively train a shared model
on UCI HAR sensor data without any raw data ever leaving their premises.

Three security layers stack on top of each other:

- **Federated Learning** — raw data never leaves each client; only encrypted gradients are shared
- **Differential Privacy** — Opacus DP-SGD ensures gradients cannot reveal training data (ε=10)
- **Trusted Execution Environment** — model aggregation runs inside a hardware-sealed SGX enclave

---

## Key Results

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
| Member 1 | Data pipeline + Federated Learning + Deployment |
| Member 2 | Differential Privacy + Evaluation + Dashboard |
| Member 3 | TEE + SGX + Attestation + Kubernetes + Security |

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
| Dashboard Frontend | React + Recharts | 18.x |

---

## Project Structure

```
intelliclave/
├── fl/                     # Federated Learning (Flower server + client + DP)
│   ├── model.py            # HARClassifier (50→128→64→6)
│   ├── data_utils.py       # CSV loading, StandardScaler, train/test split
│   ├── fl_client.py        # Flower client + Opacus DP + AES-256-GCM crypto
│   ├── fl_server.py        # Flower server + FedAvg + decryption + sealed storage
│   ├── run_server.py       # Server launcher (--crypto, --attest flags)
│   └── run_client.py       # Client launcher (--dp, --crypto, --attest flags)
│
├── privacy/                # Differential Privacy (Opacus)
│   ├── dp_trainer.py       # DPTrainer wrapper around Opacus PrivacyEngine
│   ├── dp_flower_client.py # Standalone DP Flower client
│   ├── epsilon_sweep.py    # Privacy-utility tradeoff experiments
│   └── budget_monitor.py   # Per-client epsilon tracking
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
├── security/attacks/       # Security analysis
│   ├── model_inversion.py  # Gradient-based class reconstruction
│   ├── membership_inference.py  # Threshold attack (AUC ≈ 0.5 — resistant)
│   └── gradient_poisoning.py    # Label-flip Byzantine attack sweep
│
├── evaluation/
│   ├── cross_validation.py # 5-fold stratified CV (96.98% ± 0.34%)
│   ├── metrics.py          # F1, accuracy, AUC-ROC helpers
│   └── generate_graph6.py  # 4-panel final results figure
│
├── dashboard/
│   ├── backend/main.py     # FastAPI backend (13 endpoints, 13/13 E2E tests pass)
│   └── frontend/           # React + Recharts live dashboard
│
├── kubernetes/             # K8s deployment (minikube)
│   ├── cold_start.sh       # Full cluster cold start script
│   ├── deployments/        # Server + 3 clients + SGX variants + dashboard
│   ├── policies/           # NetworkPolicy (ports 8080 + 8001)
│   ├── secrets/            # fl-crypto-keys Secret template
│   └── volumes/            # PVCs (server 1Gi, clients 100Mi each)
│
├── docker/
│   ├── Dockerfile.server   # FL server image (torch 2.1.0 + flwr + cryptography)
│   ├── Dockerfile.client   # FL client image (+ opacus)
│   └── docker-compose.yml  # Full stack: server + 3 clients (20 rounds, DP + crypto)
│
├── data/processed/         # Frozen client CSVs (3 × non-IID, PCA-50)
├── results/                # All experiment outputs
│   ├── fl_rounds/          # Model checkpoints + metrics per round
│   ├── attacks/            # Attack simulation results (3 attacks × DP/no-DP)
│   ├── benchmarks/         # TEE overhead measurements
│   └── graphs/             # Final results figure (graph6)
│
├── attestation.json        # Live attestation record (refreshed on server start)
├── status.json             # Live training status (read by dashboard)
├── contracts.md            # Data contracts and interface specifications
└── requirements.txt        # Pinned Python dependencies
```

---

## Quick Start

### Local (no Docker)

```bash
# 1. Create and activate conda environment
conda create -n intelliclave python=3.10
conda activate intelliclave

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start dashboard backend (terminal 1)
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 4. Start dashboard frontend (terminal 2)
cd dashboard/frontend/intelliclave-ui
npm install && npm start
# Opens at http://localhost:3000
```

### Run FL baseline (no DP)

```bash
# Terminal 1 — server
python fl/run_server.py --rounds 10 --min-clients 3

# Terminals 2, 3, 4 — clients
python fl/run_client.py --id 1
python fl/run_client.py --id 2
python fl/run_client.py --id 3
```

### Run FL + DP (ε=10)

```bash
# Terminal 1 — server
python fl/run_server.py --rounds 5 --min-clients 3

# Terminals 2, 3, 4 — clients with DP
python fl/run_client.py --id 1 --dp --epsilon 10.0 --rounds 5
python fl/run_client.py --id 2 --dp --epsilon 10.0 --rounds 5
python fl/run_client.py --id 3 --dp --epsilon 10.0 --rounds 5
```

### Run FL + DP + Crypto + Attestation (full stack)

```bash
# Terminal 1 — server (generates keypair + attestation record)
python fl/run_server.py --rounds 5 --min-clients 3 --crypto --attest

# Terminals 2, 3, 4 — clients (verify attestation before connecting)
python fl/run_client.py --id 1 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 2 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 3 --dp --epsilon 10.0 --rounds 5 --crypto --attest
```

### Docker (20 rounds, DP + crypto, default)

```bash
docker compose -f docker/docker-compose.yml up --build
```

### Kubernetes (minikube)

```bash
# Full cold start (generates keypair, applies all resources, waits for readiness)
bash kubernetes/cold_start.sh

# SGX production mode
SGX=true bash kubernetes/cold_start.sh
```

---

## Verification Checklist

```bash
# Crypto layer (4/4 tests)
python crypto/certs/test_crypto.py

# SGX attestation
python tee/attestation/attestation_simulator.py
# → [ATTESTATION] ✓ ATTESTATION VERIFIED

# Attestation integration (server + 3 clients + rogue server blocked)
python tee/attestation/attestation_integration.py
# → ATTESTATION INTEGRATION DEMO PASSED ✓

# Sealed storage
python tee/sealed_storage/sealed_storage.py
# → ALL SEALED STORAGE TESTS PASSED ✓

# Dashboard E2E (13/13 tests)
python dashboard/backend/test_e2e.py
# → ALL E2E TESTS PASSED ✓ (13/13)

# K8s YAML validation
bash kubernetes/validate.sh
# → ALL YAML FILES VALID ✓
```

---

## Dashboard API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/status` | Live training status (round, ε, accuracy) |
| GET | `/results` | Per-round FL metrics + per-class F1 |
| GET | `/attestation` | TEE attestation record (MRENCLAVE, status) |
| GET | `/benchmarks` | TEE overhead measurements |
| GET | `/attacks` | Security attack simulation summaries |
| GET | `/privacy_log` | Per-client per-round epsilon audit log |
| GET | `/query_stats` | Rate limit status for calling IP |
| POST | `/predict` | HAR inference (confidence masking + rate limiting) |

---

## Security Mitigations

| Threat | Mitigation | Result |
|--------|-----------|--------|
| Data leakage | Federated Learning — raw data never leaves client | ✓ |
| Gradient leakage | DP-SGD (ε=10, δ=1/n) | ε ≤ 10 across all rounds |
| Weight tampering | AES-256-GCM + RSA-2048 + HMAC-SHA256 | Tamper detected ✓ |
| Rogue server | SGX attestation — MRENCLAVE verified before connecting | Blocked ✓ |
| Model inversion | Confidence masking + rate limiting (100 req/60s) | Partial mitigation |
| Membership inference | DP-SGD reduces confidence gap | AUC = 0.503 (resistant) |
| Gradient poisoning | FedAvg dilution (1 of 3 clients) | −3.78% at 100% flip |

---

## Known Limitations

1. **gramine-direct** — WSL2 has no SGX hardware. Production uses `gramine-sgx` with zero code changes.
2. **FedAvg Byzantine robustness** — 3.78% accuracy drop at 100% poison rate. Production fix: Krum or Trimmed Mean.
3. **Model inversion** — cosine similarity 0.90 in PCA space even with confidence masking. PCA vectors cannot be mapped to raw sensor data without `pca_model.pkl`.
4. **DP accuracy cost** — ε=10 costs 5.72% vs no-DP baseline. ε=1 → 50.89% accuracy.
5. **Dashboard** — reflects the 5-round FL+DP run. A live training run updates these in real time.

---

## Interface Contracts

All shared data formats, CSV schemas, and API contracts are in `contracts.md`.

---

## Branch Rules

```
main     → stable, working code only — never commit directly
dev      → integration branch
feature/ → individual feature branches
```

Always create a PR to merge into `dev`. Never push directly to `main`.
