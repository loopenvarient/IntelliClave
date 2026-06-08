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

Current pre-populated demo run: **35 FL rounds**, 3 non-IID clients (Dirichlet α=0.5), FedProx + DP-SGD.
Reproduce with the [End-to-End Reference Run](#end-to-end-reference-run). Full metrics and slide tables: [`report/final_results_summary.md`](report/final_results_summary.md).

| Metric | Value |
|--------|-------|
| Global FL + DP (35 rounds) | **85.0%** accuracy, **78%** macro F1 |
| Privacy budget consumed | ε ≈ **7.99** |
| Membership inference AUC | **0.505** (near-random — **resistant**) |
| Gradient poisoning — FedAvg baseline (100% label-flip) | 91.85% → 15.04% (**vulnerable**, threat demo) |
| Gradient poisoning — trimmed mean (`--robust`, 100% flip) | 81.95% → 80.93% (~**1%** drop — **resistant**) |
| Model inversion — defended (noise + temperature) | avg cosine sim **0.017** — **resistant** |
| Privacy–utility: solo local vs global FL (DP-matched) | 52.8% → 85.7% weighted F1 (**+32.9pp**) |
| Crypto layer self-test | **4/4** passed |
| TEE attestation | legitimate server verified, rogue server blocked |
| TEE overhead | 35.2% avg, 0.14% of FL round time |

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
│   ├── compare_local_vs_global.py  # Solo local vs global FL (DP-matched) → local_vs_global.json
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
│   └── gradient_poisoning.py    # Label-flip Byzantine sweep (FedAvg + --robust trimmed mean)
│
├── evaluation/
│   ├── cross_validation.py # 5-fold stratified CV — any dataset
│   ├── metrics.py          # F1, accuracy, AUC-ROC helpers
│   └── generate_graph6.py  # 4-panel final results figure (data-driven, no hardcoding)
│
├── dashboard/
│   ├── backend/
│   │   ├── main.py         # FastAPI — status, results, attacks, comparison, predict
│   │   ├── predict_routes.py  # Batch CSV + per-client local model inference
│   │   └── test_e2e.py     # Endpoint tests (no live server required)
│   └── frontend/intelliclave-ui/  # React + Recharts + Vite (7 pages, live polling)
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
│   ├── attacks/            # Attack JSONs (baseline + defended/robust variants)
│   ├── local_baselines/    # Per-client solo models (for comparison script)
│   ├── local_vs_global.json  # Privacy–utility: solo vs federated (DP-matched)
│   ├── benchmarks/         # TEE overhead measurements
│   └── graphs/             # graph6_final_results.png
│
├── report/                 # Final report, STRIDE docs, results summary
├── scripts/                # Pipeline helpers (e.g. start_dashboard.ps1)
│
├── attestation.json        # Live attestation record
├── status.json             # Live training status (read by dashboard)
├── contracts.md            # Data contracts and interface specifications
└── requirements.txt        # Pinned Python dependencies
```

Each major folder has its own **`README.md`** with focused commands and data flow (see `fl/`, `security/`, `dashboard/`, etc.).

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

# Crypto self-test (4/4)
python crypto/certs/test_crypto.py

# TLS certificate bundle
python crypto/certs/generate_tls_certs.py

# Attestation demo
python tee/attestation/attestation_integration.py

# Sealed storage
python tee/sealed_storage/sealed_storage.py

# Dashboard E2E — no server needed, uses TestClient
cd dashboard/backend && python test_e2e.py
```

### 3 — Start the dashboard

The dashboard loads **pre-populated results** from `results/` and `status.json` immediately — no training run required. When the backend is connected, panels show a **LIVE DATA** badge; offline mode falls back to demo placeholders.

**Pages:** Overview · Training · Privacy · Clients · Evaluation · TEE · Predictions

```bash
# Terminal A — backend
conda activate intelliclave
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Terminal B — frontend
cd dashboard/frontend/intelliclave-ui
npm run dev
# Opens at http://localhost:5173

# Windows — both terminals in one command
powershell -ExecutionPolicy Bypass -File scripts/start_dashboard.ps1
```

The **Evaluation** page shows per-class F1, security attack verdicts (prefers defended/robust JSON variants), and the **Privacy–Utility: Solo vs Federated** panel (`GET /comparison`).

---

## End-to-End Reference Run

Command sequence used for the UCI HAR demo (`status.json`, `results/`, dashboard). Run from the repo root with `conda activate intelliclave`.

### 1 — Prepare the dataset

Build three non-IID client CSVs from UCI HAR text files (Dirichlet partition, full 561 features, no PCA).

```bash
python data/datascripts/pipeline.py --mode textfiles --partition dirichlet --dirichlet-alpha 0.5 --n-clients 3 --no-pca
```

### 2 — Secure federated learning (DP + crypto + attestation)

Start the server first, then one client per terminal. FedProx aggregation, PrivacyWrapper inference settings (`--noise-scale`, `--temperature`), encrypted weight updates, and SGX attestation handshake.

```bash
# Terminal 1 — FL server
python fl/run_server.py --rounds 35 --min-clients 3 --crypto --attest --noise-scale 0.5 --temperature 1.0 --strategy fedprox

# Terminal 2 — Client 1
python fl/run_client.py --id 1 --dp --epsilon 8.0 --rounds 35 --local-epochs 1 --batch-size 64 --crypto --attest

# Terminal 3 — Client 2
python fl/run_client.py --id 2 --dp --epsilon 8.0 --rounds 35 --local-epochs 1 --batch-size 64 --crypto --attest

# Terminal 4 — Client 3
python fl/run_client.py --id 3 --dp --epsilon 8.0 --rounds 35 --local-epochs 1 --batch-size 64 --crypto --attest
```

Writes checkpoints and metrics under `results/fl_rounds/` and updates `status.json`.

### 3 — Security attack evaluation

Run offline attacks against the trained global model. Results land in `results/attacks/` and feed the dashboard Evaluation panel.

```bash
python security/attacks/model_inversion.py
python security/attacks/membership_inference.py
python security/attacks/gradient_poisoning.py
```

Optional — trimmed-mean defence rerun (dashboard prefers `gradient_poisoning_robust.json` when present):

```bash
python security/attacks/gradient_poisoning.py --robust
```

### 4 — Final results figure

Generate the four-panel summary chart from live result JSONs.

```bash
python evaluation/generate_graph6.py
```

### 5 — Crypto, TLS, and attestation checks

Standalone verification of the secure stack (independent of FL training).

```bash
# AES-256-GCM encrypt/decrypt, integrity, tamper rejection (4/4 self-test)
python crypto/certs/test_crypto.py

# Generate CA + server + client TLS certificate bundle
python crypto/certs/generate_tls_certs.py

# SGX attestation demo — legitimate server verified, rogue server blocked
python tee/attestation/attestation_integration.py
```

### 6 — Dashboard

```bash
# Terminal A — backend API (port 8001)
conda activate intelliclave
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Terminal B — frontend UI (port 5173)
cd dashboard/frontend/intelliclave-ui
npm run dev
```

### 7 — Dashboard API tests

Smoke-test all endpoints without starting uvicorn (uses FastAPI `TestClient`).

```bash
cd dashboard/backend
python test_e2e.py
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

Shorter 5-round example. For the full demo parameters (35 rounds, ε=8, FedProx, noise/temperature), see **[End-to-End Reference Run](#end-to-end-reference-run)** above.

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

> **Crypto key setup**: On first run, RSA keys are automatically generated and saved to `crypto/certs/keys/`. See [`docker/CRYPTO_SETUP.md`](docker/CRYPTO_SETUP.md) for production deployment with Docker secrets or Kubernetes.

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

Additional experiments beyond the [reference run](#end-to-end-reference-run):

```bash
conda activate intelliclave

# Privacy-utility tradeoff (ε sweep)
python privacy/epsilon_sweep.py --epsilons 1 2 5 10 20

# Cross-validation
python evaluation/cross_validation.py --folds 5 --epochs 10

# Privacy–utility: solo local vs global FL (DP-matched, feeds dashboard /comparison)
python fl/compare_local_vs_global.py --dp --epsilon 10 --retrain

# Evaluate saved global model
python fl/evaluate_global_model.py --checkpoint results/fl_rounds/global_model_latest.pth
```

---

## Dashboard API

Auth: `POST /token` (default `admin` / `adminpass`). See [`dashboard/API_AUTH.md`](dashboard/API_AUTH.md).

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/token` | JWT login |
| GET | `/status` | Live training status + client sample counts |
| GET | `/results` | Per-round FL metrics + per-class F1 + privacy merge |
| GET | `/comparison` | Solo local vs global FL (`results/local_vs_global.json`) |
| GET | `/attacks` | Attack summaries — prefers `_defended`, `_mitigated`, `_robust` variants |
| GET | `/attestation` | TEE attestation record (MRENCLAVE, status) |
| GET | `/benchmarks` | TEE overhead measurements |
| GET | `/privacy_log` | Per-client per-round epsilon audit log |
| GET | `/model_info` | Loaded model metadata (input_dim, classes, type) |
| GET | `/local_models` | Available per-client local checkpoints |
| GET | `/query_stats` | Rate limit status for calling IP |
| POST | `/predict` | Single-row inference (PrivacyWrapper + rate limit) |
| POST | `/predict_csv` | Batch predict on global model |
| POST | `/predict_csv_client` | Batch predict on a client's local model |

---

## Security Mitigations

| Threat (STRIDE) | Mitigation | Result |
|-----------------|-----------|--------|
| Data leakage | Federated Learning — raw data never leaves client | ✓ |
| Gradient leakage | DP-SGD (ε=10, δ=1/n) | ε ≈ 8 consumed in reference run |
| Weight tampering | AES-256-GCM + RSA-2048 + HMAC-SHA256 | Tamper detected ✓ |
| Rogue server | SGX attestation — MRENCLAVE verified before connecting | Blocked ✓ |
| Model inversion | PrivacyWrapper (noise + temperature) + rate limiting | **Resistant** (cosine sim 0.017) |
| Membership inference | DP-SGD reduces confidence gap | **Resistant** (AUC ≈ 0.505) |
| Gradient poisoning | Trimmed-mean aggregation (`--robust` simulation) | **Resistant** (~1% drop at 100% flip) |

> **Live FL server** (`fl_server.py`) still uses FedAvg/FedProx weighted averaging. Byzantine defences (trimmed mean, Krum) are validated in `security/attacks/gradient_poisoning.py` and reported on the dashboard; wiring them into live aggregation is a follow-up for production parity.

---

## Known Limitations

1. **gramine-direct** — WSL2 has no SGX hardware. Production uses `gramine-sgx` with zero code changes.
2. **Byzantine defence gap** — trimmed mean is proven in attack simulation but not yet in live `fl_server.py` aggregation (still FedAvg/FedProx).
3. **Model inversion (unmitigated)** — baseline attack shows high cosine similarity; defended config (noise + temperature) reduces risk to near-zero on the dashboard.
4. **DP accuracy cost** — privacy budget consumes utility; solo local DP baselines average ~53% F1 vs ~86% for federated global under matched ε.
5. **In-memory rate limiter** — effective limit is `N × 100` in multi-worker deployments. Replace with Redis-backed rate limiting for production.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`report/final_results_summary.md`](report/final_results_summary.md) | Command log, metrics tables, slide outline |
| [`report/security_report.md`](report/security_report.md) | Full security analysis |
| [`contracts.md`](contracts.md) | CSV schemas, API contracts, artifact formats |
| `<folder>/README.md` | Per-module commands and data flow |

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
