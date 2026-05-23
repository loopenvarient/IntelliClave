# IntelliClave
## Confidential Computing Secure Processing System
> Privacy-preserving federated learning pipeline — Differential Privacy · AES-256-GCM · TEE (simulation)

---

## What Is This

IntelliClave is a dataset-agnostic, privacy-preserving federated learning pipeline. The reference implementation uses UCI HAR sensor data split across three non-IID clients (FitLife, MediTrack, CareWatch), but the entire pipeline — FL training, DP, evaluation, security attacks, and dashboard — works with any tabular classification dataset by dropping CSVs into `data/processed/`.

Three security layers stack on top of each other:

- **Federated Learning** — raw data never leaves each client; only model updates are shared
- **Differential Privacy** — Opacus DP-SGD with Rényi accountant; cumulative ε tracked per client; privacy-utility tradeoff measured and validated across ε=1–15
- **Trusted Execution Environment** — mutual SGX attestation and MRENCLAVE-bound sealed storage; currently running in `gramine-direct` simulation mode (no hardware SGX required to run the demo — see Known Limitations)

---

## Key Results (UCI HAR reference run)

### Federated Learning

| Metric | Value |
|--------|-------|
| FL baseline accuracy (10 rounds, no DP) | 96.99% |
| FL + DP accuracy (ε=10, 5 rounds, federated) | 91.27% |
| Cross-validation accuracy — centralized baseline | 96.98% ± 0.34% |
| AUC-ROC — centralized baseline | 99.85% |

### Differential Privacy — privacy-utility tradeoff

Measured with optimised DP settings (batch=64, dropout=0.0, lr=2×10⁻³). Both curves are monotone — lower ε gives lower accuracy, confirming the privacy guarantee is real.

| ε | Same-client accuracy | Cross-client accuracy | Cost vs no-DP |
|---|---|---|---|
| 1 | 76.5% | 68.7% | −18pp |
| 3 | 82.6% | ~74% | −12pp |
| 5 | 86.2% | ~78% | −8pp |
| **10** | **89.4%** | **82.9%** | **−5pp** ← operating point |
| 15 | 88.2% | ~82% | −6pp |
| no-DP | 94.2% | — | baseline |

The 14.3pp cross-client gap between ε=1 and ε=10 is the clearest signal: the ε=1 model genuinely knows less about the data. ε=10 is the production operating point.

### Security

| Metric | Value |
|--------|-------|
| Membership inference AUC | 0.503 — threshold attack near-random (formal MI bound from DP at ε=10) |
| Gradient poisoning drop — FedAvg (100% label-flip, 1/3 clients) | −3.78% |
| Gradient poisoning drop — Krum (same attack) | Reduced — formal Byzantine guarantee (requires ≥ 5 clients) |

### TEE

| Metric | Value |
|--------|-------|
| TEE overhead | 35.2% avg, 0.14% of FL round time |
| TEE mode | **gramine-direct (simulation)** — no hardware SGX; see Known Limitations |

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
| Differential Privacy | Opacus (Rényi DP accountant) | 1.4.0 |
| TEE | Gramine (gramine-direct, simulation) | 1.9 |
| Cryptography | AES-256-GCM + RSA-2048 + TLS 1.3 | — |
| Infrastructure | Docker + Kubernetes (minikube) | — |
| Dashboard Backend | FastAPI | 0.104.1 |
| Dashboard Frontend | React + Recharts + Vite | 18.x |

---

## Project Structure

```
intelliclave/
├── config/
│   ├── constants.py              # Shared defaults (ε, n_clients, lr, DP batch size, max_grad_norm, etc.)
│   └── runtime_paths.py          # Dashboard state paths (status.json, attestation.json under results/)
│
├── fl/                           # Federated Learning core
│   ├── model.py                  # FLClassifier (MLP) + ResNetTabular + TransformerTabular
│   ├── data_utils.py             # Generic CSV loader — any dataset, auto schema inference
│   ├── fl_client.py              # Flower client + Opacus DP + AES-256-GCM crypto
│   ├── fl_server.py              # Flower server + FedAvg/FedProx/Krum/TrimmedMean
│   ├── robust_aggregation.py     # Byzantine-robust aggregators: Krum, Trimmed Mean, Median
│   ├── run_server.py             # Server launcher (--crypto, --attest, --robust-agg flags)
│   ├── run_client.py             # Client launcher (--dp, --epsilon, --crypto, --attest)
│   ├── run_fl_simulation.py      # Single-process simulation (--robust-agg, --dp supported)
│   ├── train_local.py            # Standalone local training (baseline + DP mode)
│   ├── FLOWER_UPGRADE.md         # Upgrade guide — Flower 1.6 API glue points documented
│   └── evaluate_global_model.py  # Evaluate saved checkpoint on all client CSVs
│
├── privacy/                      # Differential Privacy
│   ├── dp_trainer.py             # DPTrainer — Opacus wrapper + noise_multiplier diagnostic
│   ├── dp_preflight.py           # Pre-flight check at all ε values — three-tier warning system
│   ├── epsilon_sweep.py          # Exp 1: accuracy vs ε · Exp 2: ε over rounds · Exp 3: joint grid
│   ├── clipping_norm_sweep.py    # Clipping norm sensitivity analysis at fixed ε
│   ├── budget_monitor.py         # Cumulative ε tracking per client (Rényi accountant)
│   └── run_budget_monitor.py     # Budget monitor CLI (--max-epsilon, --privacy-json)
│
├── tee/                          # Trusted Execution Environment (simulation mode)
│   ├── attestation/              # Mutual SGX attestation — server + client quotes
│   ├── tee_mode.py               # Detects gramine-direct vs gramine-sgx at runtime
│   ├── sealed_storage/           # MRENCLAVE-bound AES-256-GCM sealed storage
│   │   └── seal_pca.py           # Seal pca_model.pkl inside TEE (closes white-box path)
│   ├── fl_enclave/               # Gramine manifests (server + 3 clients)
│   ├── full_stack_test/          # PyTorch + Flower + Opacus inside Gramine
│   └── benchmarks/               # TEE overhead measurement (35.2% avg)
│
├── crypto/certs/                 # Cryptographic layer
│   ├── crypto_layer.py           # AES-256-GCM + RSA-2048 weight encryption
│   ├── crypto_context.py         # Server keypair lifecycle
│   ├── generate_tls_certs.py     # TLS certificate generator
│   └── test_crypto.py            # 4 crypto tests (all pass)
│
├── security/
│   ├── inference_protection.py   # Output perturbation (Laplace) + lifetime query budget
│   └── attacks/                  # Security analysis (all fully argparse-configurable)
│       ├── model_inversion.py    # White-box baseline + black-box surrogate attack
│       ├── membership_inference.py  # Threshold attack (AUC ≈ 0.5; formal MI from DP ε)
│       └── gradient_poisoning.py    # Label-flip sweep across FedAvg / Krum / TrimmedMean
│
├── evaluation/
│   ├── cross_validation.py       # Centralized baseline + federated leave-one-client-out CV
│   ├── metrics.py                # F1, accuracy, AUC-ROC helpers
│   └── generate_graph6.py        # 4-panel final results figure (data-driven)
│
├── dashboard/
│   ├── backend/                  # FastAPI — 11 endpoints, inference protection, sealed checkpoints
│   │   ├── main.py
│   │   ├── training_freshness.py
│   │   └── Dockerfile
│   └── frontend/                 # React + Recharts + Vite (+ Dockerfile/nginx)
│
├── data/
│   ├── processed/                # Client CSVs (client1.csv, client2.csv, ...)
│   ├── datascripts/              # pipeline.py (3 input modes), weights.py, har_analysis.py
│   └── class_weights.json        # Optional per-class loss weights
│
├── kubernetes/                   # K8s deployment (minikube)
│   ├── cold_start.sh             # FL server + clients + Redis + dashboard API
│   ├── validate.sh               # kubectl dry-run on all manifests
│   ├── deployments/              # Server + clients + SGX variants + Redis + dashboard
│   ├── policies/                 # NetworkPolicy (ports 8080 + 8001)
│   ├── secrets/                  # fl-crypto-keys Secret template
│   └── volumes/                  # PVCs (server 1Gi, clients 100Mi each)
│
├── docker/
│   ├── Dockerfile.server         # FL server image (+ tee sealing)
│   ├── Dockerfile.client         # FL client image (+ opacus)
│   ├── docker-compose.yml        # Full stack: 3 clients + Redis + dashboard API + UI
│   └── generate_compose.py       # Generate compose for N clients (--clients N)
│
├── scripts/
│   ├── build_docker_images.sh    # Build all container images (+ K8s tags)
│   ├── harden_production.py      # Seal checkpoints, exposure scan, inversion assert
│   ├── check_checkpoint_exposure.py  # Detect plaintext .pth on disk (P1)
│   └── check_sgx.py              # SGX capability probe
│
├── results/                      # All experiment outputs (pre-populated for demo)
│   ├── fl_rounds/                # Model checkpoints (sealed) + metrics + convergence
│   ├── attacks/                  # Attack results (3 attacks × aggregators)
│   ├── benchmarks/               # TEE overhead measurements
│   └── graphs/                   # graph6_final_results.png
│
├── attestation.json              # Live attestation (mirrored from results/attestation.json)
├── status.json                   # Live training status (mirrored from results/status.json)
├── _run_pipeline.py              # Full integration test suite (17 stages)
├── contracts.md                  # Data contracts and API interface specifications
└── requirements.txt              # Pinned Python dependencies
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

# Mutual attestation demo (server + 3 clients + rogue server/client simulation)
# Note: runs in gramine-direct simulation mode — no SGX hardware required
python tee/attestation/attestation_integration.py

# Sealed storage
python tee/sealed_storage/sealed_storage.py

# Dashboard E2E (13/13) — no server needed, uses TestClient
python dashboard/backend/test_e2e.py

# Full pipeline (17 stages — crypto, TEE, training, attacks, dashboard, eval)
python _run_pipeline.py
```

### 3 — Start the dashboard

The dashboard loads pre-recorded results immediately — no training run required.
The top bar shows **LIVE**, **STALE (Nh ago)**, **TRAINING**, or **PRE-RECORDED** based on `last_training_completed_at` from `/status`.

```bash
# Terminal A — backend
conda activate intelliclave
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Terminal B — frontend
cd dashboard/frontend/intelliclave-ui
npm run dev
# Opens at http://localhost:3000 (see vite.config.js)

# Optional: point API at a remote backend
# VITE_API_URL=http://localhost:8001 npm run dev
```

> **Shared runtime files:** After training, the FL server writes `results/status.json`, `results/attestation.json`, and `results/results.json`. The dashboard reads these (with fallback to repo-root copies for backward compatibility). In Docker/K8s, set `DASHBOARD_STATE_DIR=/app/results` so all services share one volume.

---

## Running FL Training

### Simulation (single process, easiest)

```bash
conda activate intelliclave

# Baseline (no DP)
python fl/run_fl_simulation.py --rounds 10 --clients 3

# With DP (preflight + max_grad_norm=2.0 default for stronger gradient noise)
python fl/run_fl_simulation.py --rounds 5 --clients 3 --dp --epsilon 10.0 --max-grad-norm 2.0

# FedProx + ResNet model
python fl/run_fl_simulation.py --strategy fedprox --model-type resnet-tabular

# Byzantine-robust aggregation (Krum — requires >= 5 clients; blocked otherwise)
python data/datascripts/pipeline.py --n-clients 5   # once, to create client4–5.csv
python fl/run_fl_simulation.py --rounds 10 --clients 5 --robust-agg krum

# Byzantine-robust aggregation (Trimmed Mean — works at any n >= 2)
python fl/run_fl_simulation.py --rounds 10 --robust-agg trimmed-mean --byzantine-fraction 0.33
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

# Clients — --rounds must match server; default max_grad_norm=2.0
# DP preflight runs before training and surfaces noise_multiplier × C
python fl/run_client.py --id 1 --dp --epsilon 10.0 --rounds 5 --max-grad-norm 2.0
python fl/run_client.py --id 2 --dp --epsilon 10.0 --rounds 5 --max-grad-norm 2.0
python fl/run_client.py --id 3 --dp --epsilon 10.0 --rounds 5 --max-grad-norm 2.0
```

### Full stack (DP + Crypto + Mutual Attestation + Krum)

> **Krum requires ≥ 5 clients** (`n ≥ 2f+3` for f=1 Byzantine tolerance). The server will exit with a clear error if you pass `--robust-agg krum --min-clients 3`. For the 3-client demo use `--robust-agg trimmed-mean` instead.

```bash
# Generate client4 and client5 data (one-time setup)
python data/datascripts/pipeline.py --n-clients 5

# Server — 5-client Krum with crypto and attestation
python fl/run_server.py --rounds 5 --min-clients 5 --crypto --attest --robust-agg krum

# Clients
python fl/run_client.py --id 1 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 2 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 3 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 4 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 5 --dp --epsilon 10.0 --rounds 5 --crypto --attest
```

```bash
# 3-client demo — trimmed-mean is robust at any n >= 2
python fl/run_server.py --rounds 5 --min-clients 3 --crypto --attest --robust-agg trimmed-mean

python fl/run_client.py --id 1 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 2 --dp --epsilon 10.0 --rounds 5 --crypto --attest
python fl/run_client.py --id 3 --dp --epsilon 10.0 --rounds 5 --crypto --attest
```

### Docker (full stack)

Runs **3 FL clients**, **FL server**, **Redis**, **dashboard API**, and **dashboard UI** on a shared `server-results` volume (`/app/results`).

| Service | URL |
|---------|-----|
| Dashboard UI | http://localhost:3000 |
| Dashboard API | http://localhost:8001 |
| FL server (gRPC) | localhost:8080 |

```bash
# From repo root — build and start everything
docker compose -f docker/docker-compose.yml up --build

# Or build images first (also tags K8s names)
bash scripts/build_docker_images.sh
docker compose -f docker/docker-compose.yml up

# N clients (regenerate compose when N ≠ 3)
python docker/generate_compose.py --clients 5
docker compose -f docker/docker-compose.generated.yml up --build
```

**Environment variables** (optional overrides):

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERENCE_EPSILON` | `4.0` | API output noise (lower = more noise) |
| `LIFETIME_QUERY_BUDGET` | `10000` | Max `/predict` calls per IP |
| `RATE_LIMIT_MAX` | `100` | Requests per window per IP |
| `SEAL_REMOVE_PLAINTEXT` | `true` | Seal checkpoints; delete `.pth` after each round |
| `REQUIRE_SEALED_CHECKPOINTS` | `false` | Dashboard refuses plaintext `.pth` (set `true` in prod) |
| `DASHBOARD_STATE_DIR` | `/app/results` | Shared `status.json` / `attestation.json` path |

After the first FL run completes, verify the dashboard sees live data:

```bash
curl http://localhost:8001/status
curl http://localhost:8001/attestation
curl http://localhost:8001/results
```

### Kubernetes (minikube)

```bash
# Build images and load into minikube
bash scripts/build_docker_images.sh
minikube image load intelliclave-server:latest
minikube image load intelliclave-client:latest
minikube image load intelliclave-dashboard:latest

minikube start --driver=docker --cpus=4 --memory=6144
bash kubernetes/cold_start.sh          # FL + Redis + dashboard API
kubectl port-forward svc/dashboard-service 8001:8001 -n intelliclave

# Validate manifests without a cluster
bash kubernetes/validate.sh

# SGX production mode (requires SGX-capable hardware)
SGX=true bash kubernetes/cold_start.sh
```

> **Note:** K8s does not deploy the React UI container yet — use local `npm run dev` or Docker Compose for the frontend.

### Production hardening

```bash
conda activate intelliclave

# Seal checkpoints and remove plaintext .pth (P1 — white-box containment)
python scripts/harden_production.py --seal-checkpoints --delete-plaintext

# Scan for exposed plaintext weights
python scripts/check_checkpoint_exposure.py --strict

# Verify black-box model inversion defenses (API path)
python scripts/harden_production.py --verify-inversion
# Or: python security/attacks/model_inversion.py --assert

# All of the above
python scripts/harden_production.py --all
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

# Mode C — random split into N clients
python data/datascripts/pipeline.py --mode split \
    --csv combined_data.csv --label-col target --n-clients 4

# Recompute class weights after pipeline
python data/datascripts/weights.py

# Seal the PCA model inside the TEE (closes white-box inversion path)
python tee/sealed_storage/seal_pca.py --seal
```

Schema, class count, and feature dimensions are all inferred automatically.

---

## Experiments

```bash
conda activate intelliclave

# Privacy-utility tradeoff (ε sweep, Exp 1+2)
python privacy/epsilon_sweep.py --epsilons 1 2 5 10 20

# Joint (ε × clipping_norm) grid sweep — find the jointly optimal configuration
python privacy/epsilon_sweep.py --exp 3 --epsilons 1 5 10 --clipping-norms 0.1 0.5 1.0 2.0 5.0

# Clipping norm sensitivity at a fixed ε (diagnose accuracy collapse at low ε)
python privacy/clipping_norm_sweep.py --epsilon 1.0
python privacy/clipping_norm_sweep.py --epsilon 10.0

# Cross-validation — centralized baseline
python evaluation/cross_validation.py --folds 5 --epochs 10

# Cross-validation — federated leave-one-client-out (correct federated metric)
python evaluation/cross_validation.py --mode federated --epochs 10

# Both modes (reported separately — not directly comparable)
python evaluation/cross_validation.py --mode both

# Security attacks
python security/attacks/model_inversion.py                          # white-box + surrogate
python security/attacks/model_inversion.py --assert                 # CI assertion mode
python security/attacks/membership_inference.py
python security/attacks/gradient_poisoning.py --fl-rounds 5         # FedAvg + Krum + TrimmedMean

# Privacy budget audit
python privacy/run_budget_monitor.py --privacy-json results/fl_rounds/fl_privacy.json

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
| GET | `/status` | Training status — includes `training_freshness: live\|stale\|pre-recorded` |
| GET | `/results` | Per-round FL metrics + convergence diagnostics |
| GET | `/attestation` | TEE attestation record — includes `simulation_mode` and `warning` fields |
| GET | `/benchmarks` | TEE overhead measurements |
| GET | `/attacks` | Security attack simulation summaries (all aggregators) |
| GET | `/privacy_log` | Cumulative ε audit log + worst-case summary per client |
| GET | `/model_info` | Loaded model metadata (input_dim, classes, type) |
| GET | `/query_stats` | Rate limit + lifetime budget status for calling IP |
| POST | `/predict` | Inference — output perturbation + lifetime budget enforced |

---

## Security Mitigations

| Threat (STRIDE) | Mitigation | Status |
|-----------------|-----------|--------|
| Data leakage | Federated Learning — raw data never leaves client | Active |
| Gradient leakage | DP-SGD (ε=10, δ=1/n, **max_grad_norm=2.0**) — Rényi accountant | Active |
| Weight tampering | AES-256-GCM + RSA-2048 + HMAC-SHA256 | Active |
| Rogue server | Mutual SGX attestation — server MRENCLAVE verified by clients | **Simulation** — gramine-direct |
| Rogue client | Mutual SGX attestation — client MRENCLAVE verified by server | **Simulation** — same caveat |
| Model inversion (API) | Laplace output perturbation (ε_inf=4.0) + lifetime budget (10k/key) + rate limit | Active — black-box surrogate cos_sim < 0.6 |
| White-box inversion | Sealed checkpoints + no plaintext `.pth` on disk (`SEAL_REMOVE_PLAINTEXT`) | Active — requires prod deployment; API noise does **not** stop weight theft |
| Membership inference | DP-SGD (ε=10) + threshold attack | AUC ≈ 0.5 — near-random |
| Gradient poisoning | Krum / Trimmed Mean / Coordinate Median | Active — Krum requires ≥ 5 clients |

### Model inversion — what is actually protected

| Attack mode | Threat model | Mitigation | CI check |
|-------------|--------------|------------|----------|
| **Black-box (API)** | Attacker queries `/predict` only | Output perturbation + query budget | `model_inversion.py --assert` (surrogate cos_sim < 0.6) |
| **White-box** | Attacker has `global_model_latest.pth` | Checkpoint sealing + access control | Not fixable via API noise — use `scripts/check_checkpoint_exposure.py` |

Increase API noise if needed: `INFERENCE_EPSILON=2.0` or `INFERENCE_NOISE_SCALE=1.0`.

---

## Convergence Diagnostics

After every training run, `results.json` includes:

- `client_metrics` — per-client per-round accuracy, loss, and macro F1
- `convergence` — per-round accuracy delta and oscillation flag
- `convergence_summary` — overall diagnosis, oscillation rate, per-client trend

The server prints a live convergence line each round:

```
[Server][Convergence] Round 4 global_acc=0.9127 Δ=+0.0312 spread=0.0421
[Server][Convergence] Round 5 global_acc=0.9089 Δ=-0.0038  ⚠ OSCILLATING
[Server][Convergence] *** Accuracy oscillating — consider --strategy fedprox or reducing --local-epochs ***
```

---

## Known Limitations

1. **TEE is simulation mode** — All TEE results use `gramine-direct`, which runs entirely in userspace with no hardware enclave. Attestation quotes are software-computed from file hashes, not from CPU hardware. A real adversary can read enclave memory and forge quotes in this mode. The `/attestation` endpoint labels every record with `simulation_mode: true` and a `warning` field. Production requires `gramine-sgx` on SGX-capable hardware — zero code changes needed, just the right runner.

2. **Krum requires ≥ 5 clients** — `--robust-agg krum` with `--min-clients < 5` exits immediately with a clear error (enforced in `run_server.py` before the server starts). With n=3, `auto_f(3)=0` so Krum degenerates to picking a single update with zero Byzantine tolerance. Use `--robust-agg trimmed-mean` or `median` for the 3-client demo — both are robust at any n ≥ 2.

3. **DP accuracy cost is real and validated** — The privacy-utility tradeoff is monotone across ε=1–15. Optimised DP settings (batch=64, dropout=0.0, lr=2×10⁻³, **max_grad_norm=2.0**) are applied when `--dp` is set. Default `max_grad_norm=2.0` increases gradient noise (`noise_std = noise_multiplier × C`) at ε=10; tune with `privacy/clipping_norm_sweep.py --epsilon 10`. Preflight surfaces the noise multiplier before training. For ε < 5, run the clipping sweep first or pass `--auto-clipping-sweep`.

4. **Dashboard budget store resets on restart without Redis** — `docker-compose.yml` sets `REQUIRE_REDIS=true` so the Redis-backed store is used in Docker. Local dev without `REDIS_URL` falls back to in-memory (logged as a warning). Set `REDIS_URL=redis://localhost:6379/0` locally if you need persistent budgets.

5. **FL server recovery** — use `--resume --save-dir <run_dir>` after a crash; checkpoints are written each round to `fl_server_checkpoint.json`. Stragglers: `--round-timeout 300` (default in Docker).

6. **Federated CV variance** — with 3 clients, federated leave-one-client-out CV has only 3 folds and wide confidence intervals. Use `--n-clients 5` in the data pipeline for stabler estimates. `--mode both` reports centralized and federated results separately — they use different fold counts and are not directly comparable.

7. **Dashboard freshness** — `/status` exposes `training_freshness`, `stale_hours`, and `last_training_completed_at`. Set `EXPECTED_TRAINING_CADENCE_HOURS` (default 24) for the stale threshold.

8. **Model checkpoints are sealed by default** — after each round, `global_model_latest.pth` is encrypted to `.pth.sealed` and plaintext is removed when `SEAL_REMOVE_PLAINTEXT=true` (Docker/K8s default). Load via `load_checkpoint_state_dict` (unseals in RAM). Run `python scripts/harden_production.py --seal-checkpoints` for existing lab checkpoints.

9. **Docker vs local dev** — Compose runs the full stack (3 clients + UI). Experiments (`_run_pipeline.py`, epsilon sweeps, attacks) run on the host with conda. The dashboard frontend in Docker uses `VITE_API_URL=http://localhost:8001` (browser → host port mapping).

10. **Flower 1.6 API glue** — custom aggregation logic (`FitRes` construction, `FedProx` MRO, `start_numpy_client`) is tightly coupled to Flower 1.6. See `fl/FLOWER_UPGRADE.md` before upgrading.

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
