# IntelliClave — Handoff Document

_Date: 2026-05-03_

---

## What Was Built

IntelliClave is a privacy-preserving federated learning system for human activity
recognition. Three companies (FitLife, MediTrack, CareWatch) collaboratively train
a shared model without sharing raw sensor data.

**Three security layers:**

| Layer | Technology | What it does |
|-------|-----------|-------------|
| Federated Learning | Flower + FedAvg | Raw data never leaves each client |
| Differential Privacy | Opacus DP-SGD (ε=10) | Gradients cannot reveal training data |
| Trusted Execution Environment | Gramine + Intel SGX | Aggregation is hardware-sealed |

---

## Key Results

| Metric | Value |
|--------|-------|
| FL baseline accuracy (no DP, 10 rounds) | 96.99% |
| FL + DP accuracy (ε=10, 5 rounds) | 91.27% |
| Privacy cost | −5.72% |
| Membership inference AUC | 0.503 (random — resistant) |
| Gradient poisoning drop (100% flip) | −3.78% |
| TEE overhead | 35.2% avg, 0.14% of FL round time |
| Cross-validation accuracy (combined) | 96.98% ± 0.34% |
| AUC-ROC (combined) | 99.85% |

---

## How to Run Everything

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the dashboard

```bash
# Backend (terminal 1)
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Frontend (terminal 2)
cd dashboard/frontend/intelliclave-ui
npm start
# Opens at http://localhost:3000
```

### 3. Run FL baseline (no DP)

```bash
# Terminal 1 — server
python fl/run_server.py --rounds 10 --min-clients 3

# Terminals 2, 3, 4 — clients
python fl/run_client.py --id 1
python fl/run_client.py --id 2
python fl/run_client.py --id 3
```

Expected: accuracy ≈ 96.99%, macro-F1 ≈ 97.04% after 10 rounds.

### 4. Run FL + DP (ε=10)

```bash
# Terminal 1 — server (unchanged)
python fl/run_server.py --rounds 5 --min-clients 3

# Terminals 2, 3, 4 — clients with DP enabled
python fl/fl_client.py --csv data/processed/client1.csv --id 1 --dp --epsilon 10.0 --rounds 5
python fl/fl_client.py --csv data/processed/client2.csv --id 2 --dp --epsilon 10.0 --rounds 5
python fl/fl_client.py --csv data/processed/client3.csv --id 3 --dp --epsilon 10.0 --rounds 5
```

Expected: accuracy ≈ 91.27%, ε stays ≤ 10.0 across all rounds.

### 5. Run via Docker (20 rounds, default)

```bash
docker compose -f docker/docker-compose.yml up --build
```

### 6. Run attestation demo

```bash
python tee/attestation/attestation_simulator.py
# Expected: [ATTESTATION] ✓ ATTESTATION VERIFIED
```

### 7. Run crypto tests

```bash
python crypto/certs/test_crypto.py
# Expected: ALL CRYPTO TESTS PASSED ✓ (4/4)
```

### 8. Run E2E dashboard tests

```bash
python dashboard/backend/test_e2e.py
# Expected: ALL E2E TESTS PASSED ✓ (11/11)
```

### 9. Generate TLS certificates

```bash
python crypto/certs/generate_tls_certs.py
# Output: crypto/certs/tls/ca.crt, server.crt, client.crt
```

### 10. Kubernetes cold start (full SGX cluster)

```bash
bash kubernetes/cold_start.sh
# Or with SGX hardware:
SGX=true bash kubernetes/cold_start.sh
```

---

## File Structure

```
IntelliClave/
├── fl/                     # Federated Learning (Flower)
│   ├── model.py            # HARClassifier (50→128→64→6)
│   ├── data_utils.py       # CSV loading, PCA, train/test split
│   ├── fl_client.py        # Flower client + DP + crypto
│   ├── fl_server.py        # Flower server + FedAvg + crypto
│   ├── run_server.py       # Server launcher
│   └── run_client.py       # Client launcher
│
├── privacy/                # Differential Privacy (Opacus)
│   ├── dp_trainer.py       # DP wrapper
│   ├── epsilon_sweep.py    # ε sweep experiments
│   └── budget_monitor.py   # Privacy budget tracking
│
├── tee/                    # Trusted Execution Environment
│   ├── attestation/        # SGX attestation simulator + integration
│   ├── sealed_storage/     # MRENCLAVE-bound sealed storage
│   ├── fl_enclave/         # Gramine manifests (server + 3 clients)
│   ├── full_stack_test/    # PyTorch + full stack in Gramine
│   └── benchmarks/         # TEE overhead measurement
│
├── crypto/certs/           # Cryptographic layer
│   ├── crypto_layer.py     # AES-256-GCM + RSA-2048
│   ├── crypto_context.py   # Server keypair lifecycle
│   ├── test_crypto.py      # 4 crypto tests
│   └── generate_tls_certs.py  # TLS cert generator
│
├── security/               # Security analysis
│   ├── attacks/            # Model inversion, membership inference, poisoning
│   └── stride/             # STRIDE threat model (all 6 categories)
│
├── dashboard/
│   ├── backend/main.py     # FastAPI backend (8 endpoints)
│   └── frontend/           # React + Recharts dashboard
│
├── kubernetes/             # K8s deployment
│   ├── cold_start.sh       # Full cluster cold start
│   ├── deployments/        # Server + 3 clients + SGX variants
│   ├── policies/           # NetworkPolicy
│   └── volumes/            # PVCs
│
├── docker/                 # Docker
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   └── docker-compose.yml  # 20-round default
│
├── data/processed/         # Frozen client CSVs (3 × non-IID)
├── results/                # All experiment outputs
│   ├── fl_rounds/          # Model checkpoints + metrics
│   ├── attacks/            # Attack simulation results
│   ├── graphs/             # Graph 6 final results figure
│   └── plots/              # DP plots (accuracy vs ε, etc.)
│
├── report/                 # All report sections
│   ├── tee_section.md
│   ├── security_section.md
│   ├── security_report.md
│   ├── stride_complete.md
│   ├── fl_dp_integration.md
│   ├── fl_process_flow.md
│   ├── deployment_scenario.md
│   └── figures/            # All report figures
│
├── evaluation/
│   ├── cross_validation.py
│   ├── metrics.py
│   └── generate_graph6.py  # Graph 6 generator
│
├── attestation.json        # Live attestation record
├── status.json             # Live training status
└── HANDOFF.md              # This file
```

---

## What Is Complete

| Area | Status |
|------|--------|
| Data pipeline (UCI HAR, PCA, 3 non-IID CSVs) | ✅ Complete |
| FL baseline (FedAvg, 10 rounds, 96.99% acc) | ✅ Complete |
| FL + DP integration (Opacus, ε=10, 91.27% acc) | ✅ Complete |
| Epsilon sweep (5 values, plots generated) | ✅ Complete |
| Cross-validation (5-fold, all 3 clients) | ✅ Complete |
| Crypto layer (AES-256-GCM + RSA, 4/4 tests) | ✅ Complete |
| TLS certificate generation | ✅ Complete |
| TEE attestation (simulated, VERIFIED) | ✅ Complete |
| SGX sealed storage | ✅ Complete |
| Gramine manifests (server + 3 clients) | ✅ Complete |
| TEE overhead benchmarks (35.2% avg) | ✅ Complete |
| Security attacks (3 attacks, all results saved) | ✅ Complete |
| STRIDE analysis (all 6 categories) | ✅ Complete |
| Kubernetes YAMLs + SGX pods + cold start | ✅ Complete |
| Docker + docker-compose (20 rounds) | ✅ Complete |
| Dashboard backend (FastAPI, 8 endpoints, 11/11 E2E) | ✅ Complete |
| Dashboard frontend (React, live data, attestation panel) | ✅ Complete |
| Graph 6 — final results figure (4 panels) | ✅ Complete |
| Report sections (TEE, security, FL, DP, STRIDE, deployment) | ✅ Complete |

---

## What Is Still Needed (Week 4+)

| Item | Owner | Notes |
|------|-------|-------|
| Report intro section | Member 1 | System overview, motivation, contributions |
| Report dataset section | Member 1 | UCI HAR description, non-IID split rationale |
| Report architecture section | Member 1 | System diagram, three-layer stack |
| Report FL methodology section | Member 1 | FL process, FedAvg, convergence |
| Report limitations section | Member 1 | gramine-direct, FedAvg Byzantine, DP cost |
| Eval methodology report section | Member 2 | Cross-validation setup, AUC-ROC, comparison tables |
| Related work section | Member 1 | FL privacy literature, SGX prior work |
| Report compilation (all sections → single PDF) | All | Assemble all `report/*.md` into final document |
| Slides 1–17 | All | Member 1: 1–6, Member 2: 7–12, Member 3: 13–17 |
| Full integration cold start test | All | End-to-end test with all components running |
| Demo recording | Member 2 | Screen recording of dashboard + FL run |

---

## Known Limitations

1. **gramine-direct** — WSL2 has no SGX hardware. Production deployment uses `gramine-sgx` with zero code changes.
2. **FedAvg Byzantine robustness** — 3.78% accuracy drop at 100% poison rate. Production fix: Krum or Trimmed Mean aggregation.
3. **Model inversion** — cosine similarity 0.90 in PCA space even with confidence masking. PCA vectors cannot be directly mapped to raw sensor data without `pca_model.pkl`.
4. **DP accuracy cost** — ε=10 costs 5.72% accuracy vs no-DP baseline. Lower ε values cost more (ε=1 → 50.89% accuracy).
5. **Dashboard dummy data** — `status.json` and `results.json` reflect the 5-round FL+DP run. A live training run would update these in real time.

---

## Quick Verification Checklist

```bash
# 1. Crypto tests
python crypto/certs/test_crypto.py
# → ALL CRYPTO TESTS PASSED ✓ (4/4)

# 2. Attestation
python tee/attestation/attestation_simulator.py
# → [ATTESTATION] ✓ ATTESTATION VERIFIED

# 3. Sealed storage
python tee/sealed_storage/sealed_storage.py
# → ALL SEALED STORAGE TESTS PASSED ✓

# 4. Attestation integration
python tee/attestation/attestation_integration.py
# → ATTESTATION INTEGRATION DEMO PASSED ✓

# 5. Dashboard E2E
python dashboard/backend/test_e2e.py
# → ALL E2E TESTS PASSED ✓ (11/11)

# 6. K8s YAML validation
bash kubernetes/validate.sh
# → ALL YAML FILES VALID ✓
```
