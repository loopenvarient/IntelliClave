# IntelliClave
## Confidential Computing Secure Processing System
> Privacy-preserving AI pipeline using TEE · Federated Learning · Differential Privacy

---

## What Is This

IntelliClave is a privacy-preserving AI pipeline using Confidential Computing
Multiple organisations train a shared AI model collaboratively
without any organisation's data ever leaving their premises.

Three layers make this possible:

- **Federated Learning** — raw data never leaves each client
- **Differential Privacy** — gradients cannot reveal training data
- **Trusted Execution Environment** — computation is hardware-sealed

---

## Team

| Member | Responsibility |
|--------|----------------|
| M1 | Data + Federated Learning + Deployment Scenario |
| M2 | Differential Privacy + Evaluation + Dashboard |
| M3 | TEE + SGX + Attestation + Kubernetes + Security |

---

## Hardware

- **CPU:** Intel Core i7-1355U
- **SGX:** gramine-direct mode (WSL2 Ubuntu 22.04)
- **Duration:** 6 weeks

---

## Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| ML Model | PyTorch | 2.1.0 |
| Federated Learning | Flower (flwr) | 1.6.0 |
| Differential Privacy | Opacus | 1.4.0 |
| TEE | Gramine (gramine-direct) | 1.5 |
| Cryptography | AES-256-GCM + HMAC-SHA256 + TLS 1.3 | — |
| Infrastructure | Docker + Kubernetes (minikube) | — |
| Dashboard Backend | FastAPI | 0.104.1 |
| Dashboard Frontend | React + Recharts | 18.x |

---

## Project Structure
intelliclave/
├── fl/                     # Federated Learning (Flower server + client)
├── privacy/                # Differential Privacy (Opacus wrapper)
├── tee/                    # Gramine manifests + enclave code
├── crypto/                 # AES-256, HMAC, TLS layer
├── kubernetes/             # K8s deployment YAMLs
├── dashboard/              # React frontend + FastAPI backend
├── evaluation/             # Metrics script + benchmark script
├── security/               # STRIDE analysis + attack simulations
├── data/                   # Data configs (CSVs excluded from Git)
├── results/                # Experiment results (generated files)
├── report/                 # Report sections
├── docker/                 # Dockerfiles
└── requirements.txt        # Pinned Python dependencies

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Kubernetes cluster
minikube start --driver=docker --cpus=4 --memory=6144

# 4. Deploy all components
kubectl apply -f kubernetes/ -n intelliclave

# 5. Start dashboard
cd dashboard/frontend/intelliclave-ui && npm start
```

---

<!-- ## Dataset

--- -->

## Interface Contracts

All shared interfaces are documented in `contracts.md`.
Read this before writing any code that touches shared files.

---

## Branch Rules
main     → stable, working code only — never commit directly
dev      → integration branch
feature/ → individual feature branches

- Always create a PR to merge into `dev`
- Never push directly to `main`
