# IntelliClave — Complete STRIDE Threat Model

_Last updated: 2026-05-03_

---

## System Overview

IntelliClave is a privacy-preserving federated learning pipeline with three
security layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Federated Learning | Flower + FedAvg | Raw data never leaves each client |
| Differential Privacy | Opacus DP-SGD (ε=10) | Gradients cannot reveal training data |
| Trusted Execution Environment | Gramine + Intel SGX | Aggregation is hardware-sealed |

**Dataset:** UCI HAR — 10,299 samples, 6 activity classes, 3 non-IID clients
**Model:** HARClassifier (50→128→64→6), trained with FedAvg over 10 rounds

---

## Trust Boundaries

```
┌──────────────────────────────────────────────────────────────────┐
│  Client Boundary (per company — data never crosses this line)    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  FitLife    │  │  MediTrack  │  │  CareWatch  │             │
│  │  3,105 rows │  │  3,426 rows │  │  3,768 rows │             │
│  │  active     │  │  mixed      │  │  sedentary  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │ AES-256-GCM     │ AES-256-GCM    │ AES-256-GCM
          │ encrypted       │ encrypted      │ encrypted
          │ weights only    │ weights only   │ weights only
          ▼                 ▼                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Server Boundary (IntelliClave Aggregator)                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  TEE Enclave (gramine-direct / gramine-sgx)                │  │
│  │  FedAvg aggregation                                        │  │
│  │  RSA private key (sealed to MRENCLAVE)                     │  │
│  │  Model checkpoints (sealed to MRENCLAVE)                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## S — Spoofing

| ID | Threat | Component | Mitigation | Residual Risk |
|----|--------|-----------|-----------|---------------|
| S1 | Malicious client impersonates legitimate FL participant | FL Server | mTLS client certificates + K8s NetworkPolicy (ingress: role=fl-client only) | LOW |
| S2 | Replay of old round's weight updates | FL Server | Round number in FitRes metadata; TLS 1.3 session freshness | LOW |
| S3 | Rogue server impersonates IntelliClave aggregator | FL Clients | TEE attestation — MRENCLAVE verified before every FL connection | LOW |

**Tested:** S3 — rogue server with wrong MRENCLAVE blocked. See `tee/attestation/attestation_integration.py`.
**Detailed analysis:** `security/stride/STRIDE_SPOOFING.md`

---

## T — Tampering

| ID | Threat | Component | Mitigation | Residual Risk |
|----|--------|-----------|-----------|---------------|
| T1 | Weight tampering in transit | Network | AES-256-GCM auth tag + SHA-256 fingerprint | VERY LOW |
| T2 | Gradient poisoning (Byzantine) | FL Server | DP-SGD clipping (norm=1.0) + FedAvg dilution | MEDIUM |
| T3 | Model checkpoint tampering on disk | Server storage | SGX sealed storage (MRENCLAVE-bound AES-GCM) | LOW (SGX) |
| T4 | Manifest tampering to expand enclave permissions | TEE | MRENCLAVE binding — any manifest change → client rejection | LOW |

**Tested:** T1 — tampered ciphertext rejected (Test 3, `crypto/certs/test_crypto.py`). T2 — 3.78% accuracy drop at 100% poison rate (`results/attacks/gradient_poisoning.json`). T3 — tampered sealed blob rejected (`tee/sealed_storage/sealed_storage.py`).
**Detailed analysis:** `security/stride/STRIDE_TAMPERING.md`

---

## R — Repudiation

| ID | Threat | Component | Mitigation | Residual Risk |
|----|--------|-----------|-----------|---------------|
| R1 | Client denies participating in a round | FL Server | Immutable per-round log: client_id, round, epsilon, delta in `fl_privacy.json` | LOW |
| R2 | Server denies performing aggregation with specific inputs | TEE | MRENCLAVE binds aggregation code to output — any code change is detectable | LOW |
| R3 | Client denies submitting specific weight update | Crypto | SHA-256 weight fingerprint logged per round; AES-GCM payload is non-repudiable | LOW |

---

## I — Information Disclosure

| ID | Threat | Component | Mitigation | Result | Residual Risk |
|----|--------|-----------|-----------|--------|---------------|
| I1 | Membership inference | Global model | DP-SGD ε=10 | AUC=0.503 (random) | LOW |
| I2 | Model inversion | Dashboard API | Confidence masking + rate limiting (100/60s) | Cosine sim=0.90 (PCA only) | MEDIUM |
| I3 | Gradient leakage | Network | AES-256-GCM + DP-SGD noise + FedAvg | Not feasible | LOW |
| I4 | Side-channel (memory/cache) | TEE | gramine-sgx MEE (production) | N/A in prototype | MEDIUM (proto) |

**Tested:** I1 — `results/attacks/membership_inference.json`. I2 — `results/attacks/model_inversion.json` + `model_inversion_mitigated.json`.
**Detailed analysis:** `security/stride/STRIDE_INFO_DISCLOSURE.md`

---

## D — Denial of Service

| ID | Threat | Component | Mitigation | Residual Risk |
|----|--------|-----------|-----------|---------------|
| D1 | Flood FL server with fake connections | FL Server | K8s NetworkPolicy: ingress restricted to role=fl-client pods | LOW |
| D2 | Exhaust privacy budget by forcing extra rounds | DP Budget | Budget monitor halts training when budget_exhausted=True | LOW |
| D3 | Flood dashboard /predict endpoint | Dashboard API | Rate limiter: 100 req/60s per IP → HTTP 429 | LOW |
| D4 | Oversized weight updates crash server | FL Server | Flower validates weight shape against model architecture | LOW |

---

## E — Elevation of Privilege

| ID | Threat | Component | Mitigation | Residual Risk |
|----|--------|-----------|-----------|---------------|
| E1 | Compromised client accesses other clients' raw data | Architecture | FL by design — raw data never transmitted | VERY LOW |
| E2 | Attacker escapes Gramine enclave to host OS | TEE | gramine-sgx hardware isolation (production) | MEDIUM (proto) / LOW (prod) |
| E3 | Attacker obtains server RSA private key | Crypto | Key sealed to MRENCLAVE; never transmitted; HSM in production | LOW |

---

## Attack Simulation Results

| Attack | Script | Result | Risk |
|--------|--------|--------|------|
| Model inversion (unmitigated) | `security/attacks/model_inversion.py` | Avg cosine sim=0.853, 5/6 HIGH | HIGH |
| Model inversion (mitigated) | Same, confidence masked | Avg cosine sim=0.897 (PCA only) | MEDIUM |
| Membership inference | `security/attacks/membership_inference.py` | Avg AUC=0.503 (random) | LOW |
| Gradient poisoning | `security/attacks/gradient_poisoning.py` | 3.78% accuracy drop at 100% poison | MEDIUM |

---

## Mitigations Implemented

| Mitigation | File | STRIDE |
|-----------|------|--------|
| AES-256-GCM weight encryption | `crypto/certs/crypto_layer.py` | T1, I3 |
| RSA-2048 key encapsulation | `crypto/certs/crypto_layer.py` | T1 |
| SHA-256 weight fingerprinting | `crypto/certs/crypto_layer.py` | T1, R3 |
| DP-SGD (Opacus, ε=10, norm=1.0) | `fl/fl_client.py` | I1, T2 |
| TEE attestation (MRENCLAVE) | `tee/attestation/attestation_integration.py` | S3, T4, R2 |
| SGX sealed storage | `tee/sealed_storage/sealed_storage.py` | T3, E3 |
| Confidence masking | `dashboard/backend/main.py` | I2 |
| Rate limiting (100/60s) | `dashboard/backend/main.py` | D3 |
| Privacy budget monitor | `privacy/budget_monitor.py` | D2 |
| K8s NetworkPolicy | `kubernetes/policies/network-policy.yaml` | D1, S1 |
| mTLS (Flower gRPC) | Flower default transport | S1, S2 |
| Round number binding | `fl/fl_server.py` | S2 |

---

## Residual Risks Accepted for Prototype

| Risk | Reason Accepted | Production Fix |
|------|----------------|----------------|
| Model inversion cosine sim=0.90 | PCA space only — not raw sensor data | Output perturbation noise on logits |
| gramine-direct no hardware isolation | WSL2 has no SGX device | Replace with gramine-sgx — zero code changes |
| FedAvg not Byzantine-robust | 3.78% drop is acceptable for prototype | Replace with Krum or Trimmed Mean |
| Side-channel in gramine-direct | Prototype limitation | gramine-sgx MEE in production |
