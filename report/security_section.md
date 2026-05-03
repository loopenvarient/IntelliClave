# IntelliClave — Report Section: Security Analysis

---

## 1. Overview

This section documents the security architecture, threat model, attack simulations,
and cryptographic implementation of IntelliClave. Security is implemented at three
independent layers — each providing protection even if the others are bypassed.

| Layer | Technology | Threat Addressed |
|-------|-----------|-----------------|
| Transport encryption | AES-256-GCM + RSA-2048 | Weight tampering, gradient leakage |
| Differential privacy | Opacus DP-SGD (ε=10) | Membership inference, gradient inversion |
| TEE attestation | Gramine + SGX | Rogue server, manifest tampering, enclave escape |

---

## 2. Cryptographic Layer

### Design

All model weight updates are encrypted before leaving each client. The scheme
uses symmetric authenticated encryption with asymmetric key exchange:

```
Client side:
  1. Generate fresh 32-byte AES session key (per round)
  2. Encrypt weights with AES-256-GCM → ciphertext + 16-byte auth tag
  3. Encrypt session key with server RSA-2048 public key (OAEP-SHA256)
  4. Compute SHA-256 fingerprint of plaintext weights
  5. Send: {encrypted_key, nonce, ciphertext, weight_hash}

Server side:
  1. Decrypt session key with RSA private key
  2. Decrypt ciphertext with AES-GCM → raises InvalidTag if tampered
  3. Verify SHA-256 fingerprint → raises ValueError if hash mismatch
  4. Proceed with FedAvg aggregation
```

### Why AES-256-GCM

GCM provides **authenticated encryption** — confidentiality and integrity in one
pass. The 16-byte auth tag covers every byte of the ciphertext. Any modification
to the ciphertext (even a single bit flip) causes decryption to fail with
`InvalidTag`. This directly mitigates STRIDE threat T1 (weight tampering).

### Why RSA-2048 for Key Exchange

Asymmetric key exchange means clients only need the server's public key — no
shared secret needs to be distributed. In production, the public key is baked
into the client Docker image at build time and verified against the attestation
report.

### Test Results

All 4 cryptographic properties were verified:

| Test | Property | Result |
|------|---------|--------|
| 1 | Encrypt → decrypt → identical arrays (max diff: 0.00e+00) | ✅ PASS |
| 2 | Valid payload integrity check accepted | ✅ PASS |
| 3 | Tampered ciphertext (bit flip at byte 10) rejected | ✅ PASS |
| 4 | Same plaintext → different ciphertext (IND-CPA) | ✅ PASS |

---

## 3. STRIDE Threat Model

STRIDE analysis was performed across all 6 threat categories. 21 threats were
identified and mitigated.

### S — Spoofing (3 threats)

| ID | Threat | Mitigation | Risk |
|----|--------|-----------|------|
| S1 | Client impersonation | mTLS + K8s NetworkPolicy | LOW |
| S2 | Replay attack | Round number binding + TLS session freshness | LOW |
| S3 | Rogue server | TEE attestation (MRENCLAVE verification) | LOW |

**S3 tested:** Rogue server with wrong MRENCLAVE blocked by all 3 clients.

### T — Tampering (4 threats)

| ID | Threat | Mitigation | Risk |
|----|--------|-----------|------|
| T1 | Weight tampering in transit | AES-256-GCM auth tag + SHA-256 fingerprint | VERY LOW |
| T2 | Gradient poisoning (Byzantine) | DP-SGD clipping (norm=1.0) + FedAvg dilution | MEDIUM |
| T3 | Model checkpoint tampering | SGX sealed storage (MRENCLAVE-bound) | LOW |
| T4 | Manifest tampering | MRENCLAVE binding — clients reject mismatched server | LOW |

**T1 tested:** Tampered ciphertext rejected (crypto Test 3).
**T2 tested:** 3.78% accuracy drop at 100% poison rate (gradient poisoning simulation).
**T3 tested:** Tampered sealed blob rejected (sealed storage demo).

### R — Repudiation (3 threats)

| ID | Threat | Mitigation | Risk |
|----|--------|-----------|------|
| R1 | Client denies round participation | Immutable per-round log in `fl_privacy.json` | LOW |
| R2 | Server denies performing aggregation | MRENCLAVE binds code to output | LOW |
| R3 | Client denies submitting specific update | SHA-256 weight fingerprint per round | LOW |

### I — Information Disclosure (4 threats)

| ID | Threat | Mitigation | Result | Risk |
|----|--------|-----------|--------|------|
| I1 | Membership inference | DP-SGD ε=10 | AUC=0.503 (random) | LOW |
| I2 | Model inversion | Confidence masking + rate limiting | Cosine sim=0.90 (PCA only) | MEDIUM |
| I3 | Gradient leakage | AES-256-GCM + DP noise + FedAvg | Not feasible | LOW |
| I4 | Side-channel | gramine-sgx MEE (production) | N/A prototype | MEDIUM (proto) |

### D — Denial of Service (4 threats)

| ID | Threat | Mitigation | Risk |
|----|--------|-----------|------|
| D1 | Flood FL server | K8s NetworkPolicy (ingress: role=fl-client only) | LOW |
| D2 | Exhaust privacy budget | Budget monitor halts at budget_exhausted=True | LOW |
| D3 | Flood dashboard API | Rate limiter: 100 req/60s → HTTP 429 | LOW |
| D4 | Oversized weight updates | Flower shape validation | LOW |

### E — Elevation of Privilege (3 threats)

| ID | Threat | Mitigation | Risk |
|----|--------|-----------|------|
| E1 | Client accesses other clients' data | FL architecture — data never transmitted | VERY LOW |
| E2 | Enclave escape | gramine-sgx hardware isolation (production) | MEDIUM (proto) / LOW (prod) |
| E3 | Server private key theft | Sealed to MRENCLAVE; HSM in production | LOW |

---

## 4. Attack Simulations

Three attacks were implemented and run against the trained global model.

### Attack 1: Model Inversion

**Threat:** I2 — Information Disclosure

**Method:** Gradient-based input optimisation. Starting from random noise in
50-dimensional PCA space, minimise cross-entropy loss for each target class.
Measure cosine similarity between reconstructed input and real class centroid.

**Results:**

| Mode | Avg Cosine Similarity | Avg Confidence | High-Risk Classes |
|------|----------------------|---------------|------------------|
| Unmitigated | 0.853 | 0.972 | 5/6 |
| Confidence masked | 0.897 | 0.866 | 6/6 |

**Verdict:** VULNERABLE in PCA space. The model leaks class-level feature
structure. However, reconstructed inputs are 50-dimensional PCA vectors —
not raw sensor readings. Inverting PCA requires `data/samples/pca_model.pkl`,
which is not exposed by the API.

**Mitigations applied:**
- Confidence masking: API returns predicted label only (not softmax probabilities)
- Rate limiting: 100 queries per 60 seconds per IP

### Attack 2: Membership Inference

**Threat:** I1 — Information Disclosure

**Method:** Threshold attack on softmax confidence. Split each client's data
into members (training) and non-members (held-out). Query model with both sets.
Find confidence threshold that maximises member/non-member classification accuracy.
Measure AUC — random = 0.5, perfect = 1.0.

**Results:**

| Client | AUC | Confidence Gap | Risk |
|--------|-----|---------------|------|
| FitLife (1) | 0.493 | −0.002 | LOW |
| MediTrack (2) | 0.502 | +0.003 | LOW |
| CareWatch (3) | 0.514 | +0.006 | LOW |
| **Average** | **0.503** | **+0.003** | **LOW** |

**Verdict:** RESISTANT. AUC ≈ 0.5 means the attack performs at random chance.
DP-SGD prevents the model from memorising individual training samples.

### Attack 3: Gradient Poisoning

**Threat:** T2 — Tampering

**Method:** Label-flip Byzantine attack. Client 1 (FitLife) flips training labels
to WALKING_UPSTAIRS at varying rates (0%, 10%, 30%, 50%, 100%). Run full FL
simulation (5 rounds, FedAvg) and measure global model accuracy degradation.

**Results:**

| Poison Rate | Accuracy | Macro F1 | Accuracy Drop |
|-------------|----------|----------|---------------|
| 0% (clean) | 95.78% | 95.87% | — |
| 10% | 95.73% | 95.82% | −0.05% |
| 30% | 95.73% | 95.86% | −0.05% |
| 50% | 95.78% | 95.83% | 0.00% |
| 100% | 91.99% | 91.92% | **−3.78%** |

**Verdict:** MEDIUM risk. FedAvg + DP-SGD gradient clipping limits damage to
3.78% even at 100% label flip on one client. The attack is detectable via
anomaly detection on client update norms (not yet implemented — future work).

---

## 5. Differential Privacy Results

DP-SGD was integrated into the FL pipeline using Opacus. The key design decision
was attaching the PrivacyEngine once in `__init__` with `total_epochs = local_epochs × num_fl_rounds`, ensuring Opacus calibrates noise for the full training duration.

### Privacy Budget Consumption (5 rounds, ε=10)

| Round | Avg ε | Budget Remaining | Exhausted |
|-------|-------|-----------------|-----------|
| 1 | 4.982 | 5.018 | False |
| 2 | 6.548 | 3.452 | False |
| 3 | 7.826 | 2.174 | False |
| 4 | 8.957 | 1.043 | False |
| 5 | 9.992 | 0.008 | False |

Budget never exhausted. ε stays within target across all rounds.

### Privacy-Utility Tradeoff

| Target ε | Test Accuracy | Privacy Cost vs Baseline |
|----------|--------------|--------------------------|
| 1.0 | 50.89% | −46.1% |
| 2.0 | 62.00% | −34.9% |
| 5.0 | 61.51% | −35.5% |
| **10.0** | **69.89%** | **−27.1%** |
| 20.0 | 77.46% | −19.5% |
| No DP (FL baseline) | 96.99% | — |

**Selected operating point: ε=10 → 91.27% accuracy (FL+DP, 5 rounds).**

The FL+DP model achieves 91.27% weighted accuracy across all 3 clients,
compared to 96.99% for the no-DP FL baseline. The 5.72% privacy cost is
the price paid for the ε=10 differential privacy guarantee.

### Per-Client FL+DP Results

| Client | Test Size | Accuracy | Macro F1 |
|--------|----------|----------|----------|
| FitLife (1) | 621 | 90.18% | 89.87% |
| MediTrack (2) | 686 | 89.50% | 89.29% |
| CareWatch (3) | 754 | 93.77% | 93.97% |
| **Overall weighted** | **2,061** | **91.27%** | **91.17%** |

---

## 6. Kubernetes Security

### Network Policy

The `fl-network-policy` enforces strict traffic isolation:

- **Ingress:** Only pods with `role: fl-client` label can reach the FL server on port 8080
- **Egress:** Clients can only send traffic to pods with `app: fl-server` label
- **DNS:** All pods can reach kube-dns (UDP/TCP port 53)
- **Default:** All other traffic denied

This prevents external attackers from reaching the FL server and prevents
clients from communicating with each other.

### SGX Pod Security

- Init containers run attestation before FL starts — clients abort if MRENCLAVE mismatches
- Client CSVs mounted read-only — raw data cannot be exfiltrated via volume
- Crypto keys loaded from K8s Secret — never baked into Docker image
- SGX device plugin restricts which nodes can run enclave workloads

### YAML Validation

All K8s YAML files pass `kubectl apply --dry-run=client` validation.
Run: `bash kubernetes/validate.sh`

---

## 7. Residual Risks

| Risk | Severity | Reason Accepted | Production Fix |
|------|---------|----------------|----------------|
| Model inversion cosine sim=0.90 | MEDIUM | PCA space only — not raw sensor data | Output perturbation noise on logits |
| gramine-direct no hardware isolation | MEDIUM | WSL2 has no SGX device | Replace with gramine-sgx — zero code changes |
| FedAvg not Byzantine-robust | MEDIUM | 3.78% drop acceptable for prototype | Replace FedAvg with Krum or Trimmed Mean |
| Side-channel in gramine-direct | MEDIUM | Prototype limitation | gramine-sgx MEE in production |

---

## 8. Evidence Summary

| Claim | Evidence File |
|-------|--------------|
| Crypto tamper detection | `crypto/certs/test_crypto.py` — Test 3 PASS |
| Attestation VERIFIED | `attestation.json`, `tee/attestation/attestation_integration.py` |
| Sealed storage working | `tee/sealed_storage/sealed_storage.py` — all tests PASS |
| Membership inference resistant | `results/attacks/membership_inference.json` — AUC=0.503 |
| Model inversion results | `results/attacks/model_inversion.json` |
| Gradient poisoning results | `results/attacks/gradient_poisoning.json` |
| DP budget never exhausted | `results/fl_rounds/fl_privacy.json` |
| TEE overhead acceptable | `results/benchmarks/tee_overhead_final.json` — 35.2% avg |
| K8s YAML valid | `kubernetes/validate.sh` |
