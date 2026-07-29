# IntelliClave — STRIDE: Spoofing Category (Detailed)

---

## What Spoofing Means in This System

Spoofing in IntelliClave means an attacker pretends to be a legitimate participant —
either a trusted FL client, the aggregation server, or both — to inject poisoned
updates, steal model weights, or redirect training.

There are three spoofing surfaces:

| Surface | Attacker Goal |
|---------|--------------|
| Client → Server | Impersonate a legitimate client to inject poisoned gradients |
| Server → Client | Impersonate the aggregation server to steal client weight updates |
| Replay | Replay a previous round's valid updates to manipulate aggregation |

---

## Threat S1 — Client Identity Spoofing

### Scenario

A fourth party (not FitLife, MediTrack, or CareWatch) connects to the FL server
and submits weight updates as if it were Client 2 (MediTrack). The attacker's goal
is to inject poisoned gradients that degrade the global model or cause targeted
misclassification of a specific activity class.

### Attack Path

```
Attacker machine
    │
    │  Connects to fl-server:8080 (Flower gRPC)
    │  Sends: client_id=2, weights=<poisoned>
    ▼
FL Server (FedAvg)
    │
    │  Aggregates poisoned weights with legitimate client weights
    ▼
Global model degraded
```

### Why This Is Possible Without Mitigation

Flower's default gRPC transport does not enforce client authentication.
Any machine that can reach port 8080 can submit weight updates.

### Mitigations Implemented

**1. TLS 1.3 mutual authentication (mTLS)**

Flower uses gRPC over TLS by default. In IntelliClave, the server is configured
with a server certificate. Clients must present a valid client certificate signed
by the IntelliClave CA to connect.

```
Client certificate chain:
  IntelliClave Root CA
    └── fl-client-1.intelliclave (FitLife)
    └── fl-client-2.intelliclave (MediTrack)
    └── fl-client-3.intelliclave (CareWatch)
```

An attacker without a valid signed certificate cannot complete the TLS handshake.

**2. Kubernetes NetworkPolicy**

The `fl-network-policy` restricts which pods can reach the FL server:

```yaml
ingress:
  - from:
      - podSelector:
          matchLabels:
            role: fl-client
    ports:
      - protocol: TCP
        port: 8080
```

Only pods with `role: fl-client` label can send traffic to port 8080.
An external attacker cannot reach the server at all from outside the cluster.

**3. Gradient poisoning resilience (tested)**

Even if a client is compromised, FedAvg + DP-SGD limits damage:
- DP-SGD clips each gradient to `max_grad_norm=1.0` before aggregation
- 100% label flip on one client causes only **3.8% accuracy drop**
- See: `results/attacks/gradient_poisoning.json`

### Residual Risk

LOW. Requires stealing a valid TLS client certificate AND bypassing the
Kubernetes NetworkPolicy. In production, certificates are rotated per round.

---

## Threat S2 — Replay Attack

### Scenario

An attacker captures valid weight updates from Client 1 during Round 3 and
replays them in Round 5. The goal is to roll back the global model's learning
or amplify the influence of a specific client's data.

### Attack Path

```
Round 3:
  Client 1 → Server: weights_round3 (captured by attacker)

Round 5:
  Attacker → Server: weights_round3 (replayed as if from Client 1)
  Server aggregates stale weights → global model regresses
```

### Mitigations Implemented

**1. Round number binding**

Every client submission includes the current round number in the Flower
`FitRes` metadata. The server's `aggregate_fit()` validates that the round
number in the submission matches the current server round:

```python
# fl/fl_server.py — aggregate_fit()
for client_proxy, fit_res in results:
    submitted_round = fit_res.metrics.get("round", -1)
    if submitted_round != server_round:
        # reject stale submission
        continue
```

A replayed Round 3 submission is rejected in Round 5.

**2. Weight fingerprinting**

Each round's aggregated weights are SHA-256 fingerprinted and stored in
`fl_privacy.json`. If a replayed submission changes the aggregated fingerprint
unexpectedly, it is detectable in the audit log.

**3. TLS session freshness**

TLS 1.3 uses ephemeral key exchange (ECDHE). Each session has a unique session
key. A captured TLS session cannot be replayed — the session key is gone after
the connection closes.

### Residual Risk

LOW. Round number validation + TLS session freshness together make replay
attacks infeasible.

---

## Threat S3 — Rogue Server (Server Impersonation)

### Scenario

An attacker sets up a fake FL server at the same address as the IntelliClave
aggregator. Clients connect to the fake server, send their weight updates
(which contain gradient information), and the attacker collects them.
The attacker then performs gradient inversion to reconstruct training data.

### Attack Path

```
DNS poisoning or ARP spoofing
    │
    ▼
Fake server at fl-server:8080
    │
    │  Clients connect (no attestation check)
    │  Clients send weight updates
    ▼
Attacker collects gradients → gradient inversion attack
```

### Why This Is Dangerous

Without attestation, clients have no way to verify they are talking to the
real IntelliClave enclave. A rogue server could:
- Collect weight updates and attempt gradient inversion
- Return manipulated global weights to degrade client models
- Selectively drop clients to bias the global model

### Mitigations Implemented

**1. TEE Remote Attestation**

Before sending any weight updates, each client requests an attestation report
from the server. The report contains the server's MRENCLAVE — a SHA-256 hash
of the enclave's code and manifest.

```python
# Client-side attestation check (tee/attestation/attestation_simulator.py)
report = request_attestation(server_addr)
expected_mrenclave = load_expected_mrenclave()  # hardcoded at build time

if report["mrenclave"] != expected_mrenclave:
    raise SecurityError("Server MRENCLAVE mismatch — possible rogue server")
```

A fake server cannot produce a valid attestation report with the correct
MRENCLAVE unless it is running exactly the same enclave code.

**2. TLS certificate pinning**

The server's TLS certificate is pinned in each client's configuration.
A rogue server with a different certificate is rejected at the TLS handshake.

**3. Kubernetes NetworkPolicy (egress)**

Clients can only send traffic to pods with `app: fl-server` label:

```yaml
egress:
  - to:
      - podSelector:
          matchLabels:
            app: fl-server
    ports:
      - protocol: TCP
        port: 8080
```

DNS poisoning within the cluster is prevented by restricting egress to
labelled pods only.

### Attestation Demo Output

```
[ATTESTATION] Requesting quote...
[ATTESTATION] Quote received: 218 bytes
[ATTESTATION] MRENCLAVE: 574d8f62a9632715...
[ATTESTATION] ✓ ATTESTATION VERIFIED
[ATTESTATION] Mode: Gramine simulation (WSL2)
```

Run: `python3 tee/attestation/attestation_simulator.py`

### Residual Risk

LOW (prototype) / VERY LOW (production).

In `gramine-direct` mode, the attestation is simulated — a sufficiently
privileged attacker on the same host could forge the MRENCLAVE.
In `gramine-sgx` production mode, the MRENCLAVE is computed by the CPU
during enclave load and cannot be forged without breaking Intel's attestation
infrastructure.

---

## Spoofing Mitigations Summary

| Threat | Primary Mitigation | Secondary Mitigation | Residual Risk |
|--------|-------------------|---------------------|---------------|
| S1 — Client impersonation | mTLS client certificates | K8s NetworkPolicy (ingress) | LOW |
| S2 — Replay attack | Round number binding in FitRes | TLS 1.3 session freshness | LOW |
| S3 — Rogue server | TEE attestation (MRENCLAVE check) | TLS certificate pinning + K8s egress policy | LOW (prototype) / VERY LOW (prod) |

---

## Test Commands

```bash
# Verify attestation produces VERIFIED output
python3 tee/attestation/attestation_simulator.py

# Verify crypto layer rejects tampered payloads (Test 3)
python3 crypto/certs/test_crypto.py

# Verify K8s network policy syntax
bash kubernetes/validate.sh

# Verify gradient poisoning resilience
python3 security/attacks/gradient_poisoning.py
# Expected: risk_level = "MEDIUM", accuracy_drop < 0.05
```

---

## Evidence Files

| Evidence | File |
|---------|------|
| Attestation VERIFIED output | `attestation.json` |
| Crypto tamper rejection | `crypto/certs/test_crypto.py` Test 3 |
| Gradient poisoning results | `results/attacks/gradient_poisoning.json` |
| Network policy YAML | `kubernetes/policies/network-policy.yaml` |
| K8s dry-run validation | `kubernetes/validate.sh` |
