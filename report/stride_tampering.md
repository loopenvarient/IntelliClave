# IntelliClave — STRIDE: Tampering Category (Detailed)

---

## What Tampering Means in This System

Tampering in IntelliClave means an attacker modifies data in transit or at rest
to corrupt the global model, degrade accuracy, or cause targeted misclassification.
There are four tampering surfaces:

| Surface | What Can Be Tampered | Attacker Goal |
|---------|---------------------|---------------|
| Network | Encrypted weight updates in transit | Corrupt FedAvg aggregation |
| Client | Local training labels before gradient computation | Poison global model |
| Server disk | Model checkpoints after aggregation | Backdoor the saved model |
| Enclave | Gramine manifest to expand permissions | Escape TEE boundary |

---

## Threat T1 — Weight Tampering in Transit

### Scenario

An attacker with network access (man-in-the-middle) intercepts the gRPC stream
between Client 2 (MediTrack) and the FL server. They flip bits in the serialised
weight update to corrupt the FedAvg aggregation.

### Attack Path

```
Client 2 → [network] → Server
              ↑
         Attacker flips bytes in the gRPC payload
              ↓
Server receives corrupted weights → FedAvg produces wrong global model
```

### Mitigation: AES-256-GCM Authenticated Encryption

Every weight update is encrypted with a fresh AES-256-GCM session key before
transmission. GCM provides **authenticated encryption** — the 16-byte auth tag
covers every byte of the ciphertext. Any modification to the ciphertext causes
`InvalidTag` to be raised during decryption.

```python
# crypto/certs/crypto_layer.py — aes_decrypt()
aesgcm = AESGCM(session_key)
return aesgcm.decrypt(nonce, ciphertext, associated_data=None)
# Raises cryptography.exceptions.InvalidTag if ciphertext was modified
```

**Test 3 in `crypto/certs/test_crypto.py` confirms this:**
```
[Test 3] Tampered ciphertext → rejected
  Tampered ciphertext    : bits flipped at byte 10
  Decryption rejected    : True
  ✓ PASS
```

### Mitigation: SHA-256 Weight Fingerprint

In addition to GCM, a SHA-256 hash of the plaintext weights is included in the
encrypted payload. After decryption, the server recomputes the hash and compares:

```python
actual_hash = hashlib.sha256(plaintext).hexdigest()
if actual_hash != expected_hash:
    raise ValueError("Weight integrity check failed")
```

Two independent integrity checks — GCM tag + SHA-256 — must both pass.

### Residual Risk

**VERY LOW.** Breaking AES-256-GCM requires either:
- Forging a valid GCM auth tag (computationally infeasible)
- Obtaining the session key (new key per round, RSA-2048 protected)

---

## Threat T2 — Gradient Poisoning (Byzantine Attack)

### Scenario

Client 1 (FitLife) is compromised. The attacker flips training labels before
local training, causing the client to submit poisoned gradient updates that
degrade the global model or cause targeted misclassification of WALKING_UPSTAIRS.

### Attack Path

```
Client 1 local data: labels flipped (WALKING → WALKING_UPSTAIRS)
    ↓
Local training produces poisoned gradients
    ↓
Poisoned weights sent to server (encrypted, but content is poisoned)
    ↓
FedAvg averages poisoned + 2 clean clients
    ↓
Global model accuracy degrades
```

### Test Results (from `results/attacks/gradient_poisoning.json`)

| Poison Rate | Accuracy | Macro F1 | Accuracy Drop |
|-------------|----------|----------|---------------|
| 0% (clean)  | 95.78%   | 95.87%   | —             |
| 10%         | 95.73%   | 95.82%   | −0.05%        |
| 30%         | 95.73%   | 95.86%   | −0.05%        |
| 50%         | 95.78%   | 95.83%   | 0.00%         |
| 100%        | 91.99%   | 91.92%   | **−3.78%**    |

**Risk level: MEDIUM.** Even 100% label flip on one client causes only 3.78%
accuracy drop. FedAvg dilutes the poisoned client's influence (1 of 3 clients,
weighted by sample count).

### Mitigation: DP-SGD Gradient Clipping

Opacus clips each per-sample gradient to `max_grad_norm=1.0` before aggregation.
This limits the maximum influence any single sample — and therefore any single
poisoned label — can have on the weight update:

```
Without clipping: poisoned gradient magnitude = unbounded
With clipping:    poisoned gradient magnitude ≤ 1.0
```

The noise added by DP-SGD further masks the poisoning signal.

### Mitigation: FedAvg Sample-Weighted Averaging

FedAvg weights each client's update by their sample count:
- Client 1: 2,484 train samples (smallest)
- Client 2: 2,741 train samples
- Client 3: 3,014 train samples (largest)

Client 1's poisoned update has the smallest weight in the average.

### Residual Risk

**MEDIUM.** FedAvg is not Byzantine-robust by design. Production mitigation:
replace FedAvg with **Krum** or **Trimmed Mean** aggregation, which explicitly
reject outlier updates before averaging.

---

## Threat T3 — Model Checkpoint Tampering

### Scenario

An attacker with access to the server's filesystem modifies
`results/fl_rounds/global_model_latest.pth` after aggregation to insert a
backdoor — e.g., causing the model to always predict LAYING for a specific
input pattern.

### Attack Path

```
Server writes global_model_latest.pth after round N
    ↓
Attacker modifies the .pth file on disk
    ↓
Clients load the backdoored model in round N+1
    ↓
Backdoor propagates to all clients
```

### Mitigation: SGX Sealed Storage

In `gramine-sgx` mode, model checkpoints are sealed to the server enclave's
MRENCLAVE using `tee/sealed_storage/sealed_storage.py`:

```python
# After each round, server seals the checkpoint
seal_model_checkpoint("results/fl_rounds/global_model_latest.pth")
```

The sealed file can only be read by the same enclave. An attacker who modifies
the `.pth.sealed` file will cause `InvalidTag` on the next unseal attempt.

**Demo output from `tee/sealed_storage/sealed_storage.py`:**
```
[4] Unsealing with WRONG MRENCLAVE (simulates tampered enclave)...
    Rejected: Sealed data belongs to a different enclave.
    Tampered enclave blocked : ✓
```

### Mitigation: SHA-256 Round Fingerprints

The server computes `fingerprint_weights(weights)` after each round and logs
it to `fl_privacy.json`. Any post-hoc modification of a checkpoint changes
the fingerprint, making tampering detectable during audit.

### Residual Risk

**LOW** (gramine-sgx) / **MEDIUM** (gramine-direct prototype).
In gramine-direct, the host OS can read and modify sealed files.
In gramine-sgx production, the sealing key is hardware-bound.

---

## Threat T4 — Manifest Tampering (Enclave Escape)

### Scenario

An attacker modifies `fl_server_enclave.manifest.template` to add a new
filesystem mount that exposes `/etc/shadow` or the host's private key store
inside the enclave. This would allow the enclave to exfiltrate host secrets.

### Attack Path

```
Attacker modifies manifest:
  [[fs.mounts]]
  path = "/host_secrets"
  uri  = "file:/etc"          ← new malicious mount
    ↓
gramine-manifest recompiles → new MRENCLAVE
    ↓
Clients check MRENCLAVE against expected value
    ↓
MRENCLAVE mismatch → clients refuse to connect
```

### Mitigation: MRENCLAVE Binding

MRENCLAVE is computed as SHA-256(manifest + code). Any change to the manifest
produces a different MRENCLAVE. Clients have the expected MRENCLAVE hardcoded
at build time (written to `tee/attestation/expected_mrenclave.txt`).

**Demo output from `tee/attestation/attestation_integration.py`:**
```
--- Rogue Server Simulation ---
[Client 1][Attestation] Server MRENCLAVE : 000000000000000000000000...
[Client 1][Attestation] Expected         : aba9ce94d51ef83820b219bc...
  Rogue server blocked: [Client 1][Attestation] ✗ MRENCLAVE MISMATCH
  ✓ Rogue server correctly rejected
```

### Residual Risk

**LOW.** Requires both modifying the manifest AND distributing a new expected
MRENCLAVE to all clients — which requires compromising the build pipeline.

---

## Tampering Mitigations Summary

| Threat | Primary Mitigation | Secondary Mitigation | Residual Risk |
|--------|-------------------|---------------------|---------------|
| T1 — Weight tampering in transit | AES-256-GCM auth tag | SHA-256 weight fingerprint | VERY LOW |
| T2 — Gradient poisoning | DP-SGD gradient clipping (norm=1.0) | FedAvg sample-weighted averaging | MEDIUM |
| T3 — Checkpoint tampering | SGX sealed storage (MRENCLAVE-bound) | SHA-256 round fingerprints in fl_privacy.json | LOW (SGX) / MEDIUM (direct) |
| T4 — Manifest tampering | MRENCLAVE binding — clients reject mismatched server | TEE attestation before every FL connection | LOW |

---

## Evidence Files

| Evidence | File |
|---------|------|
| AES-GCM tamper rejection | `crypto/certs/test_crypto.py` Test 3 — PASS |
| Gradient poisoning results | `results/attacks/gradient_poisoning.json` |
| Sealed storage tamper rejection | `tee/sealed_storage/sealed_storage.py` demo |
| Manifest tamper → MRENCLAVE mismatch | `tee/attestation/attestation_integration.py` demo |
| Weight fingerprints per round | `results/fl_rounds/fl_privacy.json` |
