# IntelliClave — Report Section: Trusted Execution Environment

---

## 1. Overview

The Trusted Execution Environment (TEE) layer provides hardware-level isolation
for the FL aggregation server. It ensures that even a compromised host OS cannot
read the RSA private key, inspect model weights during aggregation, or tamper
with the aggregation logic without detection.

IntelliClave uses **Gramine** as the TEE framework, running on Intel SGX.
The prototype runs in `gramine-direct` mode on WSL2 (no SGX hardware available).
Production deployment uses `gramine-sgx` with zero code changes.

---

## 2. Why a TEE Is Needed

Federated learning protects raw data by keeping it local. However, without a TEE,
the aggregation server is a single point of trust:

| Without TEE | With TEE (gramine-sgx) |
|-------------|----------------------|
| Server admin can read RSA private key | Private key sealed to MRENCLAVE — unreadable by host |
| Server admin can inspect aggregated weights | Aggregation runs inside hardware-isolated enclave |
| Attacker with root access can modify aggregation logic | Any code change changes MRENCLAVE — clients reject |
| No proof that correct aggregation was performed | Attestation report cryptographically binds code to output |

---

## 3. Gramine Architecture

Gramine is a library OS that runs an unmodified application inside an SGX enclave.
The application (Python + FL server) runs unchanged — Gramine intercepts system
calls and routes them through the enclave boundary.

```
┌─────────────────────────────────────────────────────┐
│  SGX Enclave (hardware-isolated)                    │
│  ┌─────────────────────────────────────────────┐    │
│  │  fl/fl_server.py                            │    │
│  │  fl/fl_client.py (per client)               │    │
│  │  crypto/certs/crypto_layer.py               │    │
│  │  tee/attestation/attestation_simulator.py   │    │
│  ├─────────────────────────────────────────────┤    │
│  │  Gramine LibOS (system call interception)   │    │
│  └─────────────────────────────────────────────┘    │
│  Memory Encryption Engine (MEE) — hardware          │
│  Sealing key — derived from MRENCLAVE + CPU         │
└─────────────────────────────────────────────────────┘
         ↕ Encrypted memory bus
┌─────────────────────────────────────────────────────┐
│  Host OS (untrusted in gramine-sgx)                 │
│  Docker / Kubernetes                                │
└─────────────────────────────────────────────────────┘
```

### Manifest System

Gramine uses a manifest file to define the enclave's whitelist — exactly which
files, libraries, and environment variables the enclave is allowed to access.
Any file not listed in the manifest does not exist inside the enclave.

The manifest is compiled into a binary and its hash is included in MRENCLAVE.
Any change to the manifest changes MRENCLAVE, which clients detect via attestation.

---

## 4. MRENCLAVE and Attestation

MRENCLAVE is a SHA-256 hash computed by the SGX hardware as it loads the enclave
pages. It uniquely identifies the exact code and configuration running inside the
enclave.

In the IntelliClave simulation:

```python
# tee/attestation/attestation_simulator.py
def compute_mrenclave(manifest_path, code_path):
    h = hashlib.sha256()
    h.update(open(manifest_path, "rb").read())
    h.update(open(code_path, "rb").read())
    return h.hexdigest()
```

**Current MRENCLAVE:** `aba9ce94d51ef83820b219bc34b89d8a...`

### Attestation Flow

```
Server starts
    │
    ├─ Computes MRENCLAVE from manifest + fl_server.py
    ├─ Generates simulated SGX quote (218 bytes)
    ├─ Writes attestation.json to shared volume
    └─ Starts Flower gRPC listener
            │
Client starts
    │
    ├─ Reads attestation.json
    ├─ Verifies MRENCLAVE == expected value (hardcoded at build time)
    ├─ MATCH  → connects to FL server
    └─ MISMATCH → aborts (possible rogue server or tampered code)
```

### Confirmed Output

```
[AttestationServer] ✓ Server attestation complete — ready for clients
[Client 1][Attestation] ✓ ATTESTATION VERIFIED — connecting to FL server
[Client 2][Attestation] ✓ ATTESTATION VERIFIED — connecting to FL server
[Client 3][Attestation] ✓ ATTESTATION VERIFIED — connecting to FL server
Rogue server (wrong MRENCLAVE): ✗ MRENCLAVE MISMATCH — blocked ✓
```

---

## 5. SGX Sealed Storage

Sealed storage binds sensitive data to the enclave identity. Data encrypted with
the sealing key can only be decrypted by the same enclave (same MRENCLAVE) on the
same CPU. The host OS cannot read sealed data even with root access.

### What Gets Sealed in IntelliClave

| Data | Why Sealed |
|------|-----------|
| `server_private.pem` | RSA private key for weight decryption — must never be readable by host |
| `global_model_latest.pth` | Aggregated model checkpoint — prevents post-hoc backdoor injection |
| `fl_privacy.json` | Epsilon audit log — prevents tampering with privacy records |

### Sealing Key Derivation

```python
# tee/sealed_storage/sealed_storage.py
def _derive_sealing_key(mrenclave: str) -> bytes:
    salt = b"intelliclave-sgx-seal-salt-v1"
    return hmac.new(salt, mrenclave.encode(), hashlib.sha256).digest()
    # In gramine-sgx: replaced by egetkey(SEAL_KEY) from CPU hardware
```

### Confirmed Behaviour

```
[2] Sealing secret (49 bytes)...
    Sealed blob size : 109 bytes

[3] Unsealing with correct MRENCLAVE...
    Recovered : {"private_key": "RSA-2048-SIMULATED", "round": 5}
    Match     : ✓

[4] Unsealing with WRONG MRENCLAVE (tampered enclave)...
    Rejected: Sealed data belongs to a different enclave.
    Tampered enclave blocked : ✓
```

---

## 6. Gramine Manifests

Four manifests were written for IntelliClave:

| Manifest | Purpose | Enclave Size |
|----------|---------|-------------|
| `tee/hello_gramine/hello.manifest.template` | Hello World — confirms Gramine setup | 256 MB |
| `tee/full_stack_test/pytorch_test.manifest.template` | PyTorch forward pass inside Gramine | 1 GB |
| `tee/full_stack_test/fullstack_test.manifest.template` | torch + flwr + opacus full stack | 1 GB |
| `tee/fl_enclave/fl_server_enclave.manifest.template` | Production FL server | 4 GB |
| `tee/fl_enclave/fl_client[1-3]_enclave.manifest.template` | Production FL clients | 2 GB each |

Each manifest explicitly lists every file the enclave is allowed to access.
The server manifest mounts the crypto keys directory; client manifests mount
only their own CSV file (read-only) — not other clients' data.

---

## 7. TEE Overhead Benchmarks

Five operations were measured outside Gramine (baseline) and inside
`gramine-direct` (TEE). Each operation was run 10 times; mean and std reported.

| Operation | Baseline | TEE | Overhead |
|-----------|---------|-----|---------|
| Model inference (batch=100) | 0.41 ± 0.49 ms | 0.56 ± 0.52 ms | +36.0% |
| Training step (batch=32) | 2.62 ± 1.27 ms | 3.54 ± 1.38 ms | +35.3% |
| AES-256-GCM encrypt | 1.45 ± 0.31 ms | 1.96 ± 0.34 ms | +35.2% |
| AES-256-GCM decrypt | 4.78 ± 2.82 ms | 6.45 ± 2.94 ms | +35.1% |
| Model save to disk | 2.76 ± 1.68 ms | 3.73 ± 1.81 ms | +35.1% |
| **Total per round** | **11.02 ms** | **14.90 ms** | **+35.2%** |

**Key finding:** TEE overhead is consistent at ~35% across all operations.
The total per-round overhead is 3.88 ms, which is **0.14% of total FL round
time** (dominant cost is DP-SGD training at ~2–4 seconds per round).
The TEE is not a performance bottleneck.

---

## 8. WSL2 Limitation and Production Path

### Current Limitation

WSL2 does not expose the SGX device (`/dev/sgx_enclave`). Therefore:
- `gramine-sgx` cannot be used in WSL2
- `gramine-direct` is used instead — same code, no hardware isolation
- Attestation is simulated (HMAC-based, not hardware-signed)
- Sealed storage uses software-derived key (not CPU egetkey)

### Production Path

| Step | Action | Code Change |
|------|--------|------------|
| 1 | Deploy on SGX-capable server (Intel Xeon E-series or newer) | None |
| 2 | Replace `gramine-direct` with `gramine-sgx` in all run commands | None |
| 3 | Enable `remote_attestation = "dcap"` in manifest | 1 line |
| 4 | Register with Intel DCAP attestation service | Infrastructure only |

**Zero application code changes required.** Gramine handles the transition
transparently.

---

## 9. Kubernetes SGX Deployment

SGX pods require the Intel SGX device plugin to be installed in the cluster.
The plugin exposes `sgx.intel.com/enclave` and `sgx.intel.com/provision` as
schedulable resources.

```yaml
# kubernetes/deployments/fl-server-sgx.yaml
resources:
  limits:
    sgx.intel.com/enclave:   "1"
    sgx.intel.com/provision: "1"
```

The cold start sequence (`kubernetes/cold_start.sh`):
1. Start minikube with Docker driver
2. Create namespace + crypto Secret
3. Apply all K8s resources
4. Wait for server init container (attestation)
5. Wait for client init containers (attestation verification)
6. FL training begins automatically

---

## 10. Summary

| Milestone | Status | Evidence |
|-----------|--------|---------|
| Gramine installed (v1.9) | ✅ | `tee/SGX_STATUS.md` |
| Hello World in Gramine | ✅ | `tee/hello_gramine/hello.py` |
| PyTorch in Gramine | ✅ | `tee/full_stack_test/pytorch_test.py` |
| Full stack (torch+flwr+opacus) in Gramine | ✅ | `tee/full_stack_test/fullstack_test.py` |
| FL server manifest | ✅ | `tee/fl_enclave/fl_server_enclave.manifest.template` |
| FL client manifests (×3) | ✅ | `tee/fl_enclave/fl_client[1-3]_enclave.manifest.template` |
| Attestation VERIFIED | ✅ | `attestation.json`, `tee/attestation/attestation_integration.py` |
| Sealed storage working | ✅ | `tee/sealed_storage/sealed_storage.py` |
| TEE overhead measured | ✅ | `results/benchmarks/tee_overhead_final.json` |
| K8s SGX pods | ✅ | `kubernetes/deployments/fl-server-sgx.yaml` |
| Cold start script | ✅ | `kubernetes/cold_start.sh` |
