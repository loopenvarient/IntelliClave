# tee/

Trusted Execution Environment layer (Gramine / Intel SGX).

## Purpose

This folder simulates and integrates **TEE attestation** and **sealed storage** for the FL stack. In the prototype (WSL2), Gramine runs in `gramine-direct` mode without real SGX hardware. Production can switch to `gramine-sgx` with no code changes.

## Key files

| Path | Role |
|------|------|
| `attestation/attestation_simulator.py` | Simulated SGX quote + MRENCLAVE verification |
| `attestation/attestation_integration.py` | Hooks for FL server/client attestation flows |
| `attestation/expected_mrenclave.txt` | Reference enclave measurement |
| `sealed_storage/sealed_storage.py` | MRENCLAVE-bound AES-GCM sealed blobs |
| `fl_enclave/*.manifest.template` | Gramine manifests for FL server and clients |
| `benchmarks/run_enclave_benchmarks.py` | Baseline vs TEE execution timing |
| `SGX_STATUS.md` | Environment notes (WSL2 vs hardware SGX) |

## How it connects

- **`fl/run_server.py --attest`** and **`fl/run_client.py --attest`** call attestation integration
- Writes root **`attestation.json`** → dashboard **`/attestation`** endpoint
- Benchmark JSON → **`results/benchmarks_baseline.json`** → dashboard TEE panel
- **`kubernetes/deployments/*-sgx.yaml`** deploy Gramine-SGX variants

## Commands

```bash
python tee/attestation/attestation_simulator.py
python tee/attestation/attestation_integration.py
python tee/sealed_storage/sealed_storage.py
python tee/benchmarks/run_enclave_benchmarks.py
```

## What attestation proves

That the FL server/clients run inside an expected enclave (matching MRENCLAVE), reducing risk of spoofed or tampered participants — complementary to **`crypto/`** (wire encryption) and **`fl/`** (training logic).
