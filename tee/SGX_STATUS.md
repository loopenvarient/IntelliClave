# IntelliClave — TEE / SGX Status

## Environment

| Item | Value |
|------|-------|
| OS | Ubuntu 22.04 (WSL2 on Windows 11) |
| Kernel | WSL2 Linux kernel |
| Gramine version | 1.9 |
| Mode | gramine-direct |
| SGX hardware | Not present in WSL2 |
| Docker | Working (hello-world confirmed) |

---

## SGX Hardware Check

```bash
grep -m1 flags /proc/cpuinfo | grep -o sgx   # → (empty — no SGX in WSL2)
ls /dev/sgx* 2>/dev/null                      # → (empty — no SGX device)
uname -r                                      # → WSL2 kernel confirmed
```

Result: SGX hardware is **not available** in WSL2. This is expected and documented.

---

## Why gramine-direct Is Acceptable

Gramine operates in two modes: `gramine-sgx` uses real Intel SGX hardware to create
a hardware-sealed enclave with remote attestation backed by Intel's attestation service.
`gramine-direct` runs the same enclave application code and manifest system on the host
OS without hardware isolation. For this research prototype, `gramine-direct` is fully
acceptable because it validates the entire software stack — manifest correctness, library
mounts, enclave application logic, and the attestation simulator — without requiring
SGX-capable hardware. All code, manifests, and attestation logic are identical between
the two modes. The only difference is the trust boundary: in `gramine-direct` the OS is
trusted, whereas in `gramine-sgx` the hardware enforces isolation.

**For production deployment:** Replace `gramine-direct` with `gramine-sgx` on an
SGX-capable server — zero code changes required.

---

## Confirmed Milestones

| Milestone | Status | Evidence |
|-----------|--------|---------|
| Docker working | ✅ | `docker run hello-world` → "Hello from Docker!" |
| Gramine installed | ✅ | `gramine-direct --version` → Gramine 1.9 |
| Hello World in Gramine | ✅ | `tee/hello_gramine/hello.py` runs under gramine-direct |
| PyTorch in Gramine | ✅ | `tee/full_stack_test/pytorch_test.py` — forward pass confirmed |
| Full stack in Gramine | ✅ | torch + flwr + opacus all import inside gramine-direct |
| Baseline benchmarks | ✅ | `results/benchmarks_baseline.json` — never re-run |
| Minikube running | ✅ | `minikube start --driver=docker` confirmed |
| K8s YAMLs valid | ✅ | `kubectl apply --dry-run=client` passed |
| Crypto tests passing | ✅ | All 4 tests pass — see `crypto/certs/test_crypto.py` |
| Attestation VERIFIED | ✅ | `tee/attestation/attestation_simulator.py` prints VERIFIED |

---

## Build + Run Commands

### Hello World
```bash
cd tee/hello_gramine
gramine-manifest hello.manifest.template hello.manifest
gramine-direct python3 hello.py
```

### PyTorch Test
```bash
cd tee/full_stack_test
gramine-manifest pytorch_test.manifest.template pytorch_test.manifest
gramine-direct python3 pytorch_test.py
```

### Full Stack Test (torch + flwr + opacus)
```bash
cd tee/full_stack_test
gramine-manifest fullstack_test.manifest.template fullstack_test.manifest
gramine-direct python3 fullstack_test.py
```

### Attestation Demo
```bash
python3 tee/attestation/attestation_simulator.py
```
