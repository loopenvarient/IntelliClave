"""
tee/benchmarks/run_enclave_benchmarks.py

Enclave benchmark script.

Measures 5 operations both OUTSIDE Gramine (baseline) and records the
TEE overhead ratios from the pre-recorded baseline.

Operations measured:
  1. Model inference      — forward pass, batch of 100
  2. Training step        — forward + backward + optimizer, batch of 32
  3. AES-256-GCM encrypt  — one round of weight encryption
  4. AES-256-GCM decrypt  — one round of weight decryption
  5. Model save to disk   — torch.save() of HARClassifier

Each operation: 10 runs, records mean ± std in milliseconds.

Output:
  results/benchmarks/enclave_benchmarks.json

Run (outside Gramine — baseline):
  python3 tee/benchmarks/run_enclave_benchmarks.py

Run (inside Gramine — TEE overhead):
  cd tee/benchmarks
  gramine-manifest benchmark.manifest.template benchmark.manifest
  gramine-direct python3 run_enclave_benchmarks.py --tee
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "crypto", "certs"))

from crypto_layer import generate_rsa_keypair, encrypt_weights, decrypt_weights  # noqa

N_RUNS     = 10
BATCH_INF  = 100   # inference batch size
BATCH_TRAIN = 32   # training batch size
INPUT_DIM  = 50
N_CLASSES  = 6
OUT_DIR    = os.path.join(_ROOT, "results", "benchmarks")
OUT_FILE   = os.path.join(OUT_DIR, "enclave_benchmarks.json")


# ── Model ─────────────────────────────────────────────────────────────────────
class HARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )
    def forward(self, x):
        return self.net(x)


def get_weights(model):
    return [v.cpu().numpy().copy() for v in model.state_dict().values()]


# ── Timer ─────────────────────────────────────────────────────────────────────
def measure(fn, n=N_RUNS):
    """Run fn() n times, return (mean_ms, std_ms)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 3), round(float(np.std(times)), 3)


# ── Benchmark functions ───────────────────────────────────────────────────────
def bench_inference(model):
    x = torch.randn(BATCH_INF, INPUT_DIM)
    model.eval()
    def fn():
        with torch.no_grad():
            model(x)
    return measure(fn)


def bench_training_step(model, optimizer, criterion):
    x = torch.randn(BATCH_TRAIN, INPUT_DIM)
    y = torch.randint(0, N_CLASSES, (BATCH_TRAIN,))
    def fn():
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    return measure(fn)


def bench_aes_encrypt(weights, public_key):
    def fn():
        encrypt_weights(weights, public_key)
    return measure(fn)


def bench_aes_decrypt(payload, private_key):
    def fn():
        decrypt_weights(payload, private_key)
    return measure(fn)


def bench_model_save(model):
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "bench_model.pth")
    def fn():
        torch.save(model.state_dict(), tmp)
    return measure(fn)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tee", action="store_true",
                        help="Mark results as TEE mode (running inside Gramine)")
    args = parser.parse_args()

    mode = "tee" if args.tee else "baseline"
    print("=" * 55)
    print(f"IntelliClave — Enclave Benchmarks ({mode.upper()})")
    print(f"  {N_RUNS} runs per operation | batch_inf={BATCH_INF} | batch_train={BATCH_TRAIN}")
    print("=" * 55)

    model     = HARClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    priv_key, pub_key = generate_rsa_keypair()
    weights   = get_weights(model)
    payload   = encrypt_weights(weights, pub_key)

    results = {}

    print("\n[1] Model inference (batch=100)...", end=" ", flush=True)
    mean, std = bench_inference(model)
    results["model_inference"] = {"mean_ms": mean, "std_ms": std}
    print(f"{mean:.2f} ± {std:.2f} ms")

    print("[2] Training step (batch=32)...", end=" ", flush=True)
    mean, std = bench_training_step(model, optimizer, criterion)
    results["training_step"] = {"mean_ms": mean, "std_ms": std}
    print(f"{mean:.2f} ± {std:.2f} ms")

    print("[3] AES-256-GCM encrypt weights...", end=" ", flush=True)
    mean, std = bench_aes_encrypt(weights, pub_key)
    results["aes_encrypt"] = {"mean_ms": mean, "std_ms": std}
    print(f"{mean:.2f} ± {std:.2f} ms")

    print("[4] AES-256-GCM decrypt weights...", end=" ", flush=True)
    mean, std = bench_aes_decrypt(payload, priv_key)
    results["aes_decrypt"] = {"mean_ms": mean, "std_ms": std}
    print(f"{mean:.2f} ± {std:.2f} ms")

    print("[5] Model save to disk...", end=" ", flush=True)
    mean, std = bench_model_save(model)
    results["model_save"] = {"mean_ms": mean, "std_ms": std}
    print(f"{mean:.2f} ± {std:.2f} ms")

    # ── Load baseline for overhead calculation ────────────────────────────────
    baseline_path = os.path.join(_ROOT, "results", "benchmarks_baseline.json")
    overhead_note = ""
    if mode == "tee" and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        overhead_note = "TEE overhead vs baseline recorded in benchmarks_baseline.json"

    output = {
        "mode":       mode,
        "n_runs":     N_RUNS,
        "batch_inf":  BATCH_INF,
        "batch_train": BATCH_TRAIN,
        "results":    results,
        "note":       overhead_note,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = OUT_FILE.replace(".json", f"_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 55)
    print(f"Saved → {out_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
