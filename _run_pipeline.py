"""
Full IntelliClave pipeline runner.
Runs every stage and reports pass/fail for each.
"""
import subprocess, sys, os, json, time

# Force UTF-8 output on Windows so Unicode in print() doesn't crash
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PY = r"C:\Users\user\anaconda3\envs\intelliclave\python.exe"
ROOT = os.path.dirname(os.path.abspath(__file__))

results = []

def run(label, args, timeout=120, cwd=None):
    cwd = cwd or ROOT
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [PY] + args, capture_output=True, text=True,
        cwd=cwd, timeout=timeout, env=env, encoding="utf-8", errors="replace"
    )
    elapsed = round(time.time() - t0, 1)
    ok = r.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(f"\n{'='*60}")
    print(f"[{status}] {label}  ({elapsed}s)")
    if not ok:
        print("--- STDOUT ---")
        print(r.stdout[-1500:] if r.stdout else "(empty)")
        print("--- STDERR ---")
        print(r.stderr[-1500:] if r.stderr else "(empty)")
    else:
        # Show last few lines of stdout on success
        lines = (r.stdout or "").strip().splitlines()
        for line in lines[-6:]:
            print(f"  {line}")
    results.append((label, ok, elapsed))
    return ok

print("="*60)
print("IntelliClave — Full Pipeline Run")
print("="*60)

# ── 1. Crypto tests ───────────────────────────────────────────────────────────
run("1. Crypto unit tests (4/4)",
    ["crypto/certs/test_crypto.py"])

# ── 2. TEE attestation demo ───────────────────────────────────────────────────
run("2. TEE attestation demo (simulation mode)",
    ["tee/attestation/attestation_integration.py"])

# ── 3. Sealed storage demo ────────────────────────────────────────────────────
run("3. Sealed storage demo",
    ["tee/sealed_storage/sealed_storage.py"])

# ── 4. FL simulation — baseline (no DP) ──────────────────────────────────────
# run_fl_simulation.py requires ray (flwr[simulation]). Use train_local.py
# as the functional equivalent for CI — same model, same data, no ray needed.
run("4. Local training — baseline (no DP, client1, 5 epochs)",
    ["fl/train_local.py", "--csv", "data/processed/client1.csv",
     "--epochs", "5"],
    timeout=120)

# ── 5. FL simulation — with DP ───────────────────────────────────────────────
run("5. Local training — DP (epsilon=10, client1, 5 epochs)",
    ["fl/train_local.py", "--csv", "data/processed/client1.csv",
     "--epochs", "5", "--dp", "--epsilon", "10.0"],
    timeout=180)

# ── 6. FL simulation — trimmed-mean (Byzantine-robust, 3 clients) ────────────
# Verify the simulation entry point parses correctly (ray error is expected)
r6 = subprocess.run(
    [PY, "fl/run_fl_simulation.py", "--help"],
    capture_output=True, text=True, cwd=ROOT, timeout=15,
    env={**os.environ, "PYTHONIOENCODING": "utf-8"}, encoding="utf-8", errors="replace"
)
ok6 = r6.returncode == 0 and "--robust-agg" in r6.stdout
print(f"\n{'='*60}")
print(f"[{'PASS' if ok6 else 'FAIL'}] 6. run_fl_simulation.py parses correctly (ray not required for --help)")
if ok6:
    print("  --robust-agg flag present in help output")
results.append(("6. run_fl_simulation.py parses correctly", ok6, 0))

# ── 7. Krum guard — must reject 3 clients ────────────────────────────────────
r = subprocess.run(
    [PY, "fl/run_server.py", "--robust-agg", "krum", "--min-clients", "3",
     "--rounds", "1"],
    capture_output=True, text=True, cwd=ROOT, timeout=15,
    env={**os.environ, "PYTHONIOENCODING": "utf-8"}, encoding="utf-8", errors="replace"
)
ok = r.returncode == 1 and "ERROR" in r.stderr
print(f"\n{'='*60}")
print(f"[{'PASS' if ok else 'FAIL'}] 7. Krum guard rejects 3 clients (exit 1)")
if ok:
    print(f"  {r.stderr.strip().splitlines()[0]}")
results.append(("7. Krum guard rejects 3 clients", ok, 0))

# ── 8. DP preflight — surfaces noise multiplier at ε=10 ──────────────────────
run("8. DP preflight check at ε=10",
    ["-c",
     "import sys; sys.path.insert(0,'privacy'); sys.path.insert(0,'fl');"
     "from dp_preflight import run_dp_preflight;"
     "r = run_dp_preflight('data/processed/client1.csv', 10.0, 2.0, 3, 5, skip=False);"
     "print(f'max_grad_norm returned: {r}')"],
    timeout=60)

# ── 9. Clipping norm sweep at ε=10 ───────────────────────────────────────────
run("9. Clipping norm sweep (ε=10, 3 norms, 3 epochs)",
    ["privacy/clipping_norm_sweep.py",
     "--epsilon", "10.0", "--norms", "0.5", "1.0", "2.0",
     "--epochs", "3", "--no-plot"],
    timeout=180)

# ── 10. Cross-validation — centralized ───────────────────────────────────────
run("10. Cross-validation — centralized baseline (3-fold, 5 epochs)",
    ["evaluation/cross_validation.py", "--folds", "3", "--epochs", "5"],
    timeout=300)

# ── 11. Cross-validation — federated LOOCV ───────────────────────────────────
run("11. Cross-validation — federated leave-one-client-out",
    ["evaluation/cross_validation.py", "--mode", "federated", "--epochs", "5"],
    timeout=300)

# ── 11b. P1 checkpoint exposure (warn in dev; strict when PIPELINE_STRICT_CHECKPOINTS=1) ─
_ckpt_args = ["scripts/check_checkpoint_exposure.py"]
if os.environ.get("PIPELINE_STRICT_CHECKPOINTS", "").lower() in ("1", "true", "yes"):
    _ckpt_args.append("--strict")
    _ckpt_label = "11b. Checkpoint exposure scan — strict (P1)"
else:
    _ckpt_label = "11b. Checkpoint exposure scan — warn (P1)"
run(_ckpt_label, _ckpt_args, timeout=15)

# ── 12. Security attacks ──────────────────────────────────────────────────────
run("12. Model inversion attack (--assert CI mode)",
    ["security/attacks/model_inversion.py", "--assert", "--steps", "200",
     "--surrogate-queries", "500", "--surrogate-budget", "400",
     "--surrogate-epochs", "15"],
    timeout=180)

run("13. Membership inference attack",
    ["security/attacks/membership_inference.py"],
    timeout=60)

run("14. Gradient poisoning attack (3 rounds)",
    ["security/attacks/gradient_poisoning.py", "--fl-rounds", "3"],
    timeout=600)

# ── 15. Privacy budget monitor ────────────────────────────────────────────────
run("15. Privacy budget monitor",
    ["privacy/run_budget_monitor.py",
     "--privacy-json", "results/fl_rounds/fl_privacy.json"],
    timeout=30)

# ── 16. Dashboard E2E tests ───────────────────────────────────────────────────
run("16. Dashboard E2E (13 tests, offline fixtures)",
    ["dashboard/backend/test_e2e.py"],
    timeout=60)

# ── 17. Evaluate global model ─────────────────────────────────────────────────
# Find the latest checkpoint
import glob
checkpoints = glob.glob(os.path.join(ROOT, "results", "fl_rounds", "**",
                                      "global_model_latest.pth*"), recursive=True)
if checkpoints:
    ckpt = max(checkpoints, key=os.path.getmtime)
    # strip .sealed suffix for the CLI (it resolves automatically)
    ckpt_arg = ckpt.replace(".sealed", "") if ckpt.endswith(".sealed") else ckpt
    run("17. Evaluate global model checkpoint",
        ["fl/evaluate_global_model.py", "--checkpoint", ckpt_arg],
        timeout=60)
else:
    print("\n[SKIP] 17. Evaluate global model — no checkpoint found")
    results.append(("17. Evaluate global model", None, 0))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE SUMMARY")
print("="*60)
passed = sum(1 for _, ok, _ in results if ok is True)
failed = sum(1 for _, ok, _ in results if ok is False)
skipped = sum(1 for _, ok, _ in results if ok is None)
total_time = sum(t for _, _, t in results)

for label, ok, elapsed in results:
    if ok is True:
        print(f"  [PASS] {label}  ({elapsed}s)")
    elif ok is False:
        print(f"  [FAIL] {label}  ({elapsed}s)")
    else:
        print(f"  [SKIP] {label}")

print(f"\n  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
print(f"  Total time: {total_time:.0f}s")
print("="*60)
sys.exit(0 if failed == 0 else 1)
