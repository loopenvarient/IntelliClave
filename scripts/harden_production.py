"""
scripts/harden_production.py

Apply P0/P1 production hardening steps locally or in CI.

Steps:
  1. Seal plaintext checkpoints → .pth.sealed (optional --delete-plaintext)
  2. Scan for remaining plaintext exposure
  3. Run model-inversion assertion (black-box surrogate must stay < 0.6)

Usage:
    python scripts/harden_production.py --seal-checkpoints
    python scripts/harden_production.py --seal-checkpoints --delete-plaintext
    python scripts/harden_production.py --check-only
    python scripts/harden_production.py --verify-inversion
    python scripts/harden_production.py --all
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FL_ROUNDS = os.path.join(_ROOT, "results", "fl_rounds")
_SEALED_DIR = os.path.join(_ROOT, "tee", "sealed_storage")


def _seal_checkpoints(delete_plaintext: bool) -> int:
    sys.path.insert(0, _SEALED_DIR)
    from sealed_storage import seal_model_checkpoint  # noqa: E402

    patterns = [
        os.path.join(_FL_ROUNDS, "global_model_latest.pth"),
        os.path.join(_FL_ROUNDS, "global_model_round_*.pth"),
        os.path.join(_FL_ROUNDS, "run_*", "global_model_latest.pth"),
        os.path.join(_FL_ROUNDS, "run_*", "global_model_round_*.pth"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    sealed_count = 0
    for pth in sorted(set(paths)):
        if not os.path.isfile(pth):
            continue
        if os.path.isfile(pth + ".sealed") and delete_plaintext:
            os.remove(pth)
            print(f"[Harden] Removed plaintext (sealed exists): {os.path.relpath(pth, _ROOT)}")
            sealed_count += 1
            continue
        seal_model_checkpoint(pth, remove_plaintext=delete_plaintext)
        sealed_count += 1
        print(f"[Harden] Sealed: {os.path.relpath(pth, _ROOT)}")

    if sealed_count == 0:
        print("[Harden] No plaintext checkpoints found to seal.")
    else:
        print(f"[Harden] Processed {sealed_count} checkpoint(s).")
    return sealed_count


def _run_check(strict: bool) -> int:
    script = os.path.join(_ROOT, "scripts", "check_checkpoint_exposure.py")
    cmd = [sys.executable, script, "--root", _FL_ROUNDS]
    if strict:
        cmd.append("--strict")
    return subprocess.call(cmd, cwd=_ROOT)


def _run_inversion_assert() -> int:
    script = os.path.join(_ROOT, "security", "attacks", "model_inversion.py")
    env = os.environ.copy()
    env.setdefault("INFERENCE_EPSILON", "4.0")
    cmd = [
        sys.executable, script,
        "--assert",
        "--steps", "200",
        "--surrogate-queries", "500",
        "--surrogate-budget", "400",
        "--surrogate-epochs", "15",
    ]
    print("[Harden] Running model inversion --assert (P0 black-box check)...")
    return subprocess.call(cmd, cwd=_ROOT, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="IntelliClave production hardening.")
    parser.add_argument(
        "--seal-checkpoints", action="store_true",
        help="Seal all plaintext global_model*.pth under results/fl_rounds/.",
    )
    parser.add_argument(
        "--delete-plaintext", action="store_true",
        help="Remove .pth after sealing (same as FL server SEAL_REMOVE_PLAINTEXT).",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only run checkpoint exposure scan (warn mode).",
    )
    parser.add_argument(
        "--strict-check", action="store_true",
        help="Fail if any plaintext checkpoint remains after other steps.",
    )
    parser.add_argument(
        "--verify-inversion", action="store_true",
        help="Run model_inversion.py --assert (P0).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Seal + strict check + inversion assert.",
    )
    args = parser.parse_args()

    if not any([
        args.seal_checkpoints, args.check_only, args.verify_inversion, args.all,
    ]):
        parser.print_help()
        return 0

    rc = 0
    if args.all or args.seal_checkpoints:
        _seal_checkpoints(delete_plaintext=args.delete_plaintext or args.all)

    if args.all or args.check_only or args.seal_checkpoints:
        rc = _run_check(strict=args.strict_check or args.all) or rc

    if args.all or args.verify_inversion:
        rc = _run_inversion_assert() or rc

    return rc


if __name__ == "__main__":
    sys.exit(main())
