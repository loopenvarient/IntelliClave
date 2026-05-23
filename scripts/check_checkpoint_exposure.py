"""
scripts/check_checkpoint_exposure.py

P1 white-box containment: detect plaintext global model checkpoints on disk.

Plaintext .pth files enable white-box model inversion. Production should only
have .pth.sealed blobs (or load weights from enclave memory).

Usage:
    python scripts/check_checkpoint_exposure.py              # warn only
    python scripts/check_checkpoint_exposure.py --strict     # exit 1 if any plaintext
    python scripts/check_checkpoint_exposure.py --json         # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_DIR = os.path.join(_ROOT, "results", "fl_rounds")


def find_plaintext_checkpoints(root: str) -> list[str]:
    """Return paths to plaintext global_model*.pth (not *.pth.sealed)."""
    found: list[str] = []
    if not os.path.isdir(root):
        return found

    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".pth.sealed"):
                continue
            if not name.endswith(".pth"):
                continue
            if "global_model" not in name:
                continue
            found.append(os.path.join(dirpath, name))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect exposed plaintext FL checkpoints (P1 hardening)."
    )
    parser.add_argument(
        "--root", default=_DEFAULT_DIR,
        help="Directory to scan (default: results/fl_rounds).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if any plaintext global_model*.pth is found.",
    )
    parser.add_argument("--json", action="store_true", help="JSON output.")
    args = parser.parse_args()

    plaintext = find_plaintext_checkpoints(args.root)
    report = {
        "scan_root": os.path.abspath(args.root),
        "plaintext_checkpoints": plaintext,
        "count": len(plaintext),
        "status": "ok" if not plaintext else ("fail" if args.strict else "warn"),
    }

    if args.json:
        print(json.dumps(report, indent=2))
    elif plaintext:
        print(f"[CheckpointExposure] Found {len(plaintext)} plaintext checkpoint(s):")
        for p in plaintext:
            rel = os.path.relpath(p, _ROOT)
            sealed = p + ".sealed"
            has_sealed = os.path.isfile(sealed)
            suffix = " (sealed copy also exists)" if has_sealed else ""
            print(f"  - {rel}{suffix}")
        print(
            "\n  Fix: python scripts/harden_production.py --seal-checkpoints\n"
            "  Or train with FL server SEAL_REMOVE_PLAINTEXT=true"
        )
        if args.strict:
            print("\n  STRICT mode: failing.")
    else:
        print(f"[CheckpointExposure] OK — no plaintext global_model*.pth under {args.root}")

    if plaintext and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
