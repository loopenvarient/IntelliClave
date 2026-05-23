#!/usr/bin/env python3
"""Exit 0 when SGX hardware is available (for CI gating)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tee"))
from tee_mode import detect_tee_mode, sgx_device_available  # noqa: E402

if __name__ == "__main__":
    info = detect_tee_mode()
    if info["sgx_available"]:
        print(f"SGX available — use {info['mode']}")
        sys.exit(0)
    print(
        "SGX not available — gramine-direct simulation only. "
        "TEE integration tests will be skipped."
    )
    sys.exit(1)
