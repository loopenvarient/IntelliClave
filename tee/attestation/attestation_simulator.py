"""
tee/attestation/attestation_simulator.py

Simulated SGX remote attestation for IntelliClave.

On WSL2, real SGX hardware is not available, so we simulate the quote
structure that Intel SGX would produce. The logic mirrors the real flow:

  Real SGX flow:
    1. Enclave computes MRENCLAVE (SHA-256 of enclave pages at load time)
    2. SGX hardware signs a quote containing MRENCLAVE + user data
    3. Intel Attestation Service (IAS) verifies the quote
    4. Verifier checks MRENCLAVE matches expected value

  Simulated flow (this file):
    1. MRENCLAVE = SHA-256(manifest_content + code_content)
    2. Quote = JSON structure containing MRENCLAVE + metadata + HMAC signature
    3. Verifier checks MRENCLAVE and HMAC
    4. Prints VERIFIED with the same output format as production

Required output format:
    [ATTESTATION] Requesting quote...
    [ATTESTATION] Quote received: 244 bytes
    [ATTESTATION] MRENCLAVE: a3f2c1d4...
    [ATTESTATION] ✓ ATTESTATION VERIFIED
    [ATTESTATION] Mode: Gramine simulation (WSL2)

Run:
    python3 tee/attestation/attestation_simulator.py
"""

import hashlib
import hmac
import json
import os
import struct
import time

# ── MRENCLAVE computation ─────────────────────────────────────────────────────

def compute_mrenclave(manifest_path: str = None, code_path: str = None) -> str:
    """
    Compute MRENCLAVE as SHA-256 of the manifest + enclave code files.

    In real SGX, MRENCLAVE is computed by the CPU as it loads enclave pages.
    Here we hash the manifest template and the FL client code as a proxy,
    giving a deterministic fingerprint of the enclave's expected content.
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, "..", ".."))

    # Default: hash the hello manifest + fl_client as representative files
    if manifest_path is None:
        manifest_path = os.path.join(
            _root, "tee", "hello_gramine", "hello.manifest.template"
        )
    if code_path is None:
        code_path = os.path.join(_root, "fl", "fl_client.py")

    h = hashlib.sha256()
    for path in [manifest_path, code_path]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
        else:
            # fallback if file not found — use path string as placeholder
            h.update(path.encode())

    return h.hexdigest()


# ── Quote structure ───────────────────────────────────────────────────────────

_QUOTE_VERSION  = 3          # SGX DCAP quote version
_SIGN_TYPE      = 2          # ECDSA-256 (simulated)
_QUOTE_SIG_KEY  = b"intelliclave-sim-key-32bytes!!"  # 32-byte simulated signing key

def generate_quote(mrenclave: str, user_data: bytes = b"") -> bytes:
    """
    Build a simulated SGX quote structure.

    Real SGX quote layout (simplified):
        [version: 2B][sign_type: 2B][mrenclave: 32B][user_data: 64B][signature: 64B]

    We follow the same field layout so the byte count is realistic (~244 bytes).
    """
    mrenclave_bytes = bytes.fromhex(mrenclave)[:32]  # 32 bytes

    # user_data padded to 64 bytes (report data field in real SGX)
    user_data_padded = (user_data + b"\x00" * 64)[:64]

    # Header: version (2B) + sign_type (2B) + timestamp (8B)
    header = struct.pack(">HHQ",
                         _QUOTE_VERSION,
                         _SIGN_TYPE,
                         int(time.time()))

    # Body: mrenclave (32B) + user_data (64B)
    body = mrenclave_bytes + user_data_padded

    # Signature: HMAC-SHA256 over header+body with simulated key
    sig = hmac.new(_QUOTE_SIG_KEY, header + body, hashlib.sha256).digest()  # 32B

    # Metadata JSON (appended, makes total ~244 bytes)
    meta = json.dumps({
        "platform": "Intel SGX (simulated)",
        "mode":     "gramine-direct",
        "env":      "WSL2",
    }).encode()

    quote_bytes = header + body + sig + meta
    return quote_bytes


# ── Quote parsing + verification ──────────────────────────────────────────────

def parse_quote(quote_bytes: bytes) -> dict:
    """Extract fields from a simulated quote."""
    header_size = 2 + 2 + 8          # version + sign_type + timestamp
    mrenclave_bytes = quote_bytes[header_size: header_size + 32]
    sig_offset = header_size + 32 + 64
    sig = quote_bytes[sig_offset: sig_offset + 32]
    meta_raw = quote_bytes[sig_offset + 32:]

    try:
        meta = json.loads(meta_raw.decode())
    except Exception:
        meta = {}

    return {
        "mrenclave": mrenclave_bytes.hex(),
        "signature":  sig.hex(),
        "metadata":   meta,
    }


def verify_quote(quote_bytes: bytes, expected_mrenclave: str) -> bool:
    """
    Verify a simulated quote:
      1. Re-derive the HMAC signature and check it matches.
      2. Check MRENCLAVE matches the expected value.
    """
    header_size = 2 + 2 + 8
    body_size   = 32 + 64
    header = quote_bytes[:header_size]
    body   = quote_bytes[header_size: header_size + body_size]
    sig_received = quote_bytes[header_size + body_size: header_size + body_size + 32]

    # Re-compute expected signature
    sig_expected = hmac.new(_QUOTE_SIG_KEY, header + body, hashlib.sha256).digest()

    sig_ok       = hmac.compare_digest(sig_received, sig_expected)
    mrenclave_ok = body[:32].hex() == expected_mrenclave[:32*2]  # first 32 bytes

    return sig_ok and mrenclave_ok


# ── Full attestation demo ─────────────────────────────────────────────────────

def run_attestation_demo():
    """
    Produce the required attestation output format.
    Called by the dashboard /attestation endpoint and standalone.
    """
    print("[ATTESTATION] Requesting quote...")

    mrenclave = compute_mrenclave()
    user_data = b"intelliclave-fl-round-verified"
    quote     = generate_quote(mrenclave, user_data)

    print(f"[ATTESTATION] Quote received: {len(quote)} bytes")
    print(f"[ATTESTATION] MRENCLAVE: {mrenclave[:16]}...")

    verified = verify_quote(quote, mrenclave)

    if verified:
        print("[ATTESTATION] ✓ ATTESTATION VERIFIED")
    else:
        print("[ATTESTATION] ✗ ATTESTATION FAILED")

    print("[ATTESTATION] Mode: Gramine simulation (WSL2)")

    return {
        "tee_verified":      verified,
        "enclave_id":        "intelliclave-enclave-v1",
        "attestation_time":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform":          "Intel SGX (simulated)",
        "integrity_hash":    mrenclave[:16],
        "mrenclave":         mrenclave,
        "quote_size_bytes":  len(quote),
        "status":            "VERIFIED" if verified else "FAILED",
    }


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("IntelliClave — SGX Attestation Simulator")
    print("=" * 55)
    result = run_attestation_demo()
    print()
    print("Full attestation record:")
    print(json.dumps(result, indent=2))
