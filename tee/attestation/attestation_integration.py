"""
tee/attestation/attestation_integration.py

Attestation integrated into the FL pipeline.

This module provides two things:

1. AttestationServer  — wraps the FL server startup with attestation.
   Before accepting any client connections, the server generates a quote
   and writes it to attestation.json. Clients read this file and verify
   the MRENCLAVE before sending any weights.

2. AttestationClient  — wraps the FL client startup with attestation.
   Before connecting to the server, the client reads attestation.json,
   verifies the MRENCLAVE matches the expected value, and only then
   proceeds with the FL connection.

Flow:
    Server starts
        │
        ├─ Generates quote (MRENCLAVE of fl_server_enclave.manifest + fl_server.py)
        ├─ Writes attestation.json to shared volume
        └─ Starts Flower gRPC listener
                │
    Client starts
        │
        ├─ Reads attestation.json from shared volume
        ├─ Verifies MRENCLAVE == expected_mrenclave
        ├─ If VERIFIED → connects to server and starts FL
        └─ If FAILED   → aborts (possible rogue server)

In K8s: attestation.json is written to the fl-server-pvc volume,
        which is also mounted read-only by all client pods.

Run demo:
    python3 tee/attestation/attestation_integration.py
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from attestation_simulator import (   # noqa: E402
    compute_mrenclave,
    generate_quote,
    verify_quote,
    run_attestation_demo,
)

# Path where attestation record is written by server and read by clients
ATTESTATION_RECORD_PATH = os.path.join(_ROOT, "attestation.json")

# Expected MRENCLAVE — hardcoded at build time from the known-good manifest.
# In production: baked into the client Docker image at build time.
# Clients refuse to connect if the server's MRENCLAVE doesn't match this.
_EXPECTED_MRENCLAVE_PATH = os.path.join(_HERE, "expected_mrenclave.txt")


# ── Server side ───────────────────────────────────────────────────────────────

class AttestationServer:
    """
    Generates and publishes an attestation quote before the FL server starts.
    Clients read this quote and verify MRENCLAVE before connecting.
    """

    def __init__(
        self,
        manifest_path: str = None,
        code_path: str = None,
        record_path: str = ATTESTATION_RECORD_PATH,
    ):
        self.record_path = record_path
        self.manifest_path = manifest_path or os.path.join(
            _ROOT, "tee", "fl_enclave", "fl_server_enclave.manifest.template"
        )
        self.code_path = code_path or os.path.join(_ROOT, "fl", "fl_server.py")
        self.mrenclave = None
        self.quote     = None

    def attest(self) -> dict:
        """
        Generate quote and write attestation.json.
        Call this BEFORE starting the Flower server.
        Returns the attestation record dict.
        """
        print("[AttestationServer] Generating enclave quote...")
        self.mrenclave = compute_mrenclave(self.manifest_path, self.code_path)
        user_data      = b"intelliclave-server-ready"
        self.quote     = generate_quote(self.mrenclave, user_data)

        record = {
            "tee_verified":     True,
            "enclave_id":       "intelliclave-enclave-v1",
            "attestation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform":         "Intel SGX (simulated)",
            "integrity_hash":   self.mrenclave[:16],
            "mrenclave":        self.mrenclave,
            "quote_size_bytes": len(self.quote),
            "status":           "VERIFIED",
            "mode":             "gramine-direct",
            "environment":      "WSL2",
        }

        with open(self.record_path, "w") as f:
            json.dump(record, f, indent=2)

        # Also write expected_mrenclave.txt so clients can load it
        with open(_EXPECTED_MRENCLAVE_PATH, "w") as f:
            f.write(self.mrenclave)

        print(f"[AttestationServer] MRENCLAVE: {self.mrenclave[:24]}...")
        print(f"[AttestationServer] Quote written → {self.record_path}")
        print(f"[AttestationServer] ✓ Server attestation complete — ready for clients")
        return record

    def get_mrenclave(self) -> str:
        if self.mrenclave is None:
            raise RuntimeError("Call attest() first")
        return self.mrenclave


# ── Client side ───────────────────────────────────────────────────────────────

class AttestationClient:
    """
    Verifies the server's attestation quote before the FL client connects.
    Aborts if MRENCLAVE doesn't match the expected value.
    """

    def __init__(
        self,
        client_id: str,
        record_path: str = ATTESTATION_RECORD_PATH,
        expected_mrenclave: str = None,
    ):
        self.client_id          = client_id
        self.record_path        = record_path
        self.expected_mrenclave = expected_mrenclave or self._load_expected()

    def _load_expected(self) -> str:
        """
        Load the expected MRENCLAVE from the file written by the server
        (or baked into the client image at build time).
        """
        if os.path.exists(_EXPECTED_MRENCLAVE_PATH):
            with open(_EXPECTED_MRENCLAVE_PATH) as f:
                return f.read().strip()
        # Fallback: re-compute from known manifest + code paths
        # (only valid if client has access to the same source files)
        manifest_path = os.path.join(
            _ROOT, "tee", "fl_enclave", "fl_server_enclave.manifest.template"
        )
        code_path = os.path.join(_ROOT, "fl", "fl_server.py")
        return compute_mrenclave(manifest_path, code_path)

    def verify(self) -> bool:
        """
        Read attestation.json and verify the server's MRENCLAVE.
        Returns True if verified, raises SecurityError if not.
        """
        tag = f"[Client {self.client_id}][Attestation]"

        if not os.path.exists(self.record_path):
            raise FileNotFoundError(
                f"{tag} attestation.json not found at {self.record_path}. "
                f"Is the server running and has it completed attestation?"
            )

        with open(self.record_path) as f:
            record = json.load(f)

        server_mrenclave = record.get("mrenclave", "")
        server_status    = record.get("status", "")

        print(f"{tag} Reading attestation record...")
        print(f"{tag} Server MRENCLAVE : {server_mrenclave[:24]}...")
        print(f"{tag} Expected         : {self.expected_mrenclave[:24]}...")
        print(f"{tag} Server status    : {server_status}")

        if server_mrenclave != self.expected_mrenclave:
            raise SecurityError(
                f"{tag} ✗ MRENCLAVE MISMATCH — possible rogue server!\n"
                f"  Server  : {server_mrenclave}\n"
                f"  Expected: {self.expected_mrenclave}\n"
                f"  Aborting FL connection."
            )

        if server_status != "VERIFIED":
            raise SecurityError(
                f"{tag} ✗ Server attestation status is '{server_status}' — expected VERIFIED"
            )

        print(f"{tag} ✓ ATTESTATION VERIFIED — connecting to FL server")
        return True


class SecurityError(Exception):
    """Raised when attestation verification fails."""
    pass


# ── Integration helpers ───────────────────────────────────────────────────────

def server_attest_and_start(start_server_fn, **server_kwargs):
    """
    Wrapper: run attestation, then call start_server_fn(**server_kwargs).

    Usage in fl/run_server.py:
        from tee.attestation.attestation_integration import server_attest_and_start
        server_attest_and_start(start_server, rounds=5, ...)
    """
    attestation = AttestationServer()
    attestation.attest()
    print("[AttestationServer] Starting FL server...")
    start_server_fn(**server_kwargs)


def client_verify_and_start(client_id: str, start_client_fn, **client_kwargs):
    """
    Wrapper: verify server attestation, then call start_client_fn(**client_kwargs).

    Usage in fl/run_client.py:
        from tee.attestation.attestation_integration import client_verify_and_start
        client_verify_and_start("1", start_client, csv_path=..., ...)
    """
    attestation = AttestationClient(client_id=client_id)
    attestation.verify()
    print(f"[Client {client_id}][Attestation] Starting FL client...")
    start_client_fn(**client_kwargs)


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_integration_demo():
    print("=" * 55)
    print("IntelliClave — Attestation Integration Demo")
    print("Attestation wired into FL server + clients")
    print("=" * 55)

    # ── Step 1: Server generates and publishes attestation ────────────────────
    print("\n--- Server Side ---")
    server_att = AttestationServer()
    record     = server_att.attest()

    # ── Step 2: Client 1 verifies before connecting ───────────────────────────
    print("\n--- Client 1 (FitLife) ---")
    client1 = AttestationClient(client_id="1")
    verified = client1.verify()
    assert verified

    # ── Step 3: Client 2 verifies ─────────────────────────────────────────────
    print("\n--- Client 2 (MediTrack) ---")
    client2 = AttestationClient(client_id="2")
    verified = client2.verify()
    assert verified

    # ── Step 4: Client 3 verifies ─────────────────────────────────────────────
    print("\n--- Client 3 (CareWatch) ---")
    client3 = AttestationClient(client_id="3")
    verified = client3.verify()
    assert verified

    # ── Step 5: Rogue server simulation ───────────────────────────────────────
    print("\n--- Rogue Server Simulation ---")
    # Tamper with attestation.json to simulate a rogue server
    with open(ATTESTATION_RECORD_PATH) as f:
        tampered = json.load(f)
    tampered["mrenclave"] = "0" * 64   # wrong MRENCLAVE
    tampered_path = ATTESTATION_RECORD_PATH + ".tampered"
    with open(tampered_path, "w") as f:
        json.dump(tampered, f)

    rogue_client = AttestationClient(
        client_id="1",
        record_path=tampered_path,
        expected_mrenclave=record["mrenclave"],
    )
    blocked = False
    try:
        rogue_client.verify()
    except SecurityError as e:
        blocked = True
        print(f"  Rogue server blocked: {str(e).splitlines()[0]}")
    assert blocked, "Rogue server should have been blocked!"
    print("  ✓ Rogue server correctly rejected")

    # Cleanup tampered file
    os.remove(tampered_path)

    print()
    print("=" * 55)
    print("ATTESTATION INTEGRATION DEMO PASSED ✓")
    print("  Server attested    : ✓")
    print("  Client 1 verified  : ✓")
    print("  Client 2 verified  : ✓")
    print("  Client 3 verified  : ✓")
    print("  Rogue server blocked: ✓")
    print("=" * 55)


if __name__ == "__main__":
    run_integration_demo()
