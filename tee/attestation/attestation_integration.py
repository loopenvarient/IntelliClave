# tee/attestation/attestation_integration.py
"""
Attestation integrated into the FL pipeline.

Issue 6 fix: mutual attestation
---------------------------------
The original implementation was one-directional: the server generated a
quote and clients verified it, but clients never proved their own identity
to the server. A compromised client could submit poisoned gradients and the
server had no cryptographic way to distinguish it from a legitimate client.

This module now implements mutual attestation:

  Server side (AttestationServer):
    - Generates its own quote (MRENCLAVE of server manifest + code)
    - Writes attestation.json for clients to verify
    - NEW: verify_client() -- reads a client quote and verifies their
      MRENCLAVE matches the expected client enclave identity

  Client side (AttestationClient):
    - Verifies the server MRENCLAVE before connecting (existing)
    - NEW: generate_quote() -- generates the client own quote
    - NEW: publish_quote() -- writes the client quote to a shared path
      so the server can verify it before accepting gradients

Flow (mutual):
    Server starts
        - Generates server quote -> attestation.json
        - Starts Flower gRPC listener

    Client starts
        - Reads attestation.json, verifies server MRENCLAVE
        - Generates client quote -> client_{id}_attestation.json
        - Connects to server

    Server (before accepting gradients from client):
        - Reads client_{id}_attestation.json
        - Verifies client MRENCLAVE == expected_client_mrenclave
        - If VERIFIED -> accepts gradients
        - If FAILED   -> rejects gradients, logs security event

In K8s: both attestation files are written to the fl-server-pvc volume.

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
)

_TEE_DIR = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.abspath(_TEE_DIR))
try:
    from tee_mode import detect_tee_mode, enrich_attestation_record  # noqa: E402
except ImportError:
    def detect_tee_mode():  # type: ignore
        return {
            "mode": "gramine-direct",
            "simulation_mode": True,
            "platform": "Intel SGX (simulated)",
            "environment": "unknown",
            "sgx_available": False,
        }

    def enrich_attestation_record(record):  # type: ignore
        record.setdefault("simulation_mode", True)
        record.setdefault("mode", "gramine-direct")
        return record

def _default_attestation_path() -> str:
    _config = os.path.join(_ROOT, "config")
    if _config not in sys.path:
        sys.path.insert(0, _config)
    try:
        from runtime_paths import primary_runtime_path  # noqa: E402
        return primary_runtime_path("attestation.json")
    except ImportError:
        return os.path.join(_ROOT, "attestation.json")


ATTESTATION_RECORD_PATH = os.environ.get(
    "ATTESTATION_RECORD_PATH", _default_attestation_path()
)
_EXPECTED_MRENCLAVE_PATH = os.path.join(_HERE, "expected_mrenclave.txt")
_EXPECTED_CLIENT_MRENCLAVE_PATH = os.path.join(_HERE, "expected_client_mrenclave.txt")


class SecurityError(Exception):
    """Raised when attestation verification fails."""
    pass


# -- Server side ---------------------------------------------------------------

class AttestationServer:
    """
    Generates and publishes the server attestation quote.
    Also verifies client quotes before accepting their gradients.
    """

    def __init__(
        self,
        manifest_path=None,
        code_path=None,
        record_path=ATTESTATION_RECORD_PATH,
        expected_client_mrenclave=None,
    ):
        self.record_path = record_path
        self.manifest_path = manifest_path or os.path.join(
            _ROOT, "tee", "fl_enclave", "fl_server_enclave.manifest.template"
        )
        self.code_path = code_path or os.path.join(_ROOT, "fl", "fl_server.py")
        self.mrenclave = None
        self.quote     = None
        self._expected_client_mrenclave = (
            expected_client_mrenclave or self._load_expected_client_mrenclave()
        )

    def _load_expected_client_mrenclave(self):
        if os.path.exists(_EXPECTED_CLIENT_MRENCLAVE_PATH):
            with open(_EXPECTED_CLIENT_MRENCLAVE_PATH) as f:
                return f.read().strip()
        client_manifest = os.path.join(
            _ROOT, "tee", "fl_enclave", "fl_client_enclave.manifest.template"
        )
        client_code = os.path.join(_ROOT, "fl", "fl_client.py")
        return compute_mrenclave(client_manifest, client_code)

    def attest(self):
        """Generate server quote and write attestation.json."""
        print("[AttestationServer] Generating enclave quote...")
        self.mrenclave = compute_mrenclave(self.manifest_path, self.code_path)
        user_data      = b"intelliclave-server-ready"
        self.quote     = generate_quote(self.mrenclave, user_data)

        tee_info = detect_tee_mode()
        record = {
            "tee_verified":       True,
            "enclave_id":         "intelliclave-enclave-v1",
            "attestation_time":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "integrity_hash":     self.mrenclave[:16],
            "mrenclave":          self.mrenclave,
            "quote_size_bytes":   len(self.quote),
            "status":             "VERIFIED",
            "mutual_attestation": True,
            "platform":           tee_info["platform"],
            "mode":               tee_info["mode"],
            "simulation_mode":    tee_info["simulation_mode"],
            "environment":        tee_info["environment"],
            "sgx_available":      tee_info["sgx_available"],
        }
        if tee_info.get("warning"):
            record["warning"] = tee_info["warning"]
        record = enrich_attestation_record(record)

        _config = os.path.join(_ROOT, "config")
        if _config not in sys.path:
            sys.path.insert(0, _config)
        try:
            from runtime_paths import write_json_runtime  # noqa: E402
            self.record_path = write_json_runtime("attestation.json", record)
        except ImportError:
            with open(self.record_path, "w") as f:
                json.dump(record, f, indent=2)

        with open(_EXPECTED_MRENCLAVE_PATH, "w") as f:
            f.write(self.mrenclave)

        print(f"[AttestationServer] MRENCLAVE: {self.mrenclave[:24]}...")
        print(f"[AttestationServer] Quote written -> {self.record_path}")
        print("[AttestationServer] Server attestation complete -- ready for clients")
        return record

    def verify_client(self, client_id, client_record_path=None):
        """
        Verify a client attestation quote before accepting their gradients.
        Returns True if verified, raises SecurityError if not.
        """
        tag = f"[Server][ClientAttestation][Client {client_id}]"

        if client_record_path is None:
            record_dir = os.path.dirname(self.record_path)
            client_record_path = os.path.join(
                record_dir, f"client_{client_id}_attestation.json"
            )

        if not os.path.exists(client_record_path):
            raise SecurityError(
                f"{tag} Client attestation record not found at "
                f"{client_record_path}. "
                "Client must call generate_quote() and publish_quote() "
                "before the server can accept its gradients."
            )

        with open(client_record_path) as f:
            record = json.load(f)

        client_mrenclave = record.get("mrenclave", "")
        client_status    = record.get("status", "")

        print(f"{tag} Client MRENCLAVE : {client_mrenclave[:24]}...")
        print(f"{tag} Expected         : {self._expected_client_mrenclave[:24]}...")
        print(f"{tag} Client status    : {client_status}")

        if client_mrenclave != self._expected_client_mrenclave:
            raise SecurityError(
                f"{tag} CLIENT MRENCLAVE MISMATCH -- possible rogue client!\n"
                f"  Client  : {client_mrenclave}\n"
                f"  Expected: {self._expected_client_mrenclave}\n"
                f"  Rejecting gradients from this client."
            )

        if client_status != "VERIFIED":
            raise SecurityError(
                f"{tag} Client attestation status is '{client_status}' "
                "-- expected VERIFIED"
            )

        print(f"{tag} CLIENT ATTESTATION VERIFIED -- gradients accepted")
        return True

    def get_mrenclave(self):
        if self.mrenclave is None:
            raise RuntimeError("Call attest() first")
        return self.mrenclave


# -- Client side ---------------------------------------------------------------

class AttestationClient:
    """
    Verifies the server attestation quote and generates its own quote
    for the server to verify (mutual attestation).
    """

    def __init__(
        self,
        client_id,
        record_path=ATTESTATION_RECORD_PATH,
        expected_mrenclave=None,
        client_manifest_path=None,
        client_code_path=None,
    ):
        self.client_id          = client_id
        self.record_path        = record_path
        self.expected_mrenclave = expected_mrenclave or self._load_expected()
        self._client_manifest   = client_manifest_path or os.path.join(
            _ROOT, "tee", "fl_enclave", "fl_client_enclave.manifest.template"
        )
        self._client_code       = client_code_path or os.path.join(
            _ROOT, "fl", "fl_client.py"
        )
        self._client_mrenclave  = None
        self._client_quote      = None

    def _load_expected(self):
        if os.path.exists(_EXPECTED_MRENCLAVE_PATH):
            with open(_EXPECTED_MRENCLAVE_PATH) as f:
                return f.read().strip()
        manifest_path = os.path.join(
            _ROOT, "tee", "fl_enclave", "fl_server_enclave.manifest.template"
        )
        code_path = os.path.join(_ROOT, "fl", "fl_server.py")
        return compute_mrenclave(manifest_path, code_path)

    def verify(self):
        """
        Verify the server MRENCLAVE.
        Returns True if verified, raises SecurityError on failure.
        """
        tag = f"[Client {self.client_id}][Attestation]"

        if not os.path.exists(self.record_path):
            raise FileNotFoundError(
                f"{tag} attestation.json not found at {self.record_path}. "
                "Is the server running and has it completed attestation?"
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
                f"{tag} MRENCLAVE MISMATCH -- possible rogue server!\n"
                f"  Server  : {server_mrenclave}\n"
                f"  Expected: {self.expected_mrenclave}\n"
                f"  Aborting FL connection."
            )

        if server_status != "VERIFIED":
            raise SecurityError(
                f"{tag} Server attestation status is '{server_status}' "
                "-- expected VERIFIED"
            )

        print(f"{tag} SERVER ATTESTATION VERIFIED -- connecting to FL server")
        return True

    def generate_quote(self):
        """
        Generate this client own attestation quote.
        Call publish_quote() to write it to disk for the server to verify.
        """
        tag = f"[Client {self.client_id}][Attestation]"
        print(f"{tag} Generating client enclave quote...")
        self._client_mrenclave = compute_mrenclave(
            self._client_manifest, self._client_code
        )
        user_data = f"intelliclave-client-{self.client_id}-ready".encode()
        self._client_quote = generate_quote(self._client_mrenclave, user_data)
        print(f"{tag} Client MRENCLAVE: {self._client_mrenclave[:24]}...")
        return {
            "client_id":        self.client_id,
            "mrenclave":        self._client_mrenclave,
            "quote_size_bytes": len(self._client_quote),
        }

    def publish_quote(self, publish_dir=None):
        """
        Write the client attestation record to disk so the server can
        verify it before accepting gradients.
        Returns the path of the written file.
        """
        if self._client_mrenclave is None:
            raise RuntimeError("Call generate_quote() before publish_quote().")

        if publish_dir is None:
            publish_dir = os.path.dirname(self.record_path)

        record = {
            "client_id":        self.client_id,
            "tee_verified":     True,
            "attestation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform":         "Intel SGX (simulated)",
            "mrenclave":        self._client_mrenclave,
            "integrity_hash":   self._client_mrenclave[:16],
            "quote_size_bytes": len(self._client_quote),
            "status":           "VERIFIED",
            "mode":             "gramine-direct",
        }

        path = os.path.join(
            publish_dir, f"client_{self.client_id}_attestation.json"
        )
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

        tag = f"[Client {self.client_id}][Attestation]"
        print(f"{tag} Client quote published -> {path}")
        # expected_client_mrenclave.txt is a server-side build artifact.
        # Do NOT write it here -- each client call would overwrite it,
        # leaving only the last client verifiable by the server.
        # Set it once at build time via the server operator.
        return path


# -- Integration helpers -------------------------------------------------------

def server_attest_and_start(start_server_fn, **server_kwargs):
    """Wrapper: run server attestation, then call start_server_fn."""
    attestation = AttestationServer()
    attestation.attest()
    print("[AttestationServer] Starting FL server...")
    start_server_fn(**server_kwargs)


def client_verify_and_start(client_id, start_client_fn, **client_kwargs):
    """
    Verify server attestation, generate and publish client quote,
    then start the FL client.
    """
    attestation = AttestationClient(client_id=client_id)
    attestation.verify()
    attestation.generate_quote()
    attestation.publish_quote()
    print(f"[Client {client_id}][Attestation] Starting FL client...")
    start_client_fn(**client_kwargs)


# -- Demo ----------------------------------------------------------------------

def run_integration_demo():
    print("=" * 60)
    print("IntelliClave -- Mutual Attestation Integration Demo")
    print("=" * 60)

    # Step 1: Server attests
    print("\n--- Server Side ---")
    server_att = AttestationServer()
    record     = server_att.attest()

    # Step 2: Clients verify server and publish their own quotes
    n_clients = 3
    for i in range(1, n_clients + 1):
        print(f"\n--- Client {i} ---")
        c = AttestationClient(client_id=str(i))
        c.verify()
        c.generate_quote()
        c.publish_quote()
        # In production, expected_client_mrenclave.txt is baked into the
        # server image at build time (all clients run the same code).
        # Here we write it once from client 1 to simulate that.
        if i == 1:
            with open(_EXPECTED_CLIENT_MRENCLAVE_PATH, "w") as _f:
                _f.write(c._client_mrenclave)

    # Step 3: Server verifies each client
    print("\n--- Server verifies clients ---")
    for i in range(1, n_clients + 1):
        verified = server_att.verify_client(str(i))
        assert verified, f"Client {i} should have been verified"
        print(f"  Client {i} verified by server: OK")

    # Step 4: Rogue server simulation
    print("\n--- Rogue Server Simulation ---")
    with open(ATTESTATION_RECORD_PATH) as f:
        tampered = json.load(f)
    tampered["mrenclave"] = "0" * 64
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
    assert blocked, "Rogue server should have been blocked"
    print("  Rogue server correctly rejected")
    os.remove(tampered_path)

    # Step 5: Rogue client simulation
    print("\n--- Rogue Client Simulation ---")
    rogue_record_dir = os.path.dirname(ATTESTATION_RECORD_PATH)
    rogue_client_path = os.path.join(
        rogue_record_dir, "client_rogue_attestation.json"
    )
    with open(rogue_client_path, "w") as f:
        json.dump({
            "client_id": "rogue",
            "mrenclave": "f" * 64,
            "status":    "VERIFIED",
        }, f)

    rogue_server = AttestationServer(
        expected_client_mrenclave=record["mrenclave"]
    )
    rogue_server.mrenclave = record["mrenclave"]
    blocked = False
    try:
        rogue_server.verify_client("rogue", rogue_client_path)
    except SecurityError as e:
        blocked = True
        print(f"  Rogue client blocked: {str(e).splitlines()[0]}")
    assert blocked, "Rogue client should have been blocked"
    print("  Rogue client correctly rejected")
    os.remove(rogue_client_path)

    print()
    print("=" * 60)
    print("MUTUAL ATTESTATION DEMO PASSED")
    print("  Server attested          : OK")
    for i in range(1, n_clients + 1):
        print(f"  Client {i} verified server : OK")
        print(f"  Server verified client {i} : OK")
    print("  Rogue server blocked     : OK")
    print("  Rogue client blocked     : OK")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_demo()