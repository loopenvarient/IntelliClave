import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fl_server import infer_default_input_dim, start_server  # noqa: E402

# ── Attestation import ────────────────────────────────────────────────────────
_TEE_DIR = os.path.join(os.path.dirname(__file__), "..", "tee", "attestation")
sys.path.insert(0, os.path.abspath(_TEE_DIR))
try:
    from attestation_integration import AttestationServer  # noqa: E402
    _ATTESTATION_AVAILABLE = True
except ImportError:
    _ATTESTATION_AVAILABLE = False
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IntelliClave Flower server.")
    parser.add_argument("--input-dim", type=int, default=infer_default_input_dim())
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--save-dir", default="results/fl_rounds")
    parser.add_argument("--crypto", action="store_true",
                        help="Enable AES-256-GCM + RSA weight encryption.")
    parser.add_argument("--attest", action="store_true",
                        help="Run SGX attestation before starting the FL server.")
    args = parser.parse_args()

    # Run attestation before accepting any client connections
    if args.attest:
        if _ATTESTATION_AVAILABLE:
            att = AttestationServer()
            att.attest()
        else:
            print("[run_server] WARNING: attestation module not found — skipping.")

    start_server(
        input_dim=args.input_dim,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        local_epochs=args.local_epochs,
        server_address=args.address,
        save_dir=args.save_dir,
        use_crypto=args.crypto,
    )
