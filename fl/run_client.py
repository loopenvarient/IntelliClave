import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import get_default_client_csvs  # noqa: E402
from fl_client import start_client  # noqa: E402

# ── Attestation import ────────────────────────────────────────────────────────
_TEE_DIR = os.path.join(os.path.dirname(__file__), "..", "tee", "attestation")
sys.path.insert(0, os.path.abspath(_TEE_DIR))
try:
    from attestation_integration import AttestationClient, SecurityError  # noqa: E402
    _ATTESTATION_AVAILABLE = True
except ImportError:
    _ATTESTATION_AVAILABLE = False
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    default_csvs = get_default_client_csvs()

    parser = argparse.ArgumentParser(description="Run one IntelliClave Flower client.")
    parser.add_argument("--id", required=True, help="Client ID, for example 1, 2, or 3.")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--crypto", action="store_true",
                        help="Enable AES-256-GCM + RSA weight encryption.")
    parser.add_argument("--pubkey", default=None,
                        help="Path to server_public.pem. "
                             "Defaults to crypto/certs/keys/server_public.pem")
    parser.add_argument("--model-type", default="mlp",
                        choices=["mlp", "resnet-tabular", "transformer-tabular"],
                        help="Model architecture (default: mlp).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3).")
    parser.add_argument("--local-epochs", type=int, default=3,
                        help="Local training epochs per FL round (default: 3).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="DataLoader batch size (default: 32).")
    parser.add_argument("--dp", action="store_true",
                        help="Enable Differential Privacy via Opacus.")
    parser.add_argument("--epsilon", type=float, default=10.0,
                        help="DP target epsilon. Default=10.0.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0,
                        help="DP gradient clipping norm C (default: 2.0, sweep-optimised for eps=10). "
                             "Tune with privacy/clipping_norm_sweep.py before low epsilon.")
    parser.add_argument("--skip-dp-preflight", action="store_true",
                        help="Skip low-ε clipping norm preflight (not recommended).")
    parser.add_argument("--auto-clipping-sweep", action="store_true",
                        help="At ε<5, run a quick clipping norm sweep if calibration fails.")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Total FL rounds — must match server --rounds.")
    parser.add_argument("--attest", action="store_true",
                        help="Verify server SGX attestation before connecting.")
    args = parser.parse_args()

    # Verify server attestation before connecting
    if args.attest:
        if _ATTESTATION_AVAILABLE:
            try:
                att = AttestationClient(client_id=args.id)
                att.verify()
            except (FileNotFoundError, SecurityError) as e:
                print(f"[run_client] ATTESTATION FAILED: {e}")
                raise SystemExit(1)
        else:
            print("[run_client] WARNING: attestation module not found — skipping.")

    client_index = int(args.id) - 1
    csv_path = args.csv
    if csv_path is None:
        if client_index < 0 or client_index >= len(default_csvs):
            raise ValueError(
                f"Client ID {args.id} is out of range for the default CSV mapping."
            )
        csv_path = default_csvs[client_index]

    max_grad_norm = args.max_grad_norm
    if args.dp:
        _privacy_dir = os.path.join(os.path.dirname(__file__), "..", "privacy")
        sys.path.insert(0, os.path.abspath(_privacy_dir))
        from dp_preflight import run_dp_preflight  # noqa: E402

        max_grad_norm = run_dp_preflight(
            csv_path=csv_path,
            target_epsilon=args.epsilon,
            max_grad_norm=args.max_grad_norm,
            local_epochs=args.local_epochs,
            num_fl_rounds=args.rounds,
            batch_size=args.batch_size,
            model_type=args.model_type,
            auto_clipping_sweep=args.auto_clipping_sweep,
            skip=args.skip_dp_preflight,
        )

    # Load public key if crypto requested
    server_public_pem = None
    if args.crypto:
        pubkey_path = args.pubkey or os.path.join(
            os.path.dirname(__file__), "..", "crypto", "certs", "keys", "server_public.pem"
        )
        pubkey_path = os.path.abspath(pubkey_path)
        if not os.path.exists(pubkey_path):
            print(f"[Crypto] ERROR: public key not found at {pubkey_path}")
            print("         Start the server with --crypto first to generate the keypair.")
            raise SystemExit(1)
        with open(pubkey_path, "rb") as f:
            server_public_pem = f.read()

    start_client(
        csv_path=csv_path,
        client_id=args.id,
        server_address=args.server,
        model_type=args.model_type,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_crypto=args.crypto,
        server_public_pem=server_public_pem,
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        max_grad_norm=max_grad_norm,
        num_fl_rounds=args.rounds,
    )
