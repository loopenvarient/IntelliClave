import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import get_default_client_csvs  # noqa: E402
from fl_client import start_client  # noqa: E402


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
    parser.add_argument("--dp", action="store_true",
                        help="Enable Differential Privacy via Opacus.")
    parser.add_argument("--epsilon", type=float, default=10.0,
                        help="DP target epsilon. Default=10.0.")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Total FL rounds — must match server --rounds.")
    args = parser.parse_args()

    client_index = int(args.id) - 1
    csv_path = args.csv
    if csv_path is None:
        if client_index < 0 or client_index >= len(default_csvs):
            raise ValueError(
                f"Client ID {args.id} is out of range for the default CSV mapping."
            )
        csv_path = default_csvs[client_index]

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
        use_crypto=args.crypto,
        server_public_pem=server_public_pem,
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        num_fl_rounds=args.rounds,
    )
