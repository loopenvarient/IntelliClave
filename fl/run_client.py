import argparse
import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)                          # fixed: removed redundant _HERE insert

from data_utils import get_default_client_csvs     # noqa: E402
from fl_client import start_client                 # noqa: E402

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
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local training epochs per FL round (default: 1).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="DataLoader batch size (default: 32).")
    parser.add_argument("--dp", action="store_true",
                        help="Enable Differential Privacy via Opacus.")
    parser.add_argument("--epsilon", type=float, default=1.5,
                        help="DP target epsilon. Default=1.5.")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Total FL rounds — must match server --rounds.")
    parser.add_argument("--attest", action="store_true",
                        help="Verify server SGX attestation before connecting.")
    parser.add_argument(
        "--norm-config",
        default=None,
        help="Path to global_normalization.json from the FL server run directory "
             "(optional; stats are also applied from server round config).",
    )
    # ── Fixed: use BooleanOptionalAction so --no-save-local-model works ───────
    parser.add_argument(
        "--save-local-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the final local model checkpoint to results/local_models/ "
             "for use in the dashboard Predictions page (default: True).",
    )
    args = parser.parse_args()

    # Verify server attestation before connecting
    if args.attest:
        if _ATTESTATION_AVAILABLE:
            try:
                att = AttestationClient(client_id=args.id)
                att.verify()
            except (FileNotFoundError, SecurityError) as e:
                print(f"[run_client] ATTESTATION FAILED: {e}")
                raise SystemExit(1) from e
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

    # Load public key if crypto requested
    server_public_key_pem = None
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
            server_public_key_pem = f.read()

    # ── Run federated training ─────────────────────────────────────────────────
    trained_model = start_client(
        csv_path=csv_path,
        client_id=args.id,
        server_address=args.server,
        model_type=args.model_type,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_crypto=args.crypto,
        server_public_key_pem=server_public_key_pem,
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        num_fl_rounds=args.rounds,
        norm_config_path=args.norm_config,
        save_local_model=args.save_local_model,    # fixed: forward the flag
    )

    # ── Save local model checkpoint for dashboard Predictions page ─────────────
    if args.save_local_model:
        import torch

        local_model_dir = os.path.join(_ROOT, "results", "local_models")
        os.makedirs(local_model_dir, exist_ok=True)
        save_path = os.path.join(local_model_dir, f"client{args.id}_local_model.pth")

        saved = False

        # Path 1: start_client returns the trained model directly
        if trained_model is not None and hasattr(trained_model, "state_dict"):
            torch.save(trained_model.state_dict(), save_path)
            saved = True
            print(f"[run_client] Local model saved → {save_path}")

        # Path 2: fl_client writes a temp checkpoint we can copy
        if not saved:
            temp_candidates = [
                os.path.join(_HERE, f"client{args.id}_model_temp.pth"),
                os.path.join(_ROOT, f"client{args.id}_model_temp.pth"),
                os.path.join(_HERE, "local_model_temp.pth"),
            ]
            for temp_path in temp_candidates:
                if os.path.exists(temp_path):
                    import shutil
                    shutil.copy2(temp_path, save_path)
                    saved = True
                    print(f"[run_client] Local model saved (from temp) → {save_path}")
                    break

        if not saved:
            print(
                f"[run_client] WARNING: Could not save local model for client {args.id}.\n"
                f"             To enable the Predictions comparison page, modify fl_client.py\n"
                f"             so start_client() returns the trained model, or saves it to:\n"
                f"             {save_path}"
            )