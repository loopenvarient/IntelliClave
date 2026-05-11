import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fl_server import (  # noqa: E402
    infer_default_input_dim,
    infer_default_num_classes,
    make_timestamped_save_dir,
    monitor_client_distributions,
    start_server,
)

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
    parser.add_argument("--num-classes", type=int, default=0,
                        help="Number of output classes. Inferred from data if not set.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--save-dir", default="",
                        help="Output directory. Auto-generates a timestamped path if not set.")
    parser.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"],
                        help="Aggregation strategy (default: fedavg).")
    parser.add_argument("--fraction-fit", type=float, default=1.0,
                        help="Fraction of clients per round. Set < 1.0 for dropout tolerance.")
    parser.add_argument("--proximal-mu", type=float, default=1.0,
                        help="FedProx proximal term (only used with --strategy fedprox).")
    parser.add_argument("--model-type", default="mlp",
                        choices=["mlp", "resnet-tabular", "transformer-tabular"],
                        help="Model architecture used during training (default: mlp).")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop if no improvement for N rounds (0 = disabled).")
    parser.add_argument("--early-stopping-metric", default="loss",
                        choices=["loss", "accuracy", "macro_f1"],
                        help="Metric to monitor for early stopping (default: loss).")
    parser.add_argument("--no-distribution-monitor", action="store_true",
                        help="Skip the pre-training distribution report.")
    parser.add_argument("--crypto", action="store_true",
                        help="Enable AES-256-GCM + RSA weight encryption.")
    parser.add_argument("--attest", action="store_true",
                        help="Run SGX attestation before starting the FL server.")
    args = parser.parse_args()

    if args.attest:
        if _ATTESTATION_AVAILABLE:
            att = AttestationServer()
            att.attest()
        else:
            print("[run_server] WARNING: attestation module not found — skipping.")

    start_server(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        local_epochs=args.local_epochs,
        server_address=args.address,
        save_dir=args.save_dir,
        use_crypto=args.crypto,
        strategy_name=args.strategy,
        fraction_fit=args.fraction_fit,
        proximal_mu=args.proximal_mu,
        model_type=args.model_type,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        monitor_distributions=not args.no_distribution_monitor,
    )
