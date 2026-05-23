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
from robust_aggregation import validate_robust_agg_cli  # noqa: E402

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
    parser.add_argument(
        "--robust-agg", default="fedavg",
        choices=["fedavg", "krum", "multi-krum", "trimmed-mean", "median"],
        help="Byzantine-robust aggregation (default: fedavg). "
             "krum/multi-krum require --min-clients >= 5.",
    )
    parser.add_argument(
        "--byzantine-fraction", type=float, default=0.33,
        help="Assumed Byzantine client fraction for Krum / Trimmed Mean.",
    )
    parser.add_argument(
        "--round-timeout", type=float, default=0.0,
        help="Drop straggler clients after N seconds per fit/evaluate (0 = off).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume FL from the last checkpoint in --save-dir.",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Top-K sparsify client updates before upload (reduces effective bandwidth).",
    )
    parser.add_argument(
        "--top-k-fraction", type=float, default=0.1,
        help="Fraction of weights to keep per layer when --compress is set (default 0.1).",
    )
    parser.add_argument("--crypto", action="store_true",
                        help="Enable AES-256-GCM + RSA weight encryption.")
    parser.add_argument("--attest", action="store_true",
                        help="Run SGX attestation before starting the FL server.")
    args = parser.parse_args()

    # ── Krum client-count guard ───────────────────────────────────────────────
    # Krum requires n >= 2f+3 for formal Byzantine robustness.
    # With f=1 that means at least 5 clients. At n=3, auto_f(3)=0 so Krum
    # degenerates to picking a single update with zero Byzantine tolerance.
    # validate_robust_agg_cli enforces this and exits with code 1 on violation.
    if args.robust_agg in ("krum", "multi-krum") and args.min_clients < 5:
        print(
            f"[run_server] ERROR: --robust-agg {args.robust_agg} requires "
            f"--min-clients >= 5 (got {args.min_clients}).\n"
            f"  Krum needs n >= 2f+3 clients for f=1 Byzantine tolerance.\n"
            f"  Options:\n"
            f"    - Use --min-clients 5 (generate data: "
            f"python data/datascripts/pipeline.py --n-clients 5)\n"
            f"    - Use --robust-agg trimmed-mean or median (robust at any n >= 2)\n"
            f"    - Keep --robust-agg fedavg (default) for the 3-client demo",
            file=sys.stderr,
        )
        raise SystemExit(1)
    validate_robust_agg_cli(args.robust_agg, args.min_clients)

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
        robust_agg=args.robust_agg,
        byzantine_fraction=args.byzantine_fraction,
        round_timeout=args.round_timeout,
        resume=args.resume,
        compress_updates=args.compress,
        top_k_fraction=args.top_k_fraction,
    )
