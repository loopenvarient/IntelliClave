import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fl_server import infer_default_input_dim, start_server  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IntelliClave Flower server.")
    parser.add_argument("--input-dim", type=int, default=infer_default_input_dim())
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--save-dir", default="results/fl_rounds")
    args = parser.parse_args()

    start_server(
        input_dim=args.input_dim,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        local_epochs=args.local_epochs,
        server_address=args.address,
        save_dir=args.save_dir,
    )
