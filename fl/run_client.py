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
    args = parser.parse_args()

    client_index = int(args.id) - 1
    csv_path = args.csv
    if csv_path is None:
        if client_index < 0 or client_index >= len(default_csvs):
            raise ValueError(
                f"Client ID {args.id} is out of range for the default CSV mapping."
            )
        csv_path = default_csvs[client_index]

    start_client(csv_path=csv_path, client_id=args.id, server_address=args.server)
