"""
Single-process FL simulation using client CSVs.

Usage:
    python fl/run_fl_simulation.py
    python fl/run_fl_simulation.py --rounds 15 --clients 3 --local-epochs 2
    python fl/run_fl_simulation.py --strategy fedprox --model-type resnet-tabular
    python fl/run_fl_simulation.py --early-stopping-patience 3
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import Metrics

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (  # noqa: E402
    get_default_client_csvs,
    load_class_weights,
    load_csv_data,
    validate_client_schemas,
)
from fl_server import (  # noqa: E402
    SaveModelStrategy,
    _write_run_summary,
    build_strategy,
    make_timestamped_save_dir,
    monitor_client_distributions,
    weighted_average,
)
from model import get_model  # noqa: E402
from train_local import evaluate, train_one_epoch  # noqa: E402


NUM_ROUNDS  = 10
NUM_CLIENTS = 3


class SimClient(fl.client.NumPyClient):
    def __init__(
        self,
        csv_path: str,
        cid: str,
        local_epochs: int,
        learning_rate: float,
        model_type: str = "mlp",
        batch_size: int = 32,
    ):
        self.cid          = cid
        self.local_epochs = local_epochs
        self.device       = torch.device("cpu")

        self.train_loader, self.test_loader, self.metadata = load_csv_data(
            csv_path, batch_size=batch_size
        )
        self.model = get_model(
            self.metadata.input_dim,
            self.metadata.num_classes,
            model_type=model_type,
        ).to(self.device)
        class_weights = load_class_weights(
            num_classes=self.metadata.num_classes, device=self.device
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_parameters(self, config):
        return [value.cpu().numpy() for value in self.model.state_dict().values()]

    def set_parameters(self, params):
        state_dict = OrderedDict(
            (key, torch.tensor(value))
            for key, value in zip(self.model.state_dict().keys(), params)
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        epochs = int(config.get("local_epochs", self.local_epochs))
        loss = 0.0
        for _ in range(epochs):
            loss = train_one_epoch(
                self.model, self.train_loader,
                self.optimizer, self.criterion, self.device,
            )
        # Report validation accuracy (test split), not training accuracy
        accuracy, macro_f1 = evaluate(self.model, self.test_loader, self.device)
        return (
            self.get_parameters({}),
            len(self.train_loader.dataset),
            {"loss": float(loss), "accuracy": float(accuracy), "macro_f1": float(macro_f1)},
        )

    def evaluate(self, params, config):
        self.set_parameters(params)
        total_loss = 0.0
        n_examples = 0
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                logits = self.model(X_batch.to(self.device))
                loss   = self.criterion(logits, y_batch.to(self.device))
                total_loss += loss.item() * len(y_batch)
                n_examples += len(y_batch)
        accuracy, macro_f1 = evaluate(self.model, self.test_loader, self.device)
        return (
            float(total_loss / n_examples),
            n_examples,
            {"accuracy": float(accuracy), "macro_f1": float(macro_f1)},
        )


def main(
    num_rounds: int = NUM_ROUNDS,
    num_clients: int = NUM_CLIENTS,
    local_epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    save_dir: str = "",
    fraction_fit: float = 1.0,
    strategy_name: str = "fedavg",
    proximal_mu: float = 1.0,
    model_type: str = "mlp",
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "loss",
    monitor_distributions: bool = True,
):
    # Auto-generate timestamped save dir
    if not save_dir:
        save_dir = make_timestamped_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Simulation] Results will be saved to: {save_dir}")

    csv_files = get_default_client_csvs(num_clients)
    missing = [p for p in csv_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Expected client CSVs are missing:\n" + "\n".join(missing) +
            "\nRun data/datascripts/pipeline.py first to prepare your data."
        )

    # Schema validation + distribution monitoring before training
    validate_client_schemas(csv_files)
    if monitor_distributions:
        monitor_client_distributions(csv_files, save_dir=save_dir)

    _, _, metadata = load_csv_data(csv_files[0], batch_size=batch_size)

    def client_fn(cid: str):
        return SimClient(
            csv_path=csv_files[int(cid)],
            cid=cid,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            model_type=model_type,
            batch_size=batch_size,
        )

    strategy = build_strategy(
        strategy_name=strategy_name,
        input_dim=metadata.input_dim,
        num_classes=metadata.num_classes,
        class_names=metadata.class_names,
        save_dir=save_dir,
        crypto_ctx=None,
        min_clients=num_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        local_epochs=local_epochs,
        proximal_mu=proximal_mu,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        model_type=model_type,
    )

    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
    except ImportError as exc:
        raise RuntimeError(
            "Flower simulation requires the optional 'ray' dependency. "
            "Install it with `pip install -U flwr[simulation]` in the IntelliClave "
            "environment, or run distributed FL with fl/run_server.py and "
            "fl/run_client.py instead."
        ) from exc

    # Save FL history
    results = {
        "losses_distributed": [
            (int(rnd), float(loss)) for rnd, loss in history.losses_distributed
        ],
        "losses_centralized": [
            (int(rnd), float(loss)) for rnd, loss in history.losses_centralized
        ],
        "metrics_distributed": {
            key: [(int(rnd), float(val)) for rnd, val in values]
            for key, values in history.metrics_distributed.items()
        },
        "metrics_centralized": {
            key: [(int(rnd), float(val)) for rnd, val in values]
            for key, values in history.metrics_centralized.items()
        },
    }
    out_path = os.path.join(save_dir, "fl_history.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFL simulation done -> {out_path}")

    if history.metrics_distributed:
        final = {k: v[-1][1] for k, v in history.metrics_distributed.items()}
        print(f"Final distributed metrics: {final}")

    # Write status.json and results/results.json for the dashboard
    _write_run_summary(save_dir, strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-process FL simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rounds",      type=int,   default=NUM_ROUNDS)
    parser.add_argument("--clients",     type=int,   default=NUM_CLIENTS)
    parser.add_argument("--local-epochs",type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--batch-size",  type=int,   default=32,
                        help="DataLoader batch size.")
    parser.add_argument("--save-dir",    default="",
                        help="Output directory. Auto-generates timestamped path if not set.")
    parser.add_argument("--fraction-fit",type=float, default=1.0,
                        help="Fraction of clients per round. Set < 1.0 for dropout tolerance.")
    parser.add_argument("--strategy",    default="fedavg",
                        choices=["fedavg", "fedprox"],
                        help="Aggregation strategy.")
    parser.add_argument("--proximal-mu", type=float, default=1.0,
                        help="FedProx proximal term (only used with --strategy fedprox).")
    parser.add_argument("--model-type",  default="mlp",
                        choices=["mlp", "resnet-tabular", "transformer-tabular"],
                        help="Model architecture.")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop if no improvement for N rounds (0 = disabled).")
    parser.add_argument("--early-stopping-metric", default="loss",
                        choices=["loss", "accuracy", "macro_f1"],
                        help="Metric to monitor for early stopping.")
    parser.add_argument("--no-distribution-monitor", action="store_true",
                        help="Skip the pre-training distribution report.")
    args = parser.parse_args()
    main(
        num_rounds=args.rounds,
        num_clients=args.clients,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        fraction_fit=args.fraction_fit,
        strategy_name=args.strategy,
        proximal_mu=args.proximal_mu,
        model_type=args.model_type,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        monitor_distributions=not args.no_distribution_monitor,
    )
