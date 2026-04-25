"""
Single-process FL simulation using the real HAR client CSVs.

Usage:
    python fl/run_fl_simulation.py
    python fl/run_fl_simulation.py --rounds 15 --clients 3 --local-epochs 2
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import Metrics

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import get_default_client_csvs, load_class_weights, load_csv_data  # noqa: E402
from fl_server import SaveModelStrategy, weighted_average  # noqa: E402
from model import get_model  # noqa: E402
from train_local import evaluate, train_one_epoch  # noqa: E402


SAVE_DIR = "results/fl_rounds"
NUM_ROUNDS = 10
NUM_CLIENTS = 3


class SimClient(fl.client.NumPyClient):
    def __init__(self, csv_path: str, cid: str, local_epochs: int, learning_rate: float):
        self.cid = cid
        self.local_epochs = local_epochs
        self.device = torch.device("cpu")
        self.train_loader, self.test_loader, self.metadata = load_csv_data(csv_path)
        self.model = get_model(
            self.metadata.input_dim,
            self.metadata.num_classes,
        ).to(self.device)
        class_weights = load_class_weights(device=self.device)
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
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )
        accuracy, macro_f1 = evaluate(self.model, self.train_loader, self.device)
        return (
            self.get_parameters({}),
            len(self.train_loader.dataset),
            {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
            },
        )

    def evaluate(self, params, config):
        self.set_parameters(params)
        total_loss = 0.0
        n_examples = 0
        self.model.eval()

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                logits = self.model(X_batch.to(self.device))
                loss = self.criterion(logits, y_batch.to(self.device))
                total_loss += loss.item() * len(y_batch)
                n_examples += len(y_batch)

        accuracy, macro_f1 = evaluate(self.model, self.test_loader, self.device)
        return (
            float(total_loss / n_examples),
            n_examples,
            {
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
            },
        )


def main(
    num_rounds: int = NUM_ROUNDS,
    num_clients: int = NUM_CLIENTS,
    local_epochs: int = 3,
    learning_rate: float = 1e-3,
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    csv_files = get_default_client_csvs(num_clients)
    missing = [path for path in csv_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Expected processed HAR CSVs are missing:\n" + "\n".join(missing)
        )

    _, _, metadata = load_csv_data(csv_files[0])

    def client_fn(cid: str):
        return SimClient(
            csv_path=csv_files[int(cid)],
            cid=cid,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
        )

    strategy = SaveModelStrategy(
        input_dim=metadata.input_dim,
        num_classes=metadata.num_classes,
        save_dir=SAVE_DIR,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda _: {"local_epochs": local_epochs},
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

    results = {
        "losses_distributed": [(int(rnd), float(loss)) for rnd, loss in history.losses_distributed],
        "losses_centralized": [(int(rnd), float(loss)) for rnd, loss in history.losses_centralized],
        "metrics_distributed": {
            key: [(int(rnd), float(value)) for rnd, value in values]
            for key, values in history.metrics_distributed.items()
        },
        "metrics_centralized": {
            key: [(int(rnd), float(value)) for rnd, value in values]
            for key, values in history.metrics_centralized.items()
        },
    }
    out_path = os.path.join(SAVE_DIR, "fl_history.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFL simulation done -> {out_path}")
    if history.metrics_distributed:
        final_metrics = {key: values[-1][1] for key, values in history.metrics_distributed.items()}
        print(f"Final distributed metrics: {final_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(
        num_rounds=args.rounds,
        num_clients=args.clients,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
    )
