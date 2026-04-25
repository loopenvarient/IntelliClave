"""
Flower FL server with FedAvg and saved global model checkpoints.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import get_default_client_csvs, infer_csv_schema  # noqa: E402
from model import get_model  # noqa: E402


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        save_dir: str = "results/fl_rounds",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.round_log: List[Dict] = []
        os.makedirs(save_dir, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated)
            np.savez(os.path.join(self.save_dir, f"round_{server_round}.npz"), *weights)
            self._save_pth(weights, server_round)
        return aggregated, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        entry = {
            "round": server_round,
            "loss": round(float(loss), 5) if loss is not None else None,
        }
        if metrics:
            entry.update({key: round(float(value), 5) for key, value in metrics.items()})
        self.round_log.append(entry)

        with open(os.path.join(self.save_dir, "fl_metrics.json"), "w") as f:
            json.dump(self.round_log, f, indent=2)

        print(f"[Server] Round {server_round} {entry}")
        return loss, metrics

    def _save_pth(self, weights: List[np.ndarray], round_number: int):
        model = get_model(self.input_dim, self.num_classes)
        state_dict = {
            key: torch.tensor(value)
            for key, value in zip(model.state_dict().keys(), weights)
        }
        model.load_state_dict(state_dict, strict=True)

        round_path = os.path.join(self.save_dir, f"global_model_round_{round_number}.pth")
        latest_path = os.path.join(self.save_dir, "global_model_latest.pth")
        torch.save(model.state_dict(), round_path)
        torch.save(model.state_dict(), latest_path)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for key in ["accuracy", "macro_f1"]:
        aggregated[key] = sum(
            num_examples * metric_dict.get(key, 0.0)
            for num_examples, metric_dict in metrics
        ) / total_examples
    return aggregated


def infer_default_input_dim() -> int:
    first_csv = get_default_client_csvs()[0]
    input_dim, _ = infer_csv_schema(first_csv)
    return input_dim


def start_server(
    input_dim: int,
    num_rounds: int = 10,
    min_clients: int = 3,
    local_epochs: int = 3,
    server_address: str = "0.0.0.0:8080",
    save_dir: str = "results/fl_rounds",
):
    strategy = SaveModelStrategy(
        input_dim=input_dim,
        num_classes=6,
        save_dir=save_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda _: {"local_epochs": local_epochs},
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
