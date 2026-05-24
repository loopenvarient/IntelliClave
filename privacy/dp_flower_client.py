"""
privacy/dp_flower_client.py

Standalone Flower client that trains with Opacus via DPTrainer.

Note: Production FL uses fl/fl_client.py with use_dp=True. This module remains
for privacy experiments and budget tooling that import from the privacy package.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import torch
import torch.nn as nn

from .data_loader import load_client_data
from .dp_trainer import DPTrainer

# Optional: apply server global normalization (same contract as fl_client)
_FL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fl"))
if _FL_DIR not in sys.path:
    sys.path.insert(0, _FL_DIR)

try:
    from data_utils import global_norm_arrays_from_config, load_csv_data  # noqa: E402
    _FL_DATA_UTILS = True
except ImportError:
    _FL_DATA_UTILS = False


class DPFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        csv_path: str,
        local_epochs: int = 3,
        num_fl_rounds: int = 5,
        target_epsilon: float = 10.0,
        max_grad_norm: float = 1.0,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        global_mean: Optional[torch.Tensor] = None,
        global_std: Optional[torch.Tensor] = None,
    ):
        """
        client_id      : string identifier, e.g. "1"
        model          : nn.Module (e.g. FLClassifier from fl/model.py)
        csv_path       : path to the client CSV
        local_epochs   : training epochs per FL round
        num_fl_rounds  : total FL rounds — must match server config
        target_epsilon : DP privacy budget
        """
        self.client_id = client_id
        self.csv_path = csv_path
        self.local_epochs = local_epochs
        self.num_fl_rounds = num_fl_rounds
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._global_norm_applied = global_mean is not None and global_std is not None

        loader, delta, num_classes, input_dim = self._load_data(
            global_mean=global_mean,
            global_std=global_std,
        )
        self.loader = loader
        self.num_examples = len(loader.dataset)

        total_epochs = local_epochs * num_fl_rounds
        self.trainer = DPTrainer(
            model=model,
            target_epsilon=target_epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
            epochs=total_epochs,
            num_classes=num_classes,
            lr=learning_rate,
        )
        self.trainer.attach(loader)
        print(
            f"[{client_id}] PrivacyEngine attached: "
            f"ε={target_epsilon}, δ={delta:.2e}, "
            f"total_epochs={total_epochs} "
            f"({local_epochs} local × {num_fl_rounds} rounds), "
            f"input_dim={input_dim}"
        )

    def _load_data(self, global_mean=None, global_std=None):
        if global_mean is not None and global_std is not None and _FL_DATA_UTILS:
            g_mean = (
                global_mean.numpy()
                if isinstance(global_mean, torch.Tensor)
                else global_mean
            )
            g_std = (
                global_std.numpy()
                if isinstance(global_std, torch.Tensor)
                else global_std
            )
            train_loader, _, metadata = load_csv_data(
                self.csv_path,
                batch_size=self._batch_size,
                drop_last_for_dp=True,
                global_mean=g_mean,
                global_std=g_std,
            )
            delta = 1.0 / metadata.train_size
            return train_loader, delta, metadata.num_classes, metadata.input_dim

        return load_client_data(self.csv_path, batch_size=self._batch_size)

    def _ensure_global_normalization(self, config: Dict) -> None:
        if self._global_norm_applied or not _FL_DATA_UTILS:
            return
        global_mean, global_std = global_norm_arrays_from_config(config)
        if global_mean is None or global_std is None:
            return

        print(f"[{self.client_id}] Applying server-coordinated global normalization")
        loader, delta, num_classes, _ = self._load_data(
            global_mean=global_mean,
            global_std=global_std,
        )
        self.loader = loader
        self.num_examples = len(loader.dataset)
        self._global_norm_applied = True

        total_epochs = self.local_epochs * self.num_fl_rounds
        model = self.trainer.get_model()
        self.trainer = DPTrainer(
            model=model,
            target_epsilon=self.trainer.target_epsilon,
            target_delta=delta,
            max_grad_norm=self.trainer.max_grad_norm,
            epochs=total_epochs,
            num_classes=num_classes,
            lr=self._learning_rate,
        )
        self.trainer.attach(loader)

    def get_parameters(self, config) -> List:
        return [
            val.cpu().numpy()
            for val in self.trainer.get_model().state_dict().values()
        ]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(
                    self.trainer.get_model().state_dict().keys(), parameters
                )
            }
        )
        self.trainer.get_model().load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config) -> Tuple[List, int, Dict]:
        self._ensure_global_normalization(config)
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", self.local_epochs))

        metrics: Dict = {}
        for _ in range(epochs):
            metrics = self.trainer.train_one_round()

        print(
            f"[{self.client_id}] ε={metrics['epsilon']:.4f} | "
            f"loss={metrics['loss']:.4f}"
        )
        return (
            self.get_parameters(config),
            self.num_examples,
            {
                "epsilon": metrics["epsilon"],
                "loss": metrics["loss"],
                "delta": metrics.get("delta"),
                "client_id": self.client_id,
            },
        )

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        self._ensure_global_normalization(config)
        self.set_parameters(parameters)
        model = self.trainer.get_model()
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X_batch, y_batch in self.loader:
                out = model(X_batch)
                total_loss += criterion(out, y_batch).item() * len(y_batch)
                correct += (out.argmax(1) == y_batch).sum().item()
                total += len(y_batch)

        if total == 0:
            return 0.0, 0, {"accuracy": 0.0, "client_id": self.client_id}

        return (
            float(total_loss / total),
            total,
            {"accuracy": float(correct / total), "client_id": self.client_id},
        )


def start_dp_client(
    csv_path: str,
    client_id: str,
    model: nn.Module,
    server_address: str = "127.0.0.1:8080",
    local_epochs: int = 3,
    num_fl_rounds: int = 10,
    target_epsilon: float = 10.0,
    max_grad_norm: float = 1.0,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> None:
    """Connect a DPFlowerClient to a running Flower server."""
    fl.client.start_numpy_client(
        server_address=server_address,
        client=DPFlowerClient(
            client_id=client_id,
            model=model,
            csv_path=csv_path,
            local_epochs=local_epochs,
            num_fl_rounds=num_fl_rounds,
            target_epsilon=target_epsilon,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            learning_rate=learning_rate,
        ),
    )


if __name__ == "__main__":
    import argparse

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(_root, "fl"))
    from data_utils import get_default_client_csvs  # noqa: E402
    from model import get_model  # noqa: E402

    parser = argparse.ArgumentParser(description="Run privacy-module DP Flower client.")
    parser.add_argument("--id", required=True, help="Client id (1, 2, 3, ...)")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=10.0)
    args = parser.parse_args()

    idx = int(args.id) - 1
    csv_path = args.csv or get_default_client_csvs()[idx]
    _, _, num_classes, input_dim = load_client_data(csv_path)

    model = get_model(input_dim, num_classes)
    start_dp_client(
        csv_path=csv_path,
        client_id=args.id,
        model=model,
        server_address=args.server,
        local_epochs=args.local_epochs,
        num_fl_rounds=args.rounds,
        target_epsilon=args.epsilon,
    )
