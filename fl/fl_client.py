"""
Flower FL client for one HAR CSV / organisation.
"""
import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_class_weights, load_csv_data  # noqa: E402
from model import get_model  # noqa: E402
from train_local import evaluate, train_one_epoch  # noqa: E402


class IntelliClaveClient(fl.client.NumPyClient):
    def __init__(
        self,
        csv_path: str,
        client_id: str,
        local_epochs: int = 3,
        learning_rate: float = 1e-3,
    ):
        self.cid = client_id
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
        print(
            f"[Client {client_id}] ready input_dim={self.metadata.input_dim} "
            f"classes={self.metadata.num_classes} csv={csv_path}"
        )

    def get_parameters(self, config) -> List[np.ndarray]:
        return [value.cpu().numpy() for value in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = OrderedDict(
            (
                key,
                torch.tensor(value),
            )
            for key, value in zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
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

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
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


def start_client(
    csv_path: str,
    client_id: str,
    server_address: str = "127.0.0.1:8080",
):
    fl.client.start_numpy_client(
        server_address=server_address,
        client=IntelliClaveClient(csv_path=csv_path, client_id=client_id),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--server", default="127.0.0.1:8080")
    args = parser.parse_args()
    start_client(args.csv, args.id, args.server)
