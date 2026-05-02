"""
Flower FL client for one HAR CSV / organisation.
"""
import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_class_weights, load_csv_data  # noqa: E402
from model import get_model  # noqa: E402
from train_local import evaluate, train_one_epoch  # noqa: E402

# ── M2: per-client delta values (row counts confirmed in contracts.md) ───────
# delta = 1 / n_train  where n_train = 80% of total rows (0.8 × client rows)
# Client 1: 3105 × 0.8 = 2484  → δ = 4.03e-4
# Client 2: 3426 × 0.8 = 2741  → δ = 3.65e-4
# Client 3: 3768 × 0.8 = 3014  → δ = 3.32e-4
# Using approximate safe values — will be overridden by actual train_size
_CLIENT_APPROX_DELTA = {
    "1": 1 / 2484,
    "2": 1 / 2741,
    "3": 1 / 3014,
}
# ─────────────────────────────────────────────────────────────────────────────


class IntelliClaveClient(fl.client.NumPyClient):
    def __init__(
        self,
        csv_path: str,
        client_id: str,
        local_epochs: int = 3,
        learning_rate: float = 1e-3,
        # ── M2: DP parameters — all optional, default = DP off ──────────────────
        use_dp: bool = False,
        target_epsilon: float = 10.0,
        max_grad_norm: float = 1.0,
        num_fl_rounds: int = 5,        # ── M2 FIX: total FL rounds for correct DP budget
        # ─────────────────────────────────────────────────────────────────────────
    ):
        self.cid = client_id
        self.local_epochs = local_epochs
        self.device = torch.device("cpu")
        self.num_fl_rounds = num_fl_rounds  # ── M2 FIX: store for use in _attach_privacy_engine

        # M1 original data loading — untouched
        self.train_loader, self.test_loader, self.metadata = load_csv_data(csv_path)
        self.model = get_model(
            self.metadata.input_dim,
            self.metadata.num_classes,
        ).to(self.device)
        class_weights = load_class_weights(device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # M1 original print — untouched
        print(
            f"[Client {client_id}] ready input_dim={self.metadata.input_dim} "
            f"classes={self.metadata.num_classes} csv={csv_path}"
        )

        # ── M2: DP setup — runs only if use_dp=True ─────────────────────────────
        self.use_dp = use_dp
        self.privacy_engine = None
        self.target_epsilon = target_epsilon
        self.max_grad_norm = max_grad_norm

        # Use actual train_size from metadata for accurate delta
        self.target_delta = 1.0 / self.metadata.train_size

        if self.use_dp:
            self._attach_privacy_engine()
        # ─────────────────────────────────────────────────────────────────────────

    # ── M2: private method — attaches Opacus to model, optimizer, loader ────────
    def _attach_privacy_engine(self):
        """
        Wraps self.model, self.optimizer, and self.train_loader with Opacus.
        Called once during __init__ when use_dp=True.
        M1's train_one_epoch() works unchanged after this because Opacus
        replaces the optimizer and loader transparently.
        """
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator

            # Validate and auto-fix model (replaces BatchNorm with GroupNorm)
            # M1's HARClassifier uses only Linear + ReLU so this is a safety net
            errors = ModuleValidator.validate(self.model, strict=False)
            if errors:
                print(f"[Client {self.cid}][M2-DP] Auto-fixing model: {errors}")
                self.model = ModuleValidator.fix(self.model)

            # ── M2 FIX: total_epochs = local_epochs × num_fl_rounds
            # Previously only local_epochs was passed, causing Opacus to
            # calibrate noise for 3 epochs but spend budget across 15 (5 rounds × 3).
            # Now Opacus knows the full training duration upfront and sets
            # noise correctly so ε stays at target across ALL rounds.
            total_epochs = self.local_epochs * self.num_fl_rounds

            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = (
                self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    epochs=total_epochs,           # ── M2 FIX: was self.local_epochs
                    target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta,
                    max_grad_norm=self.max_grad_norm,
                )
            )
            print(
                f"[Client {self.cid}][M2-DP] PrivacyEngine attached: "
                f"ε={self.target_epsilon}, δ={self.target_delta:.2e}, "
                f"max_grad_norm={self.max_grad_norm}, "
                f"train_size={self.metadata.train_size}, "
                f"total_epochs={total_epochs} ({self.local_epochs} local × {self.num_fl_rounds} rounds)"  # ── M2 FIX: log total epochs
            )

        except ImportError:
            print(
                f"[Client {self.cid}][M2-DP] WARNING: Opacus not installed. "
                "Running without DP."
            )
            self.use_dp = False
    # ─────────────────────────────────────────────────────────────────────────────

    # M1 original — untouched
    def get_parameters(self, config) -> List[np.ndarray]:
        return [value.cpu().numpy() for value in self.model.state_dict().values()]

    # M1 original — untouched
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
        # M1 original — untouched
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

        # M1 original metrics dict
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        }

        # ── M2: append epsilon to metrics if DP is active ───────────────────────
        if self.use_dp and self.privacy_engine is not None:
            eps = self.privacy_engine.get_epsilon(delta=self.target_delta)
            metrics["epsilon"] = float(eps)
            metrics["delta"] = float(self.target_delta)
            metrics["client_id"] = self.cid
            print(
                f"[Client {self.cid}][M2-DP] "
                f"loss={loss:.4f} acc={accuracy:.4f} ε={eps:.4f}"
            )
        # ─────────────────────────────────────────────────────────────────────────

        return (
            self.get_parameters({}),
            len(self.train_loader.dataset),
            metrics,
        )

    # M1 original evaluate — untouched
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


# M1 original start_client — untouched
def start_client(
    csv_path: str,
    client_id: str,
    server_address: str = "127.0.0.1:8080",
    # ── M2: DP kwargs forwarded through ─────────────────────────────────────────
    use_dp: bool = False,
    target_epsilon: float = 10.0,
    max_grad_norm: float = 1.0,
    num_fl_rounds: int = 5,            # ── M2 FIX: forwarded through to client
    # ─────────────────────────────────────────────────────────────────────────────
):
    fl.client.start_numpy_client(
        server_address=server_address,
        client=IntelliClaveClient(
            csv_path=csv_path,
            client_id=client_id,
            # ── M2 ──────────────────────────────────────────────────────────────
            use_dp=use_dp,
            target_epsilon=target_epsilon,
            max_grad_norm=max_grad_norm,
            num_fl_rounds=num_fl_rounds,   # ── M2 FIX: passed through
            # ────────────────────────────────────────────────────────────────────
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # M1 original arguments — untouched
    parser.add_argument("--csv", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--server", default="127.0.0.1:8080")
    # ── M2: DP flags — added below M1's args ────────────────────────────────────
    parser.add_argument(
        "--dp",
        action="store_true",
        help="[M2] Enable Differential Privacy via Opacus.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=10.0,
        help="[M2] Target epsilon (privacy budget). Default=10.0.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="[M2] Gradient clipping norm for DP-SGD. Default=1.0.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="[M2] Total FL rounds — must match server --rounds. Default=5.",  # ── M2 FIX
    )
    # ─────────────────────────────────────────────────────────────────────────────
    args = parser.parse_args()
    start_client(
        args.csv,
        args.id,
        args.server,
        # ── M2 ──────────────────────────────────────────────────────────────────
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        num_fl_rounds=args.rounds,         # ── M2 FIX: passed through
        # ────────────────────────────────────────────────────────────────────────
    )