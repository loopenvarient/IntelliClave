"""
Flower FL client for one HAR CSV / organisation.
"""
import argparse
import json
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

# ── Crypto layer import ───────────────────────────────────────────────────────
_CRYPTO_DIR = os.path.join(os.path.dirname(__file__), "..", "crypto", "certs")
sys.path.insert(0, os.path.abspath(_CRYPTO_DIR))
try:
    from crypto_context import CryptoContext  # noqa: E402
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    print("[fl_client] WARNING: crypto_context not found — running without encryption.")
# ─────────────────────────────────────────────────────────────────────────────

# Per-client delta values (row counts confirmed in contracts.md)
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
        # ── DP parameters — all optional, default = DP off ───────────────────────
        use_dp: bool = False,
        target_epsilon: float = 10.0,
        max_grad_norm: float = 1.0,
        num_fl_rounds: int = 5,
        # ─────────────────────────────────────────────────────────────────────────
        # ── Crypto: optional encryption of weights in transit ────────────────────
        use_crypto: bool = False,
        server_public_pem: bytes = None,
        # ─────────────────────────────────────────────────────────────────────────
    ):
        self.cid = client_id
        self.local_epochs = local_epochs
        self.device = torch.device("cpu")
        self.num_fl_rounds = num_fl_rounds

        # Set up client-side crypto context
        self.use_crypto = use_crypto and _CRYPTO_AVAILABLE and server_public_pem is not None
        self._crypto_ctx = None
        if self.use_crypto:
            self._crypto_ctx = CryptoContext.from_public_pem(server_public_pem)
            print(f"[Client {client_id}][Crypto] Encryption enabled — weights will be "
                  "AES-256-GCM encrypted before transmission.")
        elif use_crypto:
            print(f"[Client {client_id}][Crypto] WARNING: crypto requested but unavailable "
                  "— running without encryption.")

        # Data loading
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

        # DP setup — runs only if use_dp=True
        self.use_dp = use_dp
        self.privacy_engine = None
        self.target_epsilon = target_epsilon
        self.max_grad_norm = max_grad_norm

        # Use actual train_size from metadata for accurate delta
        self.target_delta = 1.0 / self.metadata.train_size

        if self.use_dp:
            self._attach_privacy_engine()

    # Attaches Opacus PrivacyEngine to model, optimizer, and train_loader
    def _attach_privacy_engine(self):
        """
        Wraps self.model, self.optimizer, and self.train_loader with Opacus.
        Called once during __init__ when use_dp=True.
        train_one_epoch() works unchanged after this because Opacus
        replaces the optimizer and loader transparently.
        """
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator

            # Validate and auto-fix model (replaces BatchNorm with GroupNorm)
            # HARClassifier uses only Linear + ReLU so this is a safety net
            errors = ModuleValidator.validate(self.model, strict=False)
            if errors:
                print(f"[Client {self.cid}][DP] Auto-fixing model: {errors}")
                self.model = ModuleValidator.fix(self.model)

            # total_epochs = local_epochs × num_fl_rounds
            # Opacus needs the full training duration upfront to calibrate noise
            # correctly so ε stays at target across ALL rounds.
            total_epochs = self.local_epochs * self.num_fl_rounds

            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = (
                self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    epochs=total_epochs,
                    target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta,
                    max_grad_norm=self.max_grad_norm,
                )
            )
            print(
                f"[Client {self.cid}][DP] PrivacyEngine attached: "
                f"ε={self.target_epsilon}, δ={self.target_delta:.2e}, "
                f"max_grad_norm={self.max_grad_norm}, "
                f"train_size={self.metadata.train_size}, "
                f"total_epochs={total_epochs} ({self.local_epochs} local × {self.num_fl_rounds} rounds)"
            )

        except ImportError:
            print(
                f"[Client {self.cid}][DP] WARNING: Opacus not installed. "
                "Running without DP."
            )
            self.use_dp = False
    # ─────────────────────────────────────────────────────────────────────────────

    # Returns plaintext weights (server reads these for init)
    def get_parameters(self, config) -> List[np.ndarray]:
        return [value.cpu().numpy() for value in self.model.state_dict().values()]
    # ─────────────────────────────────────────────────────────────────────────

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = OrderedDict(
            (key, torch.tensor(value))
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

        # Metrics dict
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        }

        # Append epsilon to metrics if DP is active
        if self.use_dp and self.privacy_engine is not None:
            eps = self.privacy_engine.get_epsilon(delta=self.target_delta)
            metrics["epsilon"] = float(eps)
            metrics["delta"] = float(self.target_delta)
            metrics["client_id"] = self.cid
            print(
                f"[Client {self.cid}][DP] "
                f"loss={loss:.4f} acc={accuracy:.4f} ε={eps:.4f}"
            )

        # Encrypt weights before sending to server if crypto is enabled
        outgoing_weights = self.get_parameters({})
        if self.use_crypto and self._crypto_ctx is not None:
            payload = self._crypto_ctx.encrypt_weights(outgoing_weights)
            payload_bytes = json.dumps(payload).encode()
            packed = np.frombuffer(payload_bytes, dtype=np.uint8).copy()
            print(f"[Client {self.cid}][Crypto] fit() weights encrypted "
                  f"({len(payload_bytes):,} bytes)")
            outgoing_weights = [packed]

        return (
            outgoing_weights,
            len(self.train_loader.dataset),
            metrics,
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
    use_dp: bool = False,
    target_epsilon: float = 10.0,
    max_grad_norm: float = 1.0,
    num_fl_rounds: int = 5,
    use_crypto: bool = False,
    server_public_pem: bytes = None,
):
    fl.client.start_numpy_client(
        server_address=server_address,
        client=IntelliClaveClient(
            csv_path=csv_path,
            client_id=client_id,
            use_dp=use_dp,
            target_epsilon=target_epsilon,
            max_grad_norm=max_grad_norm,
            num_fl_rounds=num_fl_rounds,
            use_crypto=use_crypto,
            server_public_pem=server_public_pem,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy via Opacus.")
    parser.add_argument("--epsilon", type=float, default=10.0, help="Target epsilon (privacy budget). Default=10.0.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm for DP-SGD. Default=1.0.")
    parser.add_argument("--rounds", type=int, default=5, help="Total FL rounds — must match server. Default=5.")
    parser.add_argument("--crypto", action="store_true", help="Encrypt weights in transit using AES-256-GCM + RSA.")
    parser.add_argument("--pubkey", default=None, help="Path to server public key PEM file. Defaults to crypto/certs/keys/server_public.pem")
    args = parser.parse_args()

    # Load public key if crypto enabled
    server_public_pem = None
    if args.crypto:
        pubkey_path = args.pubkey or os.path.join(
            os.path.dirname(__file__), "..", "crypto", "certs", "keys", "server_public.pem"
        )
        pubkey_path = os.path.abspath(pubkey_path)
        if not os.path.exists(pubkey_path):
            print(f"[Crypto] ERROR: public key not found at {pubkey_path}")
            print("         Start the server first (it generates the keypair), "
                  "then copy server_public.pem to clients.")
            raise SystemExit(1)
        with open(pubkey_path, "rb") as f:
            server_public_pem = f.read()
        print(f"[Crypto] Loaded server public key from {pubkey_path}")
    # ─────────────────────────────────────────────────────────────────────────────

    start_client(
        args.csv,
        args.id,
        args.server,
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        num_fl_rounds=args.rounds,
        use_crypto=args.crypto,
        server_public_pem=server_public_pem,
    )