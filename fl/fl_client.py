"""
Flower FL client for one CSV dataset / organisation.
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
from data_utils import load_class_weights, load_csv_data, DatasetMetadata  # noqa: E402
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

# delta is computed from actual train_size after data loading (1 / n_train)
# ─────────────────────────────────────────────────────────────────────────────


class IntelliClaveClient(fl.client.NumPyClient):
    def __init__(
        self,
        csv_path: str,
        client_id: str,
        local_epochs: int = 3,
        learning_rate: float = 1e-3,
        model_type: str = "mlp",
        batch_size: int = 32,
        # ── DP parameters — all optional, default = DP off ───────────────────────
        use_dp: bool = False,
        target_epsilon: float = 10.0,
        max_grad_norm: float = 2.0,
        num_fl_rounds: int = 10,
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

        # ── DP accuracy optimisations ─────────────────────────────────────────
        # When DP is active three things help recover accuracy:
        #
        # 1. Larger batch size — Opacus clips and noises per-sample gradients,
        #    then averages them. More samples per batch → better signal-to-noise
        #    ratio before the noise is added. DP_BATCH_SIZE=64 vs default 32
        #    reduces the effective noise_std by ~√2 at the same ε.
        #
        # 2. Dropout = 0.0 — DP noise already acts as strong regularisation.
        #    Stacking Dropout(0.3) on top adds variance without adding privacy,
        #    compounding the accuracy cost. Disable it when DP is on.
        #
        # 3. Higher learning rate — the gradient signal must overcome DP noise.
        #    A 2× LR boost (1e-3 → 2e-3) helps the model learn faster per step
        #    without destabilising non-DP training.
        # ─────────────────────────────────────────────────────────────────────
        if use_dp:
            # Use the larger DP batch size from constants unless caller overrides
            from config.constants import DP_BATCH_SIZE  # noqa: E402
            effective_batch_size = max(batch_size, DP_BATCH_SIZE)
            effective_lr = learning_rate * 2.0   # 2× LR for DP training
            effective_dropout = 0.0              # disable dropout under DP
            if effective_batch_size != batch_size:
                print(
                    f"[Client {client_id}][DP] Batch size: {batch_size} → "
                    f"{effective_batch_size} (larger batch reduces DP noise_std by "
                    f"~{1 - (batch_size/effective_batch_size)**0.5:.0%})"
                )
            print(
                f"[Client {client_id}][DP] LR: {learning_rate} → {effective_lr:.4f} "
                f"(2× boost to overcome DP noise)"
            )
            print(
                f"[Client {client_id}][DP] Dropout: 0.3 → 0.0 "
                f"(DP noise acts as regularisation)"
            )
        else:
            effective_batch_size = batch_size
            effective_lr = learning_rate
            effective_dropout = 0.3

        # Data loading — use the effective (possibly larger) batch size
        self.train_loader, self.test_loader, self.metadata = load_csv_data(
            csv_path, batch_size=effective_batch_size
        )
        self.model = get_model(
            self.metadata.input_dim,
            self.metadata.num_classes,
            model_type=model_type,
            dropout=effective_dropout,
        ).to(self.device)
        class_weights = load_class_weights(
            num_classes=self.metadata.num_classes, device=self.device
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=effective_lr)

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

            # ── Pre-flight noise multiplier check ────────────────────────────
            # Opacus computes the noise multiplier from (ε, δ, C, epochs,
            # batch_size) together. At low ε the multiplier can be very high,
            # causing accuracy collapse. We surface this immediately so the
            # user can tune max_grad_norm before wasting a full training run.
            try:
                nm = float(self.optimizer.noise_multiplier)
                noise_std = nm * self.max_grad_norm
                if self.target_epsilon < 5.0 and nm > 3.0:
                    print(
                        f"\n[Client {self.cid}][DP] *** ACCURACY COLLAPSE WARNING ***\n"
                        f"  target_epsilon={self.target_epsilon} is low and "
                        f"noise_multiplier={nm:.2f} is very high.\n"
                        f"  noise_std = {nm:.2f} x {self.max_grad_norm} = {noise_std:.2f} "
                        f"-- gradient signal will be overwhelmed.\n"
                        f"  Expected outcome: accuracy near random (~{100//self.metadata.num_classes}%).\n"
                        f"  Fix: run clipping norm sweep first to find the optimal C:\n"
                        f"    python privacy/clipping_norm_sweep.py "
                        f"--epsilon {self.target_epsilon} "
                        f"--csv <your_csv>\n"
                        f"  Then re-run with --max-grad-norm <optimal_value>.\n"
                    )
                elif nm > 2.0:
                    print(
                        f"[Client {self.cid}][DP] NOTE: noise_multiplier={nm:.2f} "
                        f"(noise_std={noise_std:.2f}). "
                        f"Expect some accuracy degradation. "
                        f"Run privacy/clipping_norm_sweep.py to tune if needed."
                    )
            except AttributeError:
                pass
            # ─────────────────────────────────────────────────────────────────

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

        accuracy, macro_f1 = evaluate(self.model, self.test_loader, self.device)

        # Metrics dict — client_id always included for per-client tracking
        metrics = {
            "client_id": self.cid,
            "loss":      float(loss),
            "accuracy":  float(accuracy),
            "macro_f1":  float(macro_f1),
        }

        # Append epsilon to metrics if DP is active
        if self.use_dp and self.privacy_engine is not None:
            eps = self.privacy_engine.get_epsilon(delta=self.target_delta)
            metrics["epsilon"] = float(eps)
            metrics["delta"]   = float(self.target_delta)
            print(
                f"[Client {self.cid}][DP] "
                f"loss={loss:.4f} acc={accuracy:.4f} ε={eps:.4f}"
            )

        # Encrypt weights before sending to server if crypto is enabled
        outgoing_weights = self.get_parameters({})

        try:
            from update_compression import maybe_compress_updates  # noqa: E402

            outgoing_weights, comp_stats = maybe_compress_updates(
                outgoing_weights, config or {}
            )
            if comp_stats.get("compression_enabled"):
                metrics["compression_ratio"] = comp_stats.get("compression_ratio")
                print(
                    f"[Client {self.cid}][Compress] Top-K {comp_stats.get('top_k_fraction', 0):.0%} "
                    f"— {comp_stats.get('compression_ratio', 0):.1%} non-zero weights"
                )
        except ImportError:
            pass

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
    model_type: str = "mlp",
    local_epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    use_dp: bool = False,
    target_epsilon: float = 10.0,
    max_grad_norm: float = 2.0,
    num_fl_rounds: int = 10,
    use_crypto: bool = False,
    server_public_pem: bytes = None,
):
    fl.client.start_numpy_client(
        server_address=server_address,
        client=IntelliClaveClient(
            csv_path=csv_path,
            client_id=client_id,
            model_type=model_type,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
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
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for local optimizer (default: 1e-3).")
    parser.add_argument("--local-epochs", type=int, default=3,
                        help="Local training epochs per FL round (default: 3).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="DataLoader batch size (default: 32).")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy via Opacus.")
    parser.add_argument("--epsilon", type=float, default=10.0, help="Target epsilon (privacy budget). Default=10.0.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0,
                        help="Gradient clipping norm for DP-SGD. Default=2.0 (sweep-optimised for eps=10).")
    parser.add_argument("--model-type", default="mlp",
                        choices=["mlp", "resnet-tabular", "transformer-tabular"],
                        help="Model architecture (default: mlp).")
    parser.add_argument("--rounds", type=int, default=10, help="Total FL rounds — must match server --rounds. Default=10.")
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
        model_type=args.model_type,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        num_fl_rounds=args.rounds,
        use_crypto=args.crypto,
        server_public_pem=server_public_pem,
    )