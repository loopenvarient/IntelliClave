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

# ── Sealed storage import ─────────────────────────────────────────────────────
_SEALED_DIR = os.path.join(os.path.dirname(__file__), "..", "tee", "sealed_storage")
sys.path.insert(0, os.path.abspath(_SEALED_DIR))
try:
    from sealed_storage import seal_model_checkpoint, seal_private_key  # noqa: E402
    _SEALED_STORAGE_AVAILABLE = True
except ImportError:
    _SEALED_STORAGE_AVAILABLE = False
    print("[fl_server] WARNING: sealed_storage not found — checkpoints will not be sealed.")
# ─────────────────────────────────────────────────────────────────────────────
_CRYPTO_DIR = os.path.join(os.path.dirname(__file__), "..", "crypto", "certs")
sys.path.insert(0, os.path.abspath(_CRYPTO_DIR))
try:
    from crypto_context import CryptoContext  # noqa: E402
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    print("[fl_server] WARNING: crypto_context not found — running without encryption.")
# ─────────────────────────────────────────────────────────────────────────────


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        save_dir: str = "results/fl_rounds",
        crypto_ctx: "CryptoContext" = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.round_log: List[Dict] = []
        self.privacy_log: List[Dict] = []
        self._crypto_ctx = crypto_ctx
        self.use_crypto = crypto_ctx is not None
        if self.use_crypto:
            print("[Server][Crypto] Encryption enabled — will decrypt client weights "
                  "before aggregation.")
        os.makedirs(save_dir, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        # Decrypt client weights before FedAvg aggregation if crypto is enabled
        if self.use_crypto and self._crypto_ctx is not None:
            decrypted_results = []
            for client_proxy, fit_res in results:
                try:
                    raw_arrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    # Encrypted payload arrives as a single uint8 array
                    if len(raw_arrays) == 1 and raw_arrays[0].dtype == np.uint8:
                        payload = json.loads(raw_arrays[0].tobytes().decode())
                        plain_weights = self._crypto_ctx.decrypt_weights(payload)
                        from flwr.common import FitRes as _FitRes
                        fit_res = _FitRes(
                            status=fit_res.status,
                            parameters=fl.common.ndarrays_to_parameters(plain_weights),
                            num_examples=fit_res.num_examples,
                            metrics=fit_res.metrics,
                        )
                        print(f"[Server][Crypto] Round {server_round} — "
                              f"client weights decrypted ✅")
                except Exception as e:
                    print(f"[Server][Crypto] ERROR: decryption failed "
                          f"({type(e).__name__}: {e})")
                    raise
                decrypted_results.append((client_proxy, fit_res))
            results = decrypted_results

        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated)
            np.savez(os.path.join(self.save_dir, f"round_{server_round}.npz"), *weights)
            self._save_pth(weights, server_round)

        # Extract epsilon from client fit metrics and save to privacy log
        epsilons = []
        for _, fit_res in results:
            if hasattr(fit_res, "metrics") and fit_res.metrics:
                eps = fit_res.metrics.get("epsilon")
                cid = fit_res.metrics.get("client_id", "unknown")
                delta = fit_res.metrics.get("delta")
                if eps is not None:
                    epsilons.append({
                        "client_id": cid,
                        "epsilon": float(eps),
                        "delta": float(delta) if delta else None,
                    })

        if epsilons:
            avg_eps = sum(e["epsilon"] for e in epsilons) / len(epsilons)
            privacy_entry = {
                "round": server_round,
                "avg_epsilon": round(avg_eps, 5),
                "clients": epsilons,
            }
            self.privacy_log.append(privacy_entry)
            privacy_path = os.path.join(self.save_dir, "fl_privacy.json")
            with open(privacy_path, "w") as f:
                json.dump(self.privacy_log, f, indent=2)
            print(
                f"[Server][DP] Round {server_round} "
                f"avg_ε={avg_eps:.4f} — saved to fl_privacy.json"
            )

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

        # Seal checkpoints to the server enclave identity
        if _SEALED_STORAGE_AVAILABLE:
            seal_model_checkpoint(round_path)
            seal_model_checkpoint(latest_path)


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
    use_crypto: bool = False,
):
    crypto_ctx = None
    if use_crypto and _CRYPTO_AVAILABLE:
        crypto_ctx = CryptoContext.load_or_create()
        print(f"[Server][Crypto] Public key ready at "
              f"crypto/certs/keys/server_public.pem — share with clients.")
        # Seal the private key to the enclave so it can't be read outside
        if _SEALED_STORAGE_AVAILABLE:
            priv_pem = os.path.join(
                os.path.dirname(__file__), "..", "crypto", "certs", "keys", "server_private.pem"
            )
            priv_pem = os.path.abspath(priv_pem)
            if os.path.exists(priv_pem):
                seal_private_key(priv_pem)
    elif use_crypto:
        print("[Server][Crypto] WARNING: crypto unavailable — starting without encryption.")

    strategy = SaveModelStrategy(
        input_dim=input_dim,
        num_classes=6,
        save_dir=save_dir,
        crypto_ctx=crypto_ctx,
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
    parser.add_argument(
        "--crypto",
        action="store_true",
        help="Enable AES-256-GCM + RSA weight encryption.",
    )
    args = parser.parse_args()
    start_server(
        input_dim=args.input_dim,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        local_epochs=args.local_epochs,
        server_address=args.address,
        save_dir=args.save_dir,
        use_crypto=args.crypto,
    )
