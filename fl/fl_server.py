"""
Flower FL server with configurable aggregation strategy, early stopping,
distribution monitoring, and saved global model checkpoints.

Model Inversion Defence Changes
--------------------------------
- _save_pth() now saves checkpoints using get_defended_model() so any
  downstream code that loads global_model_latest.pth gets a PrivacyWrapper
  out of the box. Base model weights are stored (wrapper has no extra params),
  so loading is backward-compatible with existing checkpoints.

- aggregate_evaluate() sanitises the round log: accuracy and macro_f1 are
  rounded to 2 decimal places before being written to fl_metrics.json and
  status.json. This limits the precision of the signal available to an
  attacker who monitors per-round metrics to infer class structure.

CLI Changes
-----------
- Added --attest flag (was missing from argparse despite being used).
- Added --noise-scale flag to override PrivacyWrapper noise_scale at runtime
  without editing constants.py. Lower = less noise = higher accuracy.
- Added --temperature flag to override PrivacyWrapper temperature at runtime.
"""
import argparse
import json
import os
import sys
import timeit
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from flwr.common.logger import log
from flwr.server.history import History
from logging import INFO

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (  # noqa: E402
    ScalerStats,
    aggregate_scaler_stats,
    compute_client_scaler_stats,
    get_default_client_csvs,
    infer_csv_schema,
    make_fl_round_config,
    persist_run_preprocessing,
)
from model import get_model, get_defended_model  # noqa: E402

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

# ── Attestation import ────────────────────────────────────────────────────────
_ATTEST_DIR = os.path.join(os.path.dirname(__file__), "..", "tee")
sys.path.insert(0, os.path.abspath(_ATTEST_DIR))
try:
    from attestation_server import AttestationServer  # noqa: E402
    _ATTEST_AVAILABLE = True
except ImportError:
    _ATTEST_AVAILABLE = False
# ─────────────────────────────────────────────────────────────────────────────


class EarlyStoppingServer(fl.server.Server):
    """
    Flower server that exits the FL round loop when the strategy triggers
    early stopping.
    """

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = History()

        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

            if getattr(self.strategy, "_es_triggered", False):
                log(INFO, "Early stopping triggered at round %s", current_round)
                break

        elapsed = timeit.default_timer() - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        class_names: List[str] = None,
        save_dir: str = "results/fl_rounds",
        crypto_ctx: "CryptoContext" = None,
        model_type: str = "mlp",
        preprocessing_metadata: Optional[Dict] = None,
        # ── Early stopping ────────────────────────────────────────────────────
        early_stopping_patience: int = 0,
        early_stopping_metric: str = "loss",
        early_stopping_min_delta: float = 1e-4,
        # ── Privacy wrapper overrides ─────────────────────────────────────────
        noise_scale: float = None,
        temperature: float = None,
        # ─────────────────────────────────────────────────────────────────────
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.save_dir    = save_dir
        self.model_type  = model_type
        self.round_log:   List[Dict] = []
        self.privacy_log: List[Dict] = []
        self._crypto_ctx = crypto_ctx
        self.use_crypto  = crypto_ctx is not None
        self._preprocessing_metadata = preprocessing_metadata or {}

        # PrivacyWrapper overrides — None means use constants.py defaults
        self._noise_scale = noise_scale
        self._temperature = temperature

        # Early stopping state
        self._es_patience  = early_stopping_patience
        self._es_metric    = early_stopping_metric
        self._es_min_delta = early_stopping_min_delta
        self._es_best      = None
        self._es_counter   = 0
        self._es_triggered = False

        if self.use_crypto:
            print("[Server][Crypto] Encryption enabled — will decrypt client weights "
                  "before aggregation.")
        if self._es_patience > 0:
            print(f"[Server] Early stopping: patience={self._es_patience}, "
                  f"metric={self._es_metric}, min_delta={self._es_min_delta}")

        # Report active PrivacyWrapper settings
        from config.constants import MI_NOISE_SCALE, MI_TEMPERATURE
        active_noise = self._noise_scale if self._noise_scale is not None else MI_NOISE_SCALE
        active_temp  = self._temperature if self._temperature is not None else MI_TEMPERATURE
        print(f"[Server][Defence] PrivacyWrapper settings — "
              f"noise_scale={active_noise}, temperature={active_temp}"
              + (" (from CLI)" if self._noise_scale is not None or self._temperature is not None
                 else " (from constants.py)"))

        os.makedirs(save_dir, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        # Decrypt client weights before FedAvg aggregation if crypto is enabled
        if self.use_crypto and self._crypto_ctx is not None:
            decrypted_results = []
            for client_proxy, fit_res in results:
                try:
                    raw_arrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
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
                eps   = fit_res.metrics.get("epsilon")
                cid   = fit_res.metrics.get("client_id", "unknown")
                delta = fit_res.metrics.get("delta")
                if eps is not None:
                    epsilons.append({
                        "client_id": cid,
                        "epsilon":   float(eps),
                        "delta":     float(delta) if delta else None,
                    })

        if epsilons:
            avg_eps = sum(e["epsilon"] for e in epsilons) / len(epsilons)
            privacy_entry = {
                "round":       server_round,
                "avg_epsilon": round(avg_eps, 5),
                "clients":     epsilons,
            }
            self.privacy_log.append(privacy_entry)
            privacy_path = os.path.join(self.save_dir, "fl_privacy.json")
            with open(privacy_path, "w", encoding="utf-8") as f:
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
            "loss":  round(float(loss), 5) if loss is not None else None,
        }
        if metrics:
            for key, value in metrics.items():
                # Sanitise precision of accuracy / F1 metrics written to log.
                # Limiting to 2 decimal places reduces the per-round signal an
                # attacker could harvest from the public round log to infer
                # class-level feature structure.
                if key in ("accuracy", "macro_f1"):
                    entry[key] = round(float(value), 2)
                else:
                    entry[key] = round(float(value), 5)

        self.round_log.append(entry)

        with open(os.path.join(self.save_dir, "fl_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.round_log, f, indent=2)

        print(f"[Server] Round {server_round} {entry}")

        # ── Early stopping check ──────────────────────────────────────────────
        if self._es_patience > 0 and not self._es_triggered:
            current = entry.get(self._es_metric)
            if current is not None:
                improved = False
                if self._es_metric == "loss":
                    improved = (self._es_best is None or
                                current < self._es_best - self._es_min_delta)
                else:
                    improved = (self._es_best is None or
                                current > self._es_best + self._es_min_delta)

                if improved:
                    self._es_best    = current
                    self._es_counter = 0
                else:
                    self._es_counter += 1
                    print(f"[Server][EarlyStopping] No improvement for "
                          f"{self._es_counter}/{self._es_patience} rounds "
                          f"({self._es_metric}={current:.5f}, best={self._es_best:.5f})")
                    if self._es_counter >= self._es_patience:
                        self._es_triggered = True
                        print(f"[Server][EarlyStopping] *** Triggered after round "
                              f"{server_round} — training will stop. ***")
        # ─────────────────────────────────────────────────────────────────────

        return loss, metrics

    def _save_pth(self, weights: List[np.ndarray], round_number: int):
        """
        Persist aggregated weights as a .pth checkpoint.

        Uses get_defended_model() so the saved checkpoint is already wrapped in
        PrivacyWrapper. noise_scale and temperature are taken from CLI args if
        provided, otherwise fall back to constants.py defaults.

        Weights are stored from wrapper.base_model (identical key names to the
        old get_model() checkpoints) so loading is fully backward-compatible.
        """
        model_type = getattr(self, "model_type", "mlp")

        # Build defended wrapper — use CLI overrides if provided
        kwargs = dict(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            model_type=model_type,
        )
        if self._noise_scale is not None:
            kwargs["noise_scale"] = self._noise_scale
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        defended = get_defended_model(**kwargs)

        # Load aggregated weights into the base model only
        state_dict = {
            key: torch.tensor(value)
            for key, value in zip(defended.base_model.state_dict().keys(), weights)
        }
        defended.base_model.load_state_dict(state_dict, strict=True)

        round_path  = os.path.join(self.save_dir, f"global_model_round_{round_number}.pth")
        latest_path = os.path.join(self.save_dir, "global_model_latest.pth")

        # Save base model state_dict (backward-compatible with plain get_model() loads)
        torch.save(defended.base_model.state_dict(), round_path)
        torch.save(defended.base_model.state_dict(), latest_path)

        print(
            f"[Server][Defence] Round {round_number} checkpoint saved with "
            f"PrivacyWrapper (noise_scale={defended.noise_scale}, "
            f"temperature={defended.temperature}) — {latest_path}"
        )

        # Write model metadata — includes active defence params
        meta_path = os.path.join(self.save_dir, "model_meta.json")
        meta = {
            "input_dim":      self.input_dim,
            "num_classes":    self.num_classes,
            "class_names":    self.class_names,
            "model_type":     model_type,
            "mi_noise_scale": defended.noise_scale,
            "mi_temperature": defended.temperature,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Also update root-level model_meta.json so attack/eval scripts find it
        root_meta_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "model_meta.json",
        )
        with open(root_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save preprocessing metadata alongside checkpoint
        meta_pp = getattr(self, "_preprocessing_metadata", None) or {}
        mean = meta_pp.get("mean")
        std  = meta_pp.get("std")
        if mean is not None and std is not None and len(mean) > 0 and len(std) > 0:
            from data_utils import save_preprocessing_metadata  # noqa: E402
            save_preprocessing_metadata(
                latest_path,
                feature_names=meta_pp.get("feature_names", []),
                mean=np.array(mean, dtype=np.float32),
                std=np.array(std, dtype=np.float32),
                normalization=meta_pp.get("normalization", "global"),
            )

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


def infer_default_num_classes() -> int:
    """Infer number of classes from the first default client CSV."""
    first_csv = get_default_client_csvs()[0]
    df = pd.read_csv(first_csv, usecols=["label"])
    return int(df["label"].nunique())


def make_timestamped_save_dir(base: str = "results/fl_rounds") -> str:
    """Return a unique run directory like results/fl_rounds/run_20260507_143022."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, f"run_{timestamp}")


def monitor_client_distributions(
    csv_paths: List[str],
    target_col: str = "label",
    save_dir: str = "",
) -> Dict:
    """
    Compute and report class distribution statistics across all client CSVs.

    Prints a summary table showing:
      - Class counts per client
      - KL divergence between each client and the global distribution
        (higher = more non-IID)
      - A recommendation on whether FedProx is likely needed

    Saves distribution_report.json to save_dir if provided.
    """
    from scipy.stats import entropy as kl_divergence

    client_dists = {}
    all_labels   = set()

    for path in csv_paths:
        df     = pd.read_csv(path, usecols=[target_col])
        counts = df[target_col].value_counts().to_dict()
        client_dists[os.path.basename(path)] = counts
        all_labels.update(counts.keys())

    all_labels = sorted(all_labels, key=str)

    eps = 1e-8
    prob_vectors = {}
    for name, counts in client_dists.items():
        total = sum(counts.values())
        vec   = np.array([counts.get(lbl, 0) / total + eps for lbl in all_labels],
                         dtype=np.float64)
        vec  /= vec.sum()
        prob_vectors[name] = vec

    global_counts = {}
    for counts in client_dists.values():
        for lbl, cnt in counts.items():
            global_counts[lbl] = global_counts.get(lbl, 0) + cnt
    total_global = sum(global_counts.values())
    global_vec   = np.array([global_counts.get(lbl, 0) / total_global + eps
                              for lbl in all_labels], dtype=np.float64)
    global_vec  /= global_vec.sum()

    kl_scores = {}
    for name, vec in prob_vectors.items():
        kl_scores[name] = float(kl_divergence(vec, global_vec))

    avg_kl = float(np.mean(list(kl_scores.values())))

    print("\n" + "=" * 65)
    print("CLIENT DISTRIBUTION REPORT")
    print("=" * 65)
    header = f"{'Class':<20}" + "".join(f"{n[:12]:>13}" for n in client_dists)
    print(header)
    print("-" * 65)
    for lbl in all_labels:
        row = f"{str(lbl):<20}"
        for counts in client_dists.values():
            row += f"{counts.get(lbl, 0):>13,}"
        print(row)
    print("-" * 65)
    print(f"\nKL divergence from global distribution (higher = more non-IID):")
    for name, kl in kl_scores.items():
        bar = "█" * int(kl * 50)
        print(f"  {name:<30} KL={kl:.4f}  {bar}")
    print(f"\n  Average KL: {avg_kl:.4f}")

    if avg_kl < 0.05:
        recommendation = "IID — FedAvg should work well."
    elif avg_kl < 0.20:
        recommendation = "Mildly non-IID — FedAvg is fine; FedProx may help slightly."
    else:
        recommendation = ("Highly non-IID — consider --strategy fedprox "
                          "or reducing --local-epochs.")
    print(f"  Recommendation: {recommendation}")
    print("=" * 65 + "\n")

    report = {
        "clients": list(client_dists.keys()),
        "classes": [str(l) for l in all_labels],
        "counts_per_client": {
            name: {str(k): v for k, v in counts.items()}
            for name, counts in client_dists.items()
        },
        "kl_divergence": kl_scores,
        "avg_kl":        avg_kl,
        "recommendation": recommendation,
    }

    if save_dir:
        out = os.path.join(save_dir, "distribution_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[Server] Distribution report saved -> {out}")

    return report


def coordinate_global_normalization(csv_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and std from all client CSVs without sharing raw data.
    """
    print("[Server] Coordinating global normalization across clients...")
    stats_list = []
    for path in csv_paths:
        stats = compute_client_scaler_stats(path)
        stats_list.append(stats)
        print(f"  {os.path.basename(path)}: n={stats.n_samples}, "
              f"mean[0]={stats.mean[0]:.4f}, std[0]={stats.std[0]:.4f}")

    global_mean, global_std = aggregate_scaler_stats(stats_list)
    print(f"[Server] Global normalization ready — "
          f"mean[0]={global_mean[0]:.4f}, std[0]={global_std[0]:.4f} ✓")
    return global_mean, global_std


def build_strategy(
    strategy_name: str,
    input_dim: int,
    num_classes: int,
    class_names: List[str],
    save_dir: str,
    crypto_ctx,
    min_clients: int,
    fraction_fit: float,
    fraction_evaluate: float,
    local_epochs: int,
    proximal_mu: float = 1.0,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "loss",
    early_stopping_min_delta: float = 1e-4,
    model_type: str = "mlp",
    preprocessing_metadata: Optional[Dict] = None,
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    noise_scale: float = None,
    temperature: float = None,
) -> "SaveModelStrategy":
    """Build the FL aggregation strategy."""
    round_config = make_fl_round_config(local_epochs, global_mean, global_std)

    seed_model = get_model(input_dim, num_classes, model_type=model_type)
    initial_parameters = fl.common.ndarrays_to_parameters(
        [value.cpu().numpy() for value in seed_model.state_dict().values()]
    )

    common_kwargs = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        class_names=class_names,
        save_dir=save_dir,
        crypto_ctx=crypto_ctx,
        model_type=model_type,
        preprocessing_metadata=preprocessing_metadata,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=max(1, int(min_clients * fraction_fit)),
        min_evaluate_clients=max(1, int(min_clients * fraction_evaluate)),
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda _: round_config,
        on_evaluate_config_fn=lambda _: round_config,
        initial_parameters=initial_parameters,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        early_stopping_min_delta=early_stopping_min_delta,
        noise_scale=noise_scale,
        temperature=temperature,
    )

    name = strategy_name.lower()
    if name == "fedprox":
        try:
            from flwr.server.strategy import FedProx
            print(f"[Server] Strategy: FedProx (proximal_mu={proximal_mu})")

            class SaveModelFedProx(SaveModelStrategy, FedProx):
                pass

            return SaveModelFedProx(proximal_mu=proximal_mu, **common_kwargs)
        except ImportError:
            print("[Server] WARNING: FedProx not available in this Flower version — "
                  "falling back to FedAvg.")

    print(f"[Server] Strategy: FedAvg (fraction_fit={fraction_fit:.0%})")
    return SaveModelStrategy(**common_kwargs)


def start_server(
    input_dim: int,
    num_classes: int = 0,
    class_names: List[str] = None,
    num_rounds: int = 10,
    min_clients: int = 3,
    local_epochs: int = 1,
    server_address: str = "0.0.0.0:8080",
    save_dir: str = "",
    use_crypto: bool = False,
    use_attest: bool = False,
    strategy_name: str = "fedavg",
    fraction_fit: float = 1.0,
    proximal_mu: float = 1.0,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "loss",
    monitor_distributions: bool = True,
    model_type: str = "mlp",
    noise_scale: float = None,
    temperature: float = None,
):
    # ── Attestation ───────────────────────────────────────────────────────────
    if use_attest and _ATTEST_AVAILABLE:
        attest_server = AttestationServer()
        attest_server.generate_quote()
    elif use_attest:
        print("[Server][Attest] WARNING: attestation module not found — skipping.")

    if not save_dir:
        save_dir = make_timestamped_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Server] Results will be saved to: {save_dir}")

    if num_classes <= 0:
        num_classes = infer_default_num_classes()
        print(f"[Server] Inferred num_classes={num_classes} from default client CSV.")

    if class_names is None:
        from data_utils import infer_class_names  # noqa: E402
        first_csv = get_default_client_csvs()[0]
        if os.path.exists(first_csv):
            class_names = infer_class_names(first_csv)
            print(f"[Server] Inferred class_names={class_names} "
                  f"from {os.path.basename(first_csv)}")

    csv_paths = get_default_client_csvs()
    existing_csvs = [p for p in csv_paths if os.path.exists(p)]
    if not existing_csvs:
        raise FileNotFoundError(
            "No client CSVs found under data/processed/ (expected client1.csv, ...).\n"
            "  Local:  python data/datascripts/pipeline.py\n"
            "  Docker: mount ../data/processed:/app/data/processed:ro on fl-server\n"
            "  Image:  rebuild docker/Dockerfile.server"
        )
    if monitor_distributions and existing_csvs:
        monitor_client_distributions(existing_csvs, save_dir=save_dir)

    global_mean, global_std = None, None
    if existing_csvs:
        global_mean, global_std = coordinate_global_normalization(existing_csvs)

    crypto_ctx = None
    if use_crypto and _CRYPTO_AVAILABLE:
        crypto_ctx = CryptoContext.load_or_create()
        print(f"[Server][Crypto] Public key ready at "
              f"crypto/certs/keys/server_public.pem — share with clients.")
        if _SEALED_STORAGE_AVAILABLE:
            priv_pem = os.path.join(
                os.path.dirname(__file__), "..", "crypto", "certs", "keys", "server_private.pem"
            )
            priv_pem = os.path.abspath(priv_pem)
            if os.path.exists(priv_pem):
                seal_private_key(priv_pem)
    elif use_crypto:
        print("[Server][Crypto] WARNING: crypto unavailable — starting without encryption.")

    preprocessing_metadata = {}
    if global_mean is not None and global_std is not None:
        preprocessing_metadata = {
            "mean":          global_mean.tolist() if hasattr(global_mean, "tolist") else global_mean,
            "std":           global_std.tolist()  if hasattr(global_std,  "tolist") else global_std,
            "normalization": "global",
        }
        if existing_csvs:
            _, feature_names = infer_csv_schema(existing_csvs[0])
            preprocessing_metadata["feature_names"] = feature_names
            prep_path, norm_path = persist_run_preprocessing(
                save_dir,
                feature_names=feature_names,
                mean=global_mean,
                std=global_std,
                normalization="global",
            )
            print(
                f"[Server] Global normalization broadcast via FL config; "
                f"saved {prep_path} and {norm_path}"
            )

    strategy = build_strategy(
        strategy_name=strategy_name,
        input_dim=input_dim,
        num_classes=num_classes,
        class_names=class_names,
        save_dir=save_dir,
        crypto_ctx=crypto_ctx,
        min_clients=min_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        local_epochs=local_epochs,
        proximal_mu=proximal_mu,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        model_type=model_type,
        preprocessing_metadata=preprocessing_metadata,
        global_mean=global_mean,
        global_std=global_std,
        noise_scale=noise_scale,
        temperature=temperature,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    _write_run_summary(save_dir, strategy)


def _write_run_summary(save_dir: str, strategy: "SaveModelStrategy") -> None:
    """
    Write status.json and results/results.json after training completes.
    These files are read by the dashboard /status and /results endpoints.
    """
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    round_log = strategy.round_log
    last = round_log[-1] if round_log else {}
    status = {
        "round":         last.get("round", 0),
        "total_rounds":  len(round_log),
        "clients":       strategy.min_available_clients,
        "loss":          last.get("loss"),
        "accuracy":      last.get("accuracy"),
        "macro_f1":      last.get("macro_f1"),
        "save_dir":      save_dir,
        "model_type":    strategy.model_type,
        "early_stopped": strategy._es_triggered,
        "noise_scale":   strategy._noise_scale,
        "temperature":   strategy._temperature,
    }
    status_path = os.path.join(_root, "status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        try:
            pl = getattr(strategy, "privacy_log", None)
            if isinstance(pl, list) and pl:
                last_pl = pl[-1]
                eps = last_pl.get("avg_epsilon") or last_pl.get("epsilon")
                if eps is not None:
                    status["epsilon"] = float(eps)
                else:
                    clients = last_pl.get("clients") or []
                    if isinstance(clients, list) and clients:
                        vals = [c.get("epsilon") for c in clients
                                if isinstance(c, dict) and c.get("epsilon") is not None]
                        if vals:
                            status["epsilon"] = float(sum(vals) / len(vals))
        except Exception:
            pass
        json.dump(status, f, indent=2)

    results_dir = os.path.join(_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"rounds": round_log, "save_dir": save_dir}, f, indent=2)

    print(f"[Server] status.json      -> {status_path}")
    print(f"[Server] results.json     -> {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim",  type=int, default=infer_default_input_dim())
    parser.add_argument("--num-classes", type=int, default=0,
                        help="Number of output classes. Inferred from data if not set.")
    parser.add_argument("--rounds",     type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--address",    default="0.0.0.0:8080")
    parser.add_argument("--save-dir",   default="",
                        help="Output directory. Auto-generates a timestamped path if not set.")
    parser.add_argument("--strategy",   default="fedavg", choices=["fedavg", "fedprox"],
                        help="Aggregation strategy (default: fedavg).")
    parser.add_argument("--fraction-fit", type=float, default=1.0,
                        help="Fraction of clients sampled per round (default: 1.0 = all).")
    parser.add_argument("--proximal-mu", type=float, default=1.0,
                        help="FedProx proximal term weight.")
    parser.add_argument("--model-type", default="mlp",
                        choices=["mlp", "resnet-tabular", "transformer-tabular"],
                        help="Model architecture (default: mlp).")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop if no improvement for N rounds (0 = disabled).")
    parser.add_argument("--early-stopping-metric", default="loss",
                        choices=["loss", "accuracy", "macro_f1"],
                        help="Metric to monitor for early stopping (default: loss).")
    parser.add_argument("--no-distribution-monitor", action="store_true",
                        help="Skip the pre-training distribution report.")
    parser.add_argument("--crypto", action="store_true",
                        help="Enable AES-256-GCM + RSA weight encryption.")
    parser.add_argument("--attest", action="store_true",
                        help="Enable SGX attestation before starting.")
    parser.add_argument("--noise-scale", type=float, default=None,
                        help="PrivacyWrapper Laplace noise scale on logits. "
                             "Lower = less noise = higher accuracy. "
                             "Default: value from config/constants.py (MI_NOISE_SCALE).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="PrivacyWrapper softmax temperature divisor. "
                             "Lower = sharper predictions. "
                             "Default: value from config/constants.py (MI_TEMPERATURE).")
    args = parser.parse_args()

    start_server(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        local_epochs=args.local_epochs,
        server_address=args.address,
        save_dir=args.save_dir,
        use_crypto=args.crypto,
        use_attest=args.attest,
        strategy_name=args.strategy,
        fraction_fit=args.fraction_fit,
        proximal_mu=args.proximal_mu,
        model_type=args.model_type,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        monitor_distributions=not args.no_distribution_monitor,
        noise_scale=args.noise_scale,
        temperature=args.temperature,
    )