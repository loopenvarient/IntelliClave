"""
Flower FL server with configurable aggregation strategy, early stopping,
distribution monitoring, and saved global model checkpoints.

Flower 1.6 compatibility notes
-------------------------------
This file contains custom glue that is tightly coupled to the Flower 1.6 API.
The following points are the most likely to break on a Flower upgrade:

  1. FitRes constructor (aggregate_fit):
       flwr.common.FitRes(status, parameters, num_examples, metrics)
       In Flower ≥ 1.8 the field order and names may change.
       Search for "_FitRes(" to find all construction sites.

  2. fl.client.start_numpy_client (fl_client.py):
       Deprecated in Flower 1.8 in favour of fl.client.start_client with
       a FlowerClient wrapper. The NumPyClient shim still works in 1.8 but
       will be removed in 2.x.

  3. SaveModelFedProx multiple inheritance (build_strategy):
       class SaveModelFedProx(SaveModelStrategy, FedProx) relies on Python
       MRO and Flower's FedProx.__init__ signature. If FedProx gains new
       required kwargs this will raise TypeError at runtime.

  4. on_fit_config_fn / fit_metrics_aggregation_fn kwargs:
       Passed as **kwargs to FedAvg.__init__. Flower 1.7+ renamed some of
       these. If you upgrade, run the test suite and check for TypeError on
       strategy construction.

  Upgrade path: pin flwr==1.6.0 in requirements.txt (already done).
  Before upgrading, run: python -c "import flwr; print(flwr.__version__)"
  and review the Flower changelog for breaking changes in the target version.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np

# ── Flower version guard ──────────────────────────────────────────────────────
# This codebase is tested against Flower 1.6.0. The custom glue in
# aggregate_fit (FitRes construction) and build_strategy (FedProx MRO) is
# known to work on 1.6.x. Warn loudly on any other version so the developer
# knows to re-test the glue points listed above before deploying.
_FLWR_REQUIRED = (1, 6)
try:
    _flwr_ver = tuple(int(x) for x in fl.__version__.split(".")[:2])
    if _flwr_ver != _FLWR_REQUIRED:
        import warnings as _warnings
        _warnings.warn(
            f"[fl_server] Flower version mismatch: expected {_FLWR_REQUIRED[0]}.{_FLWR_REQUIRED[1]}.x, "
            f"got {fl.__version__}. Custom glue (FitRes constructor, FedProx MRO, "
            f"start_numpy_client) may break. Review FLOWER_UPGRADE.md before proceeding.",
            stacklevel=2,
        )
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import torch
from flwr.common import Metrics

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (  # noqa: E402
    ScalerStats,
    aggregate_scaler_stats,
    compute_client_scaler_stats,
    get_default_client_csvs,
    infer_csv_schema,
)
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
        num_classes: int,
        class_names: List[str] = None,
        save_dir: str = "results/fl_rounds",
        crypto_ctx: "CryptoContext" = None,
        model_type: str = "mlp",
        # ── Robust aggregation (Issue 4 fix) ──────────────────────────────────
        # "fedavg"       : standard weighted average (default, no robustness)
        # "krum"         : Krum — selects the single most central update
        # "multi-krum"   : Multi-Krum — selects m best updates and averages
        # "trimmed-mean" : coordinate-wise trimmed mean
        # "median"       : coordinate-wise median (most robust)
        robust_agg: str = "fedavg",
        byzantine_fraction: float = 0.33,
        # ─────────────────────────────────────────────────────────────────────
        # ── Early stopping ────────────────────────────────────────────────────
        early_stopping_patience: int = 0,
        early_stopping_metric: str = "loss",
        early_stopping_min_delta: float = 1e-4,
        # ── Checkpoint resume ─────────────────────────────────────────────────
        round_offset: int = 0,
        target_rounds: int = 0,
        strategy_name: str = "fedavg",
        # ─────────────────────────────────────────────────────────────────────
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.save_dir    = save_dir
        self.model_type  = model_type
        self._round_offset   = int(round_offset)
        self._target_rounds  = int(target_rounds) if target_rounds else 0
        self._strategy_name  = strategy_name
        self.round_log:   List[Dict] = []
        self.privacy_log: List[Dict] = []
        self._crypto_ctx = crypto_ctx
        self.use_crypto  = crypto_ctx is not None

        # ── Robust aggregation config ─────────────────────────────────────────
        self.robust_agg         = robust_agg.lower()
        self.byzantine_fraction = byzantine_fraction
        # Import here so the module is available at aggregation time
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(__file__))
            from robust_aggregation import (  # noqa: E402
                krum as _krum,
                trimmed_mean as _trimmed_mean,
                coordinate_median as _coord_median,
                auto_f as _auto_f,
                auto_beta as _auto_beta,
            )
            self._krum          = _krum
            self._trimmed_mean  = _trimmed_mean
            self._coord_median  = _coord_median
            self._auto_f        = _auto_f
            self._auto_beta     = _auto_beta
            self._robust_available = True
        except ImportError as e:
            print(f"[Server] WARNING: robust_aggregation not available ({e}) "
                  "— falling back to FedAvg.")
            self._robust_available = False

        if self.robust_agg != "fedavg":
            if self._robust_available:
                print(f"[Server] Robust aggregation: {self.robust_agg} "
                      f"(byzantine_fraction={byzantine_fraction:.0%})")
                if self.robust_agg in ("krum", "multi-krum"):
                    from robust_aggregation import (  # noqa: E402
                        MIN_CLIENTS_FOR_KRUM,
                        auto_f,
                        check_krum_viable,
                    )
                    n_min = kwargs.get("min_available_clients", 0)
                    if n_min and n_min < MIN_CLIENTS_FOR_KRUM:
                        f = auto_f(n_min, byzantine_fraction)
                        _, msg = check_krum_viable(n_min, f)
                        raise ValueError(
                            f"Krum requires min_available_clients >= "
                            f"{MIN_CLIENTS_FOR_KRUM} (got {n_min}). {msg}"
                        )
            else:
                print(f"[Server] WARNING: robust_agg={self.robust_agg} requested "
                      "but module unavailable — using FedAvg.")
        # ─────────────────────────────────────────────────────────────────────

        # ── Per-client convergence tracking (Issue 2 fix) ─────────────────────
        # {client_id: [{"round": r, "accuracy": a, "loss": l, "macro_f1": f}, ...]}
        # Populated from fit_res.metrics in aggregate_fit each round.
        # Clients must include "client_id" in their fit metrics for attribution.
        self._client_metrics_log: Dict[str, List[Dict]] = {}

        # Per-round convergence diagnostics: accuracy delta, oscillation flag.
        # Written to fl_convergence.json after every round.
        self._convergence_log: List[Dict] = []
        self._prev_global_accuracy: Optional[float] = None
        # ─────────────────────────────────────────────────────────────────────

        # Early stopping state
        self._es_patience  = early_stopping_patience
        self._es_metric    = early_stopping_metric
        self._es_min_delta = early_stopping_min_delta
        self._es_best      = None          # best metric value seen so far
        self._es_counter   = 0             # rounds without improvement
        self._es_triggered = False

        if self.use_crypto:
            print("[Server][Crypto] Encryption enabled — will decrypt client weights "
                  "before aggregation.")
        if self._es_patience > 0:
            print(f"[Server] Early stopping: patience={self._es_patience}, "
                  f"metric={self._es_metric}, min_delta={self._es_min_delta}")
        os.makedirs(save_dir, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        # Decrypt client weights before aggregation if crypto is enabled
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
                              f"client weights decrypted")
                except Exception as e:
                    print(f"[Server][Crypto] ERROR: decryption failed "
                          f"({type(e).__name__}: {e})")
                    raise
                decrypted_results.append((client_proxy, fit_res))
            results = decrypted_results

        # ── Robust aggregation (Issue 4 fix) ──────────────────────────────────
        # When a robust strategy is selected, extract weight arrays from all
        # clients, run the robust aggregator, and repack into Flower's format
        # before calling super(). super() then sees pre-aggregated weights from
        # a single "virtual client" and passes them through unchanged.
        if (self.robust_agg != "fedavg"
                and self._robust_available
                and len(results) >= 2):

            n = len(results)
            all_weights = [
                fl.common.parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]
            sizes = [fit_res.num_examples for _, fit_res in results]

            try:
                if self.robust_agg in ("krum", "multi-krum"):
                    f = self._auto_f(n, self.byzantine_fraction)
                    m = 1 if self.robust_agg == "krum" else max(1, n - f)

                    # Viability check — warn if Krum is degenerate
                    try:
                        from robust_aggregation import check_krum_viable
                        viable, msg = check_krum_viable(n, f)
                        if not viable:
                            print(f"[Server][Robust] WARNING: {msg}")
                            print(f"[Server][Robust] Falling back to coordinate_median "
                                  f"which is robust at any client count.")
                            aggregated_weights = self._coord_median(all_weights)
                            print(f"[Server][Robust] Round {server_round} -- "
                                  f"Coordinate Median (fallback, n={n})")
                        else:
                            aggregated_weights = self._krum(all_weights, f=f, m=m)
                            print(f"[Server][Robust] Round {server_round} -- "
                                  f"Krum (n={n}, f={f}, m={m})")
                    except ImportError:
                        aggregated_weights = self._krum(all_weights, f=f, m=m)
                        print(f"[Server][Robust] Round {server_round} -- "
                              f"Krum (n={n}, f={f}, m={m})")

                elif self.robust_agg == "trimmed-mean":
                    beta = self._auto_beta(n, self.byzantine_fraction)
                    aggregated_weights = self._trimmed_mean(all_weights, beta=beta)
                    print(f"[Server][Robust] Round {server_round} — "
                          f"Trimmed Mean (n={n}, beta={beta})")

                elif self.robust_agg == "median":
                    aggregated_weights = self._coord_median(all_weights)
                    print(f"[Server][Robust] Round {server_round} — "
                          f"Coordinate Median (n={n})")

                else:
                    raise ValueError(f"Unknown robust_agg: {self.robust_agg}")

                # Replace each client's parameters with the robust aggregate
                # so super().aggregate_fit() sees identical weights and simply
                # returns them (weighted average of identical arrays = same array)
                from flwr.common import FitRes as _FitRes
                robust_params = fl.common.ndarrays_to_parameters(aggregated_weights)
                results = [
                    (
                        proxy,
                        _FitRes(
                            status=fit_res.status,
                            parameters=robust_params,
                            num_examples=fit_res.num_examples,
                            metrics=fit_res.metrics,
                        ),
                    )
                    for proxy, fit_res in results
                ]

            except (ValueError, Exception) as e:
                print(f"[Server][Robust] WARNING: {self.robust_agg} failed "
                      f"({e}) — falling back to FedAvg for this round.")
        # ─────────────────────────────────────────────────────────────────────

        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated)
            np.savez(os.path.join(self.save_dir, f"round_{server_round}.npz"), *weights)
            self._save_pth(weights, server_round)

        # ── Per-client metrics + privacy log ──────────────────────────────────
        # Extract accuracy, loss, macro_f1, and epsilon from each client's
        # fit_res.metrics. Clients tag their metrics with "client_id" so we
        # can attribute results without relying on positional ordering.
        epsilons = []
        for _, fit_res in results:
            if not (hasattr(fit_res, "metrics") and fit_res.metrics):
                continue
            m   = fit_res.metrics
            cid = str(m.get("client_id", "unknown"))

            # Per-client accuracy/loss tracking
            client_entry = {
                "round":     server_round,
                "accuracy":  round(float(m["accuracy"]), 5) if "accuracy" in m else None,
                "loss":      round(float(m["loss"]),     5) if "loss"     in m else None,
                "macro_f1":  round(float(m["macro_f1"]), 5) if "macro_f1" in m else None,
            }
            if cid not in self._client_metrics_log:
                self._client_metrics_log[cid] = []
            self._client_metrics_log[cid].append(client_entry)

            # Privacy log (DP clients only)
            eps   = m.get("epsilon")
            delta = m.get("delta")
            if eps is not None:
                epsilons.append({
                    "client_id": cid,
                    "epsilon":   float(eps),
                    "delta":     float(delta) if delta is not None else None,
                })

        # Print per-client accuracy table for this round
        if self._client_metrics_log:
            accs = {
                cid: entries[-1]["accuracy"]
                for cid, entries in self._client_metrics_log.items()
                if entries and entries[-1]["round"] == server_round
                   and entries[-1]["accuracy"] is not None
            }
            if accs:
                acc_str = "  ".join(
                    f"client_{cid}={v:.4f}" for cid, v in sorted(accs.items())
                )
                print(f"[Server][PerClient] Round {server_round} fit accuracy — {acc_str}")

        # Write fl_client_metrics.json after every round (incremental)
        client_metrics_path = os.path.join(self.save_dir, "fl_client_metrics.json")
        with open(client_metrics_path, "w") as f:
            json.dump(self._client_metrics_log, f, indent=2)

        # Privacy log
        if epsilons:
            avg_eps = sum(e["epsilon"] for e in epsilons) / len(epsilons)
            privacy_entry = {
                "round":       server_round,
                "avg_epsilon": round(avg_eps, 5),
                "clients":     epsilons,
            }
            self.privacy_log.append(privacy_entry)
            privacy_path = os.path.join(self.save_dir, "fl_privacy.json")
            with open(privacy_path, "w") as f:
                json.dump(self.privacy_log, f, indent=2)
            print(
                f"[Server][DP] Round {server_round} "
                f"avg_ε={avg_eps:.4f} — saved to fl_privacy.json"
            )
        # ─────────────────────────────────────────────────────────────────────

        return aggregated, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if failures:
            print(
                f"[Server] Round {server_round}: {len(failures)} client(s) failed or "
                f"timed out — aggregating from {len(results)} response(s)."
            )

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        logical_round = server_round + self._round_offset
        entry = {
            "round":         server_round,
            "logical_round": logical_round,
            "loss": round(float(loss), 5) if loss is not None else None,
        }
        if metrics:
            entry.update({key: round(float(value), 5) for key, value in metrics.items()})
        self.round_log.append(entry)

        with open(os.path.join(self.save_dir, "fl_metrics.json"), "w") as f:
            json.dump(self.round_log, f, indent=2)

        print(f"[Server] Round {logical_round} (server_round={server_round}) {entry}")

        # ── Round checkpoint (resume after crash) ─────────────────────────────
        if self._target_rounds > 0:
            try:
                from fl_checkpoint import save_round_checkpoint  # noqa: E402

                ckpt_path = save_round_checkpoint(
                    self.save_dir,
                    completed_round=logical_round,
                    target_rounds=self._target_rounds,
                    metrics=entry,
                    strategy_name=self._strategy_name,
                    model_type=self.model_type,
                )
                print(f"[Server][Checkpoint] Saved round {logical_round} -> {ckpt_path}")
            except Exception as exc:
                print(f"[Server][Checkpoint] WARNING: could not save checkpoint ({exc})")
        # ─────────────────────────────────────────────────────────────────────

        # ── Convergence diagnostics (Issue 2 fix) ─────────────────────────────
        # Track per-round accuracy delta and flag oscillation.
        # Oscillation: the sign of the accuracy delta flips between consecutive
        # rounds — a reliable signal that FedAvg is struggling with the current
        # data heterogeneity or learning rate.
        global_acc = entry.get("accuracy")
        if global_acc is not None:
            delta = (
                round(global_acc - self._prev_global_accuracy, 5)
                if self._prev_global_accuracy is not None else None
            )

            # Oscillation: previous delta and current delta have opposite signs
            oscillating = False
            if len(self._convergence_log) >= 2 and delta is not None:
                prev_delta = self._convergence_log[-1].get("accuracy_delta")
                if prev_delta is not None and prev_delta != 0.0:
                    oscillating = (delta * prev_delta) < 0

            # Per-client accuracy spread: max - min across clients this round
            client_accs_this_round = [
                entries[-1]["accuracy"]
                for entries in self._client_metrics_log.values()
                if entries
                   and entries[-1]["round"] == server_round
                   and entries[-1]["accuracy"] is not None
            ]
            client_spread = (
                round(max(client_accs_this_round) - min(client_accs_this_round), 5)
                if len(client_accs_this_round) >= 2 else None
            )
            client_min = (
                round(min(client_accs_this_round), 5)
                if client_accs_this_round else None
            )

            conv_entry = {
                "round":            server_round,
                "global_accuracy":  round(global_acc, 5),
                "accuracy_delta":   delta,
                "oscillating":      oscillating,
                "client_acc_spread": client_spread,
                "client_acc_min":   client_min,
                "client_count":     len(client_accs_this_round),
            }
            self._convergence_log.append(conv_entry)

            conv_path = os.path.join(self.save_dir, "fl_convergence.json")
            with open(conv_path, "w") as f:
                json.dump(self._convergence_log, f, indent=2)

            # Print convergence status
            delta_str  = f"Δ={delta:+.4f}" if delta is not None else "Δ=n/a"
            spread_str = f"spread={client_spread:.4f}" if client_spread is not None else ""
            osc_warn   = " ⚠ OSCILLATING" if oscillating else ""
            print(
                f"[Server][Convergence] Round {server_round} "
                f"global_acc={global_acc:.4f} {delta_str} "
                f"{spread_str}{osc_warn}"
            )
            if oscillating:
                print(
                    f"[Server][Convergence] *** Accuracy oscillating — "
                    f"consider --strategy fedprox or reducing --local-epochs ***"
                )

            self._prev_global_accuracy = global_acc
        # ─────────────────────────────────────────────────────────────────────

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
        # Use the stored model_type so the correct architecture is rebuilt
        model_type = getattr(self, "model_type", "mlp")
        model = get_model(self.input_dim, self.num_classes, model_type=model_type)
        state_dict = {
            key: torch.tensor(value)
            for key, value in zip(model.state_dict().keys(), weights)
        }
        model.load_state_dict(state_dict, strict=True)

        round_path  = os.path.join(self.save_dir, f"global_model_round_{round_number}.pth")
        latest_path = os.path.join(self.save_dir, "global_model_latest.pth")
        torch.save(model.state_dict(), round_path)
        torch.save(model.state_dict(), latest_path)

        # Write model metadata — includes model_type so evaluate/dashboard
        # can reconstruct the correct architecture without guessing
        meta_path = os.path.join(self.save_dir, "model_meta.json")
        meta = {
            "input_dim":   self.input_dim,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "model_type":  model_type,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Seal checkpoints — remove plaintext by default (closes white-box FS read)
        if _SEALED_STORAGE_AVAILABLE:
            _remove_plain = os.environ.get(
                "SEAL_REMOVE_PLAINTEXT", "true"
            ).lower() in ("1", "true", "yes")
            seal_model_checkpoint(round_path, remove_plaintext=_remove_plain)
            seal_model_checkpoint(latest_path, remove_plaintext=_remove_plain)


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
    Call this before training to understand your data heterogeneity.
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
    n_classes  = len(all_labels)

    # Build probability vectors (with Laplace smoothing to avoid log(0))
    eps = 1e-8
    prob_vectors = {}
    for name, counts in client_dists.items():
        total = sum(counts.values())
        vec   = np.array([counts.get(lbl, 0) / total + eps for lbl in all_labels],
                         dtype=np.float64)
        vec  /= vec.sum()
        prob_vectors[name] = vec

    # Global distribution
    global_counts = {}
    for counts in client_dists.values():
        for lbl, cnt in counts.items():
            global_counts[lbl] = global_counts.get(lbl, 0) + cnt
    total_global = sum(global_counts.values())
    global_vec   = np.array([global_counts.get(lbl, 0) / total_global + eps
                              for lbl in all_labels], dtype=np.float64)
    global_vec  /= global_vec.sum()

    # KL divergences
    kl_scores = {}
    for name, vec in prob_vectors.items():
        kl_scores[name] = float(kl_divergence(vec, global_vec))

    avg_kl = float(np.mean(list(kl_scores.values())))

    # Print summary
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
        "avg_kl": avg_kl,
        "recommendation": recommendation,
    }

    if save_dir:
        out = os.path.join(save_dir, "distribution_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Server] Distribution report saved -> {out}")

    return report


def coordinate_global_normalization(csv_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and std from all client CSVs without sharing raw data.

    Each client computes its own mean/std from its training split only.
    These summary statistics are aggregated here using a weighted pooled formula.
    The resulting global_mean and global_std are broadcast back to clients
    via load_csv_data(global_mean=..., global_std=...).

    This ensures all clients scale their features consistently, which improves
    FL convergence — especially when client distributions differ.
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


def _make_fit_config_fn(local_epochs: int, compress_updates: bool, top_k_fraction: float):
    """Build on_fit_config_fn with optional Top-K compression flags for clients."""

    def _fn(server_round: int) -> dict:
        cfg = {"local_epochs": local_epochs}
        if compress_updates:
            cfg["compress_updates"] = True
            cfg["top_k_fraction"] = float(top_k_fraction)
        return cfg

    return _fn


def _write_training_status_marker(
    root: str,
    *,
    training_active: bool,
    save_dir: str = "",
    round: int = 0,
    total_rounds: int = 0,
) -> None:
    """Update status.json with training lifecycle fields for the dashboard."""
    _config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    if _config_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(_config_dir))
    from runtime_paths import read_json_runtime, write_json_runtime  # noqa: E402

    status: Dict = {}
    try:
        status = read_json_runtime("status.json")
    except FileNotFoundError:
        status = {}
    cadence = float(os.environ.get("EXPECTED_TRAINING_CADENCE_HOURS", "24"))
    status.update({
        "training_active": training_active,
        "expected_training_cadence_hours": cadence,
        "save_dir": save_dir or status.get("save_dir", ""),
        "round": round,
        "total_rounds": total_rounds,
    })
    if not training_active:
        status["last_training_completed_at"] = datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    write_json_runtime("status.json", status)


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
    robust_agg: str = "fedavg",
    byzantine_fraction: float = 0.33,
    round_timeout: float = 0.0,
    initial_parameters=None,
    round_offset: int = 0,
    target_rounds: int = 0,
    compress_updates: bool = False,
    top_k_fraction: float = 0.1,
) -> "SaveModelStrategy":
    """
    Build the FL aggregation strategy.

    strategy_name       : "fedavg" | "fedprox"
    robust_agg          : "fedavg" | "krum" | "multi-krum" | "trimmed-mean" | "median"
                          Controls the Byzantine-robust aggregation algorithm.
                          Independent of strategy_name — FedProx + Krum is valid.
    byzantine_fraction  : assumed fraction of Byzantine clients (default 0.33).
                          Used to auto-compute f (Krum) and beta (Trimmed Mean).
    proximal_mu         : FedProx proximal term (only used with fedprox)
    fraction_fit        : fraction of clients per round (1.0 = all)
    early_stopping_patience : stop if no improvement for N rounds (0 = off)
    early_stopping_metric   : "loss" | "accuracy" | "macro_f1"
    """
    common_kwargs = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        class_names=class_names,
        save_dir=save_dir,
        crypto_ctx=crypto_ctx,
        model_type=model_type,
        robust_agg=robust_agg,
        byzantine_fraction=byzantine_fraction,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=max(1, int(min_clients * fraction_fit)),
        min_evaluate_clients=max(1, int(min_clients * fraction_evaluate)),
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=_make_fit_config_fn(
            local_epochs, compress_updates, top_k_fraction
        ),
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        early_stopping_min_delta=early_stopping_min_delta,
        round_offset=round_offset,
        target_rounds=target_rounds or 0,
        strategy_name=strategy_name,
    )

    if initial_parameters is not None:
        common_kwargs["initial_parameters"] = initial_parameters

    if round_timeout and round_timeout > 0:
        common_kwargs["fit_timeout"] = float(round_timeout)
        common_kwargs["evaluate_timeout"] = float(round_timeout)
        print(f"[Server] Client round timeout: {round_timeout:.0f}s (stragglers dropped)")

    if compress_updates:
        print(
            f"[Server] Update compression: Top-K sparsification "
            f"(fraction={top_k_fraction:.0%} per layer)"
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
    local_epochs: int = 3,
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
    robust_agg: str = "fedavg",
    byzantine_fraction: float = 0.33,
    round_timeout: float = 0.0,
    resume: bool = False,
    compress_updates: bool = False,
    top_k_fraction: float = 0.1,
):
    # Auto-generate a timestamped save directory if none given
    if not save_dir:
        save_dir = make_timestamped_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Server] Results will be saved to: {save_dir}")

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _write_training_status_marker(
        _root,
        training_active=True,
        save_dir=save_dir,
        round=0,
        total_rounds=num_rounds,
    )

    from fl_checkpoint import resolve_resume_plan  # noqa: E402

    resume_plan = resolve_resume_plan(save_dir, num_rounds, resume)
    run_rounds = resume_plan["run_rounds"]
    if run_rounds <= 0 and resume_plan["resumed"]:
        print("[Server] Nothing to train — loading last checkpoint for summary only.")
        strategy = build_strategy(
            strategy_name=strategy_name,
            input_dim=input_dim if input_dim > 0 else infer_default_input_dim(),
            num_classes=num_classes if num_classes > 0 else infer_default_num_classes(),
            class_names=class_names,
            save_dir=save_dir,
            crypto_ctx=None,
            min_clients=min_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_fit,
            local_epochs=local_epochs,
            proximal_mu=proximal_mu,
            model_type=model_type,
            robust_agg=robust_agg,
            byzantine_fraction=byzantine_fraction,
            target_rounds=num_rounds,
        )
        _write_run_summary(save_dir, strategy)
        return

    # Infer num_classes from data if not explicitly provided
    if num_classes <= 0:
        num_classes = infer_default_num_classes()
        print(f"[Server] Inferred num_classes={num_classes} from default client CSV.")

    # Distribution monitoring — run before training starts
    csv_paths = get_default_client_csvs()
    existing_csvs = [p for p in csv_paths if os.path.exists(p)]
    if monitor_distributions and existing_csvs:
        monitor_client_distributions(existing_csvs, save_dir=save_dir)

    # ── Mutual attestation (Issue 6 fix) ──────────────────────────────────────
    # Server generates its own quote and writes attestation.json before
    # accepting any client connections. Clients verify this quote before
    # connecting, and publish their own quotes for the server to verify.
    if use_attest:
        try:
            _ATT_DIR = os.path.join(os.path.dirname(__file__), "..", "tee", "attestation")
            sys.path.insert(0, os.path.abspath(_ATT_DIR))
            from attestation_integration import AttestationServer as _AttServer
            att_server = _AttServer()
            att_server.attest()
            print("[Server][Attestation] Server attestation complete.")
        except Exception as e:
            print(f"[Server][Attestation] WARNING: attestation failed ({e}) — continuing.")
    # ─────────────────────────────────────────────────────────────────────────
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
        robust_agg=robust_agg,
        byzantine_fraction=byzantine_fraction,
        round_timeout=round_timeout,
        initial_parameters=resume_plan["initial_parameters"],
        round_offset=resume_plan["round_offset"],
        target_rounds=num_rounds,
        compress_updates=compress_updates,
        top_k_fraction=top_k_fraction,
    )

    server_config_kwargs = {"num_rounds": run_rounds}
    if round_timeout and round_timeout > 0:
        try:
            server_config_kwargs["round_timeout"] = float(round_timeout)
        except TypeError:
            pass  # older Flower builds omit round_timeout on ServerConfig

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(**server_config_kwargs),
        strategy=strategy,
    )

    # Write status.json and results/results.json so the dashboard has data
    _write_run_summary(save_dir, strategy)


def _write_run_summary(save_dir: str, strategy: "SaveModelStrategy") -> None:
    """
    Write status.json and results/results.json after training completes.
    These files are read by the dashboard /status and /results endpoints.

    results.json now includes:
      - rounds:            per-round global aggregate metrics (existing)
      - client_metrics:    per-client per-round accuracy/loss/f1 (new)
      - convergence:       per-round accuracy delta + oscillation flags (new)
      - convergence_summary: overall diagnosis across the full run (new)
    """
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ── Convergence summary ───────────────────────────────────────────────────
    conv_log = getattr(strategy, "_convergence_log", [])
    oscillating_rounds = [e["round"] for e in conv_log if e.get("oscillating")]
    n_oscillating = len(oscillating_rounds)

    # Per-client final accuracy and trend
    client_metrics_log = getattr(strategy, "_client_metrics_log", {})
    client_summary = {}
    for cid, entries in client_metrics_log.items():
        accs = [e["accuracy"] for e in entries if e.get("accuracy") is not None]
        if accs:
            trend = "improving" if len(accs) >= 2 and accs[-1] > accs[0] else \
                    "degrading" if len(accs) >= 2 and accs[-1] < accs[0] else "flat"
            client_summary[cid] = {
                "final_accuracy":   round(accs[-1], 5),
                "initial_accuracy": round(accs[0],  5),
                "min_accuracy":     round(min(accs), 5),
                "max_accuracy":     round(max(accs), 5),
                "trend":            trend,
                "rounds_tracked":   len(accs),
            }

    convergence_summary = {
        "total_rounds":        len(conv_log),
        "oscillating_rounds":  oscillating_rounds,
        "n_oscillating":       n_oscillating,
        "oscillation_rate":    round(n_oscillating / len(conv_log), 4) if conv_log else 0.0,
        "diagnosis": (
            "OSCILLATING — FedAvg is struggling with data heterogeneity. "
            "Consider --strategy fedprox or reducing --local-epochs."
            if n_oscillating > len(conv_log) * 0.3 else
            "CONVERGING — accuracy is improving monotonically or near-monotonically."
            if n_oscillating == 0 else
            "MOSTLY CONVERGING — minor oscillation detected, monitor further."
        ),
        "client_summary": client_summary,
    }
    # ─────────────────────────────────────────────────────────────────────────

    # status.json — top-level run summary
    round_log = strategy.round_log
    last = round_log[-1] if round_log else {}

    # Derive epsilon from privacy log if available (last round, max across clients)
    epsilon_val = None
    if strategy.privacy_log:
        last_privacy = strategy.privacy_log[-1]
        client_epsilons = [c["epsilon"] for c in last_privacy.get("clients", [])
                           if c.get("epsilon") is not None]
        if client_epsilons:
            epsilon_val = round(max(client_epsilons), 4)

    # Build clients array for the dashboard ClientPanel
    # min_available_clients is an int — wrap it into the array format the
    # frontend expects: [{id, status, samples}, ...]
    n_clients = strategy.min_available_clients
    client_summary_list = []
    for cid, entries in client_metrics_log.items():
        accs = [e["accuracy"] for e in entries if e.get("accuracy") is not None]
        client_summary_list.append({
            "id":      cid,
            "status":  "ready",
            "samples": entries[-1].get("train_size", 0) if entries else 0,
            "final_accuracy": round(accs[-1], 4) if accs else None,
        })
    # If no per-client data yet, generate placeholder entries
    if not client_summary_list:
        client_summary_list = [
            {"id": str(i + 1), "status": "ready", "samples": 0}
            for i in range(n_clients)
        ]

    _cadence = float(os.environ.get("EXPECTED_TRAINING_CADENCE_HOURS", "24"))
    _completed_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    status = {
        "round":        last.get("round", 0),
        "total_rounds": len(round_log),
        "clients":      client_summary_list,   # array, not int
        "epsilon":      epsilon_val,           # cumulative epsilon (max across clients)
        "loss":         last.get("loss"),
        "accuracy":     last.get("accuracy"),
        "macro_f1":     last.get("macro_f1"),
        "save_dir":     save_dir,
        "model_type":   strategy.model_type,
        "early_stopped": strategy._es_triggered,
        "training_active": False,
        "last_training_completed_at": _completed_at,
        "expected_training_cadence_hours": _cadence,
        "convergence_diagnosis": convergence_summary["diagnosis"],
        "oscillating_rounds":    oscillating_rounds,
    }
    _config_dir = os.path.join(_root, "config")
    if _config_dir not in sys.path:
        sys.path.insert(0, _config_dir)
    from runtime_paths import write_json_runtime  # noqa: E402

    status_path = write_json_runtime("status.json", status)

    # results/results.json — full round history + per-client + convergence
    results_dir = os.path.join(_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "rounds":               round_log,
                "client_metrics":       client_metrics_log,
                "convergence":          conv_log,
                "convergence_summary":  convergence_summary,
                "save_dir":             save_dir,
            },
            f, indent=2,
        )

    print(f"[Server] status.json      -> {status_path} (mirrored to repo root)")
    print(f"[Server] results.json     -> {results_path}")
    print(f"[Server] Convergence: {convergence_summary['diagnosis']}")
    if oscillating_rounds:
        print(f"[Server] Oscillating rounds: {oscillating_rounds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=infer_default_input_dim())
    parser.add_argument("--num-classes", type=int, default=0,
                        help="Number of output classes. Inferred from data if not set.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--save-dir", default="",
                        help="Output directory. Auto-generates a timestamped path if not set.")
    parser.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"],
                        help="Aggregation strategy (default: fedavg).")
    parser.add_argument("--fraction-fit", type=float, default=1.0,
                        help="Fraction of clients sampled per round (default: 1.0 = all). "
                             "Set < 1.0 to tolerate client dropout.")
    parser.add_argument("--proximal-mu", type=float, default=1.0,
                        help="FedProx proximal term weight (only used with --strategy fedprox).")
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
    parser.add_argument(
        "--robust-agg", default="fedavg",
        choices=["fedavg", "krum", "multi-krum", "trimmed-mean", "median"],
        help="Byzantine-robust aggregation algorithm (default: fedavg). "
             "krum/multi-krum: select most central update(s). "
             "trimmed-mean: coordinate-wise trimmed mean. "
             "median: coordinate-wise median.",
    )
    parser.add_argument(
        "--byzantine-fraction", type=float, default=0.33,
        help="Assumed fraction of Byzantine clients for auto-tuning f/beta "
             "(default: 0.33). Used by krum, multi-krum, and trimmed-mean.",
    )
    parser.add_argument(
        "--attest",
        action="store_true",
        help="Enable SGX mutual attestation. Server generates a quote before "
             "accepting clients; clients must publish their own quotes.",
    )
    parser.add_argument(
        "--crypto",
        action="store_true",
        help="Enable AES-256-GCM + RSA weight encryption.",
    )
    parser.add_argument(
        "--round-timeout",
        type=float,
        default=0.0,
        help="Per-client fit/evaluate timeout in seconds (0 = wait indefinitely). "
             "Slow clients are dropped and the round proceeds with responders.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from fl_server_checkpoint.json in --save-dir after a crash.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable Top-K sparsification of client weight updates.",
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=0.1,
        help="Fraction of largest weights kept per layer when --compress is set.",
    )
    args = parser.parse_args()
    from robust_aggregation import validate_robust_agg_cli  # noqa: E402
    validate_robust_agg_cli(args.robust_agg, args.min_clients)
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
        robust_agg=args.robust_agg,
        byzantine_fraction=args.byzantine_fraction,
        round_timeout=args.round_timeout,
        resume=args.resume,
        compress_updates=args.compress,
        top_k_fraction=args.top_k_fraction,
    )
