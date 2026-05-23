"""
fl/robust_aggregation.py

Byzantine-robust aggregation algorithms for federated learning.

Replaces plain FedAvg when clients cannot be fully trusted.

Algorithms
----------
Krum (Blanchard et al., 2017)
    Selects the single client update whose sum of squared distances to its
    f+1 nearest neighbours is smallest, where f is the assumed number of
    Byzantine clients. Multi-Krum selects the m best candidates and averages
    them, giving a tunable trade-off between robustness and accuracy.

    Guarantee: if f < n/2, Krum converges to the true gradient direction
    even when f clients submit arbitrary updates.

    Complexity: O(n^2 * d) per round, where n = clients, d = parameter count.

Trimmed Mean (Yin et al., 2018)
    For each parameter dimension independently, sorts the n client values,
    removes the β lowest and β highest values (trim fraction β on each side),
    and averages the remaining n - 2β values.

    Guarantee: if β < n/2, the estimator is consistent under Byzantine
    attacks that affect at most β clients.

    Complexity: O(n * d * log n) per round.

Median (coordinate-wise)
    For each parameter dimension, takes the median across clients.
    Equivalent to Trimmed Mean with β = floor((n-1)/2).
    Most robust but highest variance — use when Byzantine fraction is unknown.

Usage
-----
    from fl.robust_aggregation import krum, trimmed_mean, coordinate_median

    # weights: List[List[np.ndarray]]  — one list of arrays per client
    # sizes:   List[int]               — number of training samples per client

    # Krum (select 1 best client)
    aggregated = krum(weights, f=1)

    # Multi-Krum (select m=2 best, average them)
    aggregated = krum(weights, f=1, m=2)

    # Trimmed Mean (trim 1 client from each end)
    aggregated = trimmed_mean(weights, beta=1)

    # Coordinate-wise Median
    aggregated = coordinate_median(weights)

Notes
-----
- All functions accept and return List[np.ndarray] (same format as FedAvg).
- Sizes are used only by weighted variants; Krum and Trimmed Mean ignore them
  because robustness requires treating all clients equally.
- These are simulation-mode implementations. The Flower server integration
  in fl_server.py calls these functions inside aggregate_fit().
"""

from __future__ import annotations

import sys

import numpy as np
from typing import List


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten(weights: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of weight arrays into a single 1-D vector."""
    return np.concatenate([w.ravel() for w in weights])


def _unflatten(flat: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
    """Reshape a flat vector back into the structure of template."""
    result = []
    offset = 0
    for w in template:
        size = w.size
        result.append(flat[offset: offset + size].reshape(w.shape))
        offset += size
    return result


def _pairwise_sq_distances(flat_weights: np.ndarray) -> np.ndarray:
    """
    Compute the n×n matrix of squared Euclidean distances between n flat
    weight vectors. flat_weights has shape (n, d).
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq_norms = (flat_weights ** 2).sum(axis=1, keepdims=True)  # (n, 1)
    gram     = flat_weights @ flat_weights.T                    # (n, n)
    dist_sq  = sq_norms + sq_norms.T - 2 * gram
    # Numerical noise can produce tiny negatives — clip to 0
    return np.maximum(dist_sq, 0.0)


# ── Krum ──────────────────────────────────────────────────────────────────────

def krum(
    weights: List[List[np.ndarray]],
    f: int,
    m: int = 1,
) -> List[np.ndarray]:
    """
    (Multi-)Krum aggregation.

    Parameters
    ----------
    weights : List[List[np.ndarray]]
        One list of weight arrays per client. All clients must have the same
        architecture (same shapes).
    f : int
        Number of assumed Byzantine clients. Must satisfy f < n/2.
        The algorithm selects from the n - f - 2 nearest neighbours of each
        client (standard Krum neighbourhood size).
    m : int
        Number of candidates to select and average (Multi-Krum).
        m=1 is standard Krum (most robust, highest variance).
        m=n is equivalent to FedAvg (no robustness).
        Typical choice: m = n - f.

    Returns
    -------
    List[np.ndarray]
        Aggregated weight arrays (same structure as one client's weights).

    Raises
    ------
    ValueError
        If f >= n/2 (Byzantine fraction too high for Krum to work) or
        if m > n - f (not enough honest candidates).
    """
    n = len(weights)
    if n < 2:
        return [w.copy() for w in weights[0]]

    if f >= n / 2:
        raise ValueError(
            f"Krum requires f < n/2. Got f={f}, n={n}. "
            "Reduce f or add more clients."
        )
    if m > n - f:
        raise ValueError(
            f"Multi-Krum m={m} > n-f={n-f}. "
            "Reduce m or increase the number of clients."
        )

    # Flatten all client weights into a matrix (n, d)
    flat = np.stack([_flatten(w) for w in weights])  # (n, d)
    dist_sq = _pairwise_sq_distances(flat)            # (n, n)

    # For each client i, sum distances to its (n - f - 2) nearest neighbours
    # (excluding itself). Standard Krum neighbourhood = n - f - 2.
    k = n - f - 2
    if k < 1:
        k = 1  # degenerate case: at least 1 neighbour

    scores = np.zeros(n)
    for i in range(n):
        # Distances from client i to all others, sorted ascending
        dists_i = np.sort(dist_sq[i])          # includes dist to self (0)
        # Skip the first entry (distance to self = 0), take next k
        scores[i] = dists_i[1: k + 1].sum()

    # Select the m clients with the lowest scores
    selected_idx = np.argsort(scores)[:m]

    # Average the selected candidates (uniform, not weighted by dataset size)
    selected_flat = flat[selected_idx]           # (m, d)
    aggregated_flat = selected_flat.mean(axis=0) # (d,)

    return _unflatten(aggregated_flat, weights[0])


# ── Trimmed Mean ──────────────────────────────────────────────────────────────

def trimmed_mean(
    weights: List[List[np.ndarray]],
    beta: int,
) -> List[np.ndarray]:
    """
    Coordinate-wise Trimmed Mean aggregation.

    For each parameter dimension, sorts the n client values, removes the
    beta lowest and beta highest, and averages the remaining n - 2*beta values.

    Parameters
    ----------
    weights : List[List[np.ndarray]]
        One list of weight arrays per client.
    beta : int
        Number of clients to trim from each end per dimension.
        Must satisfy 2*beta < n.

    Returns
    -------
    List[np.ndarray]
        Aggregated weight arrays.

    Raises
    ------
    ValueError
        If 2*beta >= n (trimming too aggressive).
    """
    n = len(weights)
    if n < 2:
        return [w.copy() for w in weights[0]]

    if 2 * beta >= n:
        raise ValueError(
            f"Trimmed Mean requires 2*beta < n. Got beta={beta}, n={n}. "
            "Reduce beta or add more clients."
        )

    # Stack into (n, d) matrix
    flat = np.stack([_flatten(w) for w in weights])  # (n, d)

    # Sort along the client axis (axis=0) for each dimension
    sorted_flat = np.sort(flat, axis=0)              # (n, d)

    # Trim beta from each end and average
    trimmed = sorted_flat[beta: n - beta, :]         # (n - 2*beta, d)
    aggregated_flat = trimmed.mean(axis=0)            # (d,)

    return _unflatten(aggregated_flat, weights[0])


# ── Coordinate-wise Median ────────────────────────────────────────────────────

def coordinate_median(
    weights: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Coordinate-wise Median aggregation.

    Equivalent to Trimmed Mean with beta = floor((n-1)/2).
    Most robust against Byzantine clients but highest variance.

    Parameters
    ----------
    weights : List[List[np.ndarray]]
        One list of weight arrays per client.

    Returns
    -------
    List[np.ndarray]
        Aggregated weight arrays.
    """
    flat = np.stack([_flatten(w) for w in weights])  # (n, d)
    aggregated_flat = np.median(flat, axis=0)         # (d,)
    return _unflatten(aggregated_flat, weights[0])


# ── Weighted FedAvg (reference) ───────────────────────────────────────────────

def fedavg(
    weights: List[List[np.ndarray]],
    sizes: List[int],
) -> List[np.ndarray]:
    """
    Standard weighted FedAvg. Included here so the attack simulation can
    import all aggregators from one place.

    Parameters
    ----------
    weights : List[List[np.ndarray]]
    sizes   : List[int]  — number of training samples per client

    Returns
    -------
    List[np.ndarray]
    """
    total = sum(sizes)
    result = [np.zeros_like(w) for w in weights[0]]
    for client_weights, n in zip(weights, sizes):
        for i, w in enumerate(client_weights):
            result[i] += w * (n / total)
    return result


# ── Client-count requirements ─────────────────────────────────────────────────

# Formal Krum guarantee for f=1 Byzantine tolerance: n >= 2f + 3 = 5.
MIN_CLIENTS_FOR_KRUM = 5


def validate_robust_agg_cli(
    robust_agg: str,
    n_clients: int,
    *,
    exit_on_error: bool = True,
) -> bool:
    """
    Refuse Krum / Multi-Krum when n < MIN_CLIENTS_FOR_KRUM.

    With n=3, auto_f(3)=0 so Krum picks a single update and provides zero
    Byzantine robustness. Trimmed-mean and median remain available at any n>=2.

    Returns True if the choice is allowed; exits with code 1 when not allowed
    and exit_on_error is True.
    """
    agg = (robust_agg or "fedavg").lower()
    if agg not in ("krum", "multi-krum"):
        return True

    f = auto_f(n_clients)
    viable, msg = check_krum_viable(n_clients, f)
    if viable:
        return True

    lines = [
        f"[RobustAgg] ERROR: --robust-agg {agg} requires at least "
        f"{MIN_CLIENTS_FOR_KRUM} clients (got {n_clients}).",
        f"  {msg}",
        "  Options:",
        f"    - Increase --clients / --min-clients to {MIN_CLIENTS_FOR_KRUM}+ "
        "(regenerate data: python data/datascripts/pipeline.py --n-clients 5)",
        "    - Use --robust-agg trimmed-mean or median (robust at n=3)",
        "    - Keep FedAvg (default) for the 3-client demo",
    ]
    print("\n".join(lines), file=sys.stderr)
    if exit_on_error:
        raise SystemExit(1)
    return False


# ── Auto-select beta / f from n_clients ──────────────────────────────────────

def auto_f(n_clients: int, assumed_byzantine_fraction: float = 0.33) -> int:
    """
    Compute a safe f (assumed Byzantine count) from the number of clients
    and an assumed Byzantine fraction.

    Default fraction 0.33 means we assume up to 1/3 of clients are malicious,
    which is the standard assumption in Byzantine-robust FL literature.

    Clamps to f < n/2 to satisfy Krum's requirement.

    IMPORTANT — minimum client count for meaningful Krum:
        Krum requires n >= 2f + 3 for formal Byzantine robustness.
        With f=1 this means n >= 5.
        At n=3, auto_f returns 0, so Krum degenerates to selecting the
        single "best" update with no Byzantine tolerance at all.
        Use check_krum_viable() to detect this before training.
    """
    f = int(n_clients * assumed_byzantine_fraction)
    return min(f, (n_clients - 1) // 2)


def check_krum_viable(n_clients: int, f: int) -> tuple:
    """
    Check whether Krum provides meaningful Byzantine robustness.

    Krum's formal guarantee requires n >= 2f + 3.
    Below this threshold f=0 and Krum just picks the single nearest
    update — it provides zero Byzantine tolerance.

    Returns (is_viable: bool, message: str).
    """
    min_required = 2 * f + 3
    if f == 0:
        return (
            False,
            f"Krum is DEGENERATE with {n_clients} clients: auto_f={f}, "
            f"so Krum selects the single best update with no Byzantine tolerance. "
            f"Formal robustness (f=1) requires n >= {2*1+3} = 5 clients. "
            f"Use --robust-agg trimmed-mean or median for small client counts, "
            f"or increase --clients to at least 5."
        )
    if n_clients < min_required:
        return (
            False,
            f"Krum has weak guarantees with {n_clients} clients and f={f}: "
            f"requires n >= 2f+3 = {min_required}. "
            f"Add {min_required - n_clients} more clients for full robustness."
        )
    return (True, f"Krum viable: n={n_clients}, f={f}, n >= 2f+3={min_required} satisfied.")


def auto_beta(n_clients: int, assumed_byzantine_fraction: float = 0.33) -> int:
    """
    Compute a safe beta (trim count per side) for Trimmed Mean.
    Clamps to 2*beta < n.
    """
    beta = int(n_clients * assumed_byzantine_fraction)
    return min(beta, (n_clients - 1) // 2)
