"""
fl/update_compression.py

Optional Top-K sparsification for federated weight updates.

Flower 1.6 legacy FedAvg does not expose a single compression_config flag; this
module implements the same pattern (sparse dense updates + config via
on_fit_config_fn) used in Flower compression tutorials.

Clients zero out all but the largest-magnitude values per tensor; the server
aggregates dense arrays as usual. Metrics report sparsity for bandwidth estimates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def topk_sparsify_arrays(
    arrays: List[np.ndarray],
    fraction: float = 0.1,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Keep the top `fraction` of values by absolute magnitude (per tensor).

    Parameters
    ----------
    arrays    : model weight arrays (one per layer)
    fraction  : fraction of elements to keep in (0, 1]

    Returns
    -------
    sparse_arrays, stats dict with compression_ratio and bytes_saved_estimate
    """
    fraction = float(fraction)
    if fraction <= 0.0 or fraction >= 1.0:
        return [a.copy() for a in arrays], {
            "compression_enabled": False,
            "compression_ratio":   1.0,
        }

    out = []
    total_el = 0
    nonzero_el = 0
    for arr in arrays:
        flat = arr.ravel().copy()
        n = flat.size
        total_el += n
        k = max(1, int(n * fraction))
        if k >= n:
            out.append(arr.copy())
            nonzero_el += n
            continue
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        flat[~mask] = 0.0
        nonzero_el += k
        out.append(flat.reshape(arr.shape))

    ratio = nonzero_el / total_el if total_el else 1.0
    # float32: saved fraction of payload (zeros still sent in dense mode)
    bytes_saved = (1.0 - ratio) * total_el * 4
    return out, {
        "compression_enabled": True,
        "top_k_fraction":      fraction,
        "compression_ratio":   round(ratio, 4),
        "nonzero_elements":    int(nonzero_el),
        "total_elements":      int(total_el),
        "bytes_saved_estimate": int(bytes_saved),
    }


def maybe_compress_updates(
    arrays: List[np.ndarray],
    config: dict,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Apply Top-K sparsification when config['compress_updates'] is true."""
    if not config or not config.get("compress_updates"):
        return arrays, {"compression_enabled": False}
    fraction = float(config.get("top_k_fraction", 0.1))
    return topk_sparsify_arrays(arrays, fraction=fraction)
