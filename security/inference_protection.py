"""
security/inference_protection.py

Inference-time protections against model inversion attacks.

Two independent mechanisms:

1. Output Perturbation (Laplace mechanism)
   ─────────────────────────────────────────
   Adds calibrated Laplace noise to the post-softmax probability vector
   before returning any prediction. This is inference-time differential
   privacy — distinct from training-time DP-SGD.

   Formal guarantee: the output distribution of any two inputs that differ
   in one feature dimension changes by at most exp(ε_inf) under the
   perturbed mechanism, where ε_inf = sensitivity / noise_scale.

   Sensitivity of the softmax output vector:
     The L1 sensitivity of a probability simplex is 2 (moving one unit of
     probability mass from one class to another changes the L1 norm by 2).
     We use L1 sensitivity = 2.0 as the formal bound.

   Noise scale selection:
     noise_scale = sensitivity / ε_inf
     Default ε_inf = 4.0  →  noise_scale = 2.0 / 4.0 = 0.5
     This is the value validated by the calibration sweep in
     security/attacks/model_inversion.py to degrade cosine similarity
     below 0.60 while keeping label accuracy above 85%.

   After adding noise the vector is clipped to [0, 1] and re-normalised
   so it remains a valid probability distribution. The top-1 label is
   derived from the noisy vector, so the label itself is perturbed.

2. Lifetime Query Budget
   ──────────────────────
   Each API key is allocated a hard lifetime budget of N predictions
   (default 10 000). Once exhausted, all further /predict calls return
   HTTP 429 regardless of rate-limit state.

   The budget counts completed predictions, not requests. A request that
   fails validation (wrong feature count, etc.) does not consume budget.

   Storage backend is abstracted behind QueryBudgetStore so the in-process
   dict can be swapped for Redis without changing the calling code:

       # In-process (default, single worker):
       store = InMemoryBudgetStore()

       # Redis-backed (multi-worker production):
       store = RedisBudgetStore(redis_url="redis://localhost:6379")

   The store interface is:
       store.get_remaining(api_key: str) -> int
       store.consume(api_key: str) -> int   # returns remaining after consume
       store.reset(api_key: str) -> None

Usage in dashboard/backend/main.py:
    from security.inference_protection import (
        OutputPerturbation,
        InMemoryBudgetStore,
        QueryBudgetStore,
        LIFETIME_BUDGET_DEFAULT,
    )
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────

# L1 sensitivity of the softmax probability simplex.
# Moving probability mass from one class to another changes the L1 norm by 2.
SOFTMAX_L1_SENSITIVITY: float = 2.0

# Default inference-time privacy budget (ε_inf).
# noise_scale = SOFTMAX_L1_SENSITIVITY / INFERENCE_EPSILON_DEFAULT = 0.5
INFERENCE_EPSILON_DEFAULT: float = float(
    os.environ.get("INFERENCE_EPSILON", "4.0")
)

# Derived noise scale — can be overridden via INFERENCE_NOISE_SCALE env var.
# If set explicitly it takes precedence over the epsilon-derived value.
_env_noise_scale = os.environ.get("INFERENCE_NOISE_SCALE")
NOISE_SCALE_DEFAULT: float = (
    float(_env_noise_scale)
    if _env_noise_scale is not None
    else SOFTMAX_L1_SENSITIVITY / INFERENCE_EPSILON_DEFAULT
)

# Hard lifetime query budget per API key (number of completed predictions).
LIFETIME_BUDGET_DEFAULT: int = int(
    os.environ.get("LIFETIME_QUERY_BUDGET", "10000")
)


# ── Output Perturbation ───────────────────────────────────────────────────────

class OutputPerturbation:
    """
    Adds calibrated Laplace noise to a post-softmax probability vector.

    Parameters
    ----------
    noise_scale : float
        Scale parameter b of the Laplace distribution.
        b = L1_sensitivity / ε_inf.
        Smaller b → less noise → weaker privacy.
        Default: SOFTMAX_L1_SENSITIVITY / INFERENCE_EPSILON_DEFAULT.
    seed : int | None
        Optional RNG seed for reproducibility in tests.
        Leave None in production (fresh randomness every call).
    """

    def __init__(
        self,
        noise_scale: float = NOISE_SCALE_DEFAULT,
        seed: Optional[int] = None,
    ):
        if noise_scale <= 0:
            raise ValueError(f"noise_scale must be > 0, got {noise_scale}")
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    @property
    def epsilon_inf(self) -> float:
        """Inference-time ε derived from the current noise scale."""
        return SOFTMAX_L1_SENSITIVITY / self.noise_scale

    def perturb(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Add Laplace noise to a 1-D probability tensor and re-normalise.

        Parameters
        ----------
        probs : torch.Tensor, shape (num_classes,)
            Post-softmax probability vector. Must sum to ~1.

        Returns
        -------
        torch.Tensor, shape (num_classes,)
            Noisy, clipped, re-normalised probability vector.
            The returned tensor is always a valid probability distribution.
        """
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        noise = self._rng.laplace(loc=0.0, scale=self.noise_scale, size=probs_np.shape)
        noisy = probs_np + noise

        # Clip to [0, 1] — Laplace noise can push values negative or above 1
        noisy = np.clip(noisy, 0.0, 1.0)

        # Re-normalise to a valid probability simplex
        total = noisy.sum()
        if total < 1e-12:
            # Degenerate case: all mass clipped to zero — return uniform
            noisy = np.ones_like(noisy) / len(noisy)
        else:
            noisy = noisy / total

        return torch.tensor(noisy, dtype=probs.dtype)

    def perturb_batch(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Perturb a batch of probability vectors.

        Parameters
        ----------
        probs : torch.Tensor, shape (batch_size, num_classes)

        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
        """
        return torch.stack([self.perturb(row) for row in probs])

    def calibration_report(self, num_classes: int = 6) -> dict:
        """
        Return a summary of the noise calibration for logging / audit.
        """
        return {
            "mechanism":          "Laplace",
            "l1_sensitivity":     SOFTMAX_L1_SENSITIVITY,
            "noise_scale":        round(self.noise_scale, 6),
            "epsilon_inf":        round(self.epsilon_inf, 6),
            "num_classes":        num_classes,
            "expected_noise_std": round(self.noise_scale * np.sqrt(2), 6),
            "note": (
                "noise_scale = L1_sensitivity / epsilon_inf. "
                "Validated to degrade model-inversion cosine similarity "
                "below 0.60 while keeping label accuracy above 85%."
            ),
        }


# ── Query Budget Store (abstract interface) ───────────────────────────────────

class QueryBudgetStore(ABC):
    """
    Abstract interface for lifetime query budget tracking.

    Implementations must be thread-safe (or process-safe for Redis).
    """

    @abstractmethod
    def get_remaining(self, api_key: str) -> int:
        """Return the number of predictions remaining for api_key."""

    @abstractmethod
    def consume(self, api_key: str, budget: int = LIFETIME_BUDGET_DEFAULT) -> int:
        """
        Decrement the budget for api_key by 1.

        Parameters
        ----------
        api_key : str
        budget  : int
            Initial budget to allocate if api_key is new.

        Returns
        -------
        int
            Remaining budget after this consume. 0 means exhausted.

        Raises
        ------
        BudgetExhaustedError
            If the budget is already at 0 before this call.
        """

    @abstractmethod
    def reset(self, api_key: str, budget: int = LIFETIME_BUDGET_DEFAULT) -> None:
        """Reset the budget for api_key to budget."""


class BudgetExhaustedError(Exception):
    """Raised when a query budget is fully consumed."""

    def __init__(self, api_key: str, budget: int):
        self.api_key = api_key
        self.budget = budget
        super().__init__(
            f"Lifetime query budget exhausted for key '{api_key}' "
            f"(limit={budget}). No further predictions will be served."
        )


class InMemoryBudgetStore(QueryBudgetStore):
    """
    In-process dict-backed budget store.

    Suitable for single-worker deployments. In multi-worker deployments
    (uvicorn --workers N) each worker has its own store, so the effective
    budget is N × LIFETIME_BUDGET_DEFAULT. Use RedisBudgetStore for
    production multi-worker setups.
    """

    def __init__(self):
        # Maps api_key → remaining predictions
        self._budgets: dict[str, int] = {}
        # Lock-free for now; add threading.Lock if needed under high concurrency
        self._lock_available = False
        try:
            import threading
            self._lock = threading.Lock()
            self._lock_available = True
        except ImportError:
            pass

    def _acquire(self):
        if self._lock_available:
            self._lock.acquire()

    def _release(self):
        if self._lock_available:
            self._lock.release()

    def get_remaining(self, api_key: str) -> int:
        return self._budgets.get(api_key, LIFETIME_BUDGET_DEFAULT)

    def consume(self, api_key: str, budget: int = LIFETIME_BUDGET_DEFAULT) -> int:
        self._acquire()
        try:
            current = self._budgets.get(api_key, budget)
            if current <= 0:
                raise BudgetExhaustedError(api_key, budget)
            self._budgets[api_key] = current - 1
            return current - 1
        finally:
            self._release()

    def reset(self, api_key: str, budget: int = LIFETIME_BUDGET_DEFAULT) -> None:
        self._acquire()
        try:
            self._budgets[api_key] = budget
        finally:
            self._release()


class RedisBudgetStore(QueryBudgetStore):
    """
    Redis-backed budget store for multi-worker production deployments.

    Requires: pip install redis
    Set REDIS_URL env var to your Redis connection string.

    Keys are stored as: intelliclave:budget:<api_key>
    TTL is not set — budgets persist until explicitly reset.
    """

    _KEY_PREFIX = "intelliclave:budget:"

    def __init__(self, redis_url: str = None):
        redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            import redis as _redis
            self._client = _redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except ImportError:
            raise RuntimeError(
                "redis package not installed. Run: pip install redis"
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Redis at {redis_url}: {e}"
            )

    def _key(self, api_key: str) -> str:
        return f"{self._KEY_PREFIX}{api_key}"

    def get_remaining(self, api_key: str) -> int:
        val = self._client.get(self._key(api_key))
        return int(val) if val is not None else LIFETIME_BUDGET_DEFAULT

    def consume(self, api_key: str, budget: int = LIFETIME_BUDGET_DEFAULT) -> int:
        k = self._key(api_key)
        # SET NX initialises the key only if it doesn't exist
        self._client.set(k, budget, nx=True)
        remaining = self._client.decr(k)
        if remaining < 0:
            # Restore to 0 and raise — don't let it go negative
            self._client.set(k, 0)
            raise BudgetExhaustedError(api_key, budget)
        return remaining

    def reset(self, api_key: str, budget: int = LIFETIME_BUDGET_DEFAULT) -> None:
        self._client.set(self._key(api_key), budget)


# ── Rate limiting (sliding window) ────────────────────────────────────────────

class RateLimiter(ABC):
    @abstractmethod
    def check(self, client_key: str) -> None:
        """Raise RateLimitExceeded if the client exceeded the window limit."""


class RateLimitExceeded(Exception):
    def __init__(self, client_key: str, max_requests: int, window_seconds: int):
        self.client_key = client_key
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        super().__init__(
            f"Rate limit exceeded for '{client_key}': "
            f"max {max_requests} requests per {window_seconds}s."
        )


class InMemoryRateLimiter(RateLimiter):
    """Per-process sliding-window rate limiter (single-worker dev only)."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        import time
        from collections import defaultdict

        self._max = max_requests
        self._window = window_seconds
        self._time = time
        self._log: dict = defaultdict(list)

    def check(self, client_key: str) -> None:
        now = self._time.time()
        window_start = now - self._window
        self._log[client_key] = [
            t for t in self._log[client_key] if t > window_start
        ]
        if len(self._log[client_key]) >= self._max:
            raise RateLimitExceeded(client_key, self._max, self._window)
        self._log[client_key].append(now)


class RedisRateLimiter(RateLimiter):
    """
    Redis-backed sliding-window rate limiter (multi-worker / pod restart safe
    within the window; use together with RedisBudgetStore for lifetime caps).
    """

    _KEY_PREFIX = "intelliclave:ratelimit:"

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        redis_url: str = None,
    ):
        import time

        self._max = max_requests
        self._window = window_seconds
        self._time = time
        redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            import redis as _redis
            self._client = _redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except ImportError:
            raise RuntimeError("redis package not installed. Run: pip install redis")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Redis at {redis_url}: {e}")

    def check(self, client_key: str) -> None:
        key = f"{self._KEY_PREFIX}{client_key}"
        now = self._time.time()
        pipe = self._client.pipeline()
        pipe.zremrangebyscore(key, 0, now - self._window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self._window + 1)
        _, _, count, _ = pipe.execute()
        if int(count) > self._max:
            raise RateLimitExceeded(client_key, self._max, self._window)


# ── Factory helpers (dashboard / services) ────────────────────────────────────

def create_budget_store(require_redis: bool = False) -> QueryBudgetStore:
    """
    Return RedisBudgetStore when REDIS_URL is set; otherwise InMemoryBudgetStore.

    If require_redis=True and REDIS_URL is unset, raises RuntimeError (production).
    """
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if redis_url:
        return RedisBudgetStore(redis_url=redis_url)
    if require_redis or os.environ.get("REQUIRE_REDIS", "").lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "REDIS_URL is required for production dashboard deployments. "
            "Set REDIS_URL=redis://redis:6379/0 (docker-compose) or deploy Redis in K8s."
        )
    return InMemoryBudgetStore()


def create_rate_limiter(
    max_requests: int = 100,
    window_seconds: int = 60,
    require_redis: bool = False,
) -> RateLimiter:
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if redis_url:
        return RedisRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
            redis_url=redis_url,
        )
    if require_redis or os.environ.get("REQUIRE_REDIS", "").lower() in ("1", "true", "yes"):
        raise RuntimeError("REDIS_URL is required for production rate limiting.")
    return InMemoryRateLimiter(max_requests=max_requests, window_seconds=window_seconds)
