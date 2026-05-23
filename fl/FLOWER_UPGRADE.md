# Flower Upgrade Guide

IntelliClave is pinned to **Flower 1.6.0** (`flwr==1.6.0` in `requirements.txt`).
The codebase contains custom glue that is tightly coupled to the 1.6 API.
Before upgrading, review each breakage point below and test the full FL stack.

---

## Breakage points

### 1. `FitRes` constructor — `fl_server.py` `aggregate_fit`

```python
from flwr.common import FitRes as _FitRes
fit_res = _FitRes(
    status=fit_res.status,
    parameters=...,
    num_examples=fit_res.num_examples,
    metrics=fit_res.metrics,
)
```

**Risk:** Flower 1.7+ may rename or reorder `FitRes` fields.
**Test:** After upgrading, run `python fl/run_fl_simulation.py --rounds 2 --clients 3`
and confirm no `TypeError` on strategy construction or aggregation.

---

### 2. `fl.client.start_numpy_client` — `fl_client.py`

```python
fl.client.start_numpy_client(server_address=..., client=IntelliClaveClient(...))
```

**Risk:** Deprecated in Flower 1.8, removed in 2.x. Replace with:

```python
fl.client.start_client(
    server_address=...,
    client=IntelliClaveClient(...).to_client(),
)
```

`NumPyClient.to_client()` is available from Flower 1.7 onward.

---

### 3. `SaveModelFedProx` multiple inheritance — `fl_server.py` `build_strategy`

```python
class SaveModelFedProx(SaveModelStrategy, FedProx):
    pass
return SaveModelFedProx(proximal_mu=proximal_mu, **common_kwargs)
```

**Risk:** If `FedProx.__init__` gains new required kwargs, or if MRO changes,
this raises `TypeError` at strategy construction time.
**Test:** Run `python fl/run_server.py --strategy fedprox --rounds 2 --min-clients 3`
and confirm the server starts without error.

---

### 4. `on_fit_config_fn` / `fit_metrics_aggregation_fn` kwargs

These are passed as `**kwargs` to `FedAvg.__init__`. Flower 1.7 renamed
`fit_metrics_aggregation_fn` → `fit_metrics_aggregation_fn` (no change) but
added `evaluate_fn` deprecation warnings. Check for `DeprecationWarning` on
strategy construction after upgrading.

---

## Upgrade checklist

1. Update `requirements.txt`: `flwr==<new_version>`
2. Run `python -c "import flwr; print(flwr.__version__)"` to confirm
3. Run `python fl/run_fl_simulation.py --rounds 2 --clients 3` — no errors
4. Run `python fl/run_server.py --strategy fedprox --rounds 2 --min-clients 3` — no errors
5. Run `python fl/run_client.py --id 1 --rounds 2` against the server — no errors
6. Run `python dashboard/backend/test_e2e.py` — all 13 tests pass
7. Update the version guard in `fl_server.py`:
   ```python
   _FLWR_REQUIRED = (<major>, <minor>)
   ```
8. Update this file with any new breakage points discovered

---

## Why not upgrade now?

Flower 1.7+ introduces the `ClientApp` / `ServerApp` abstraction which would
require rewriting `fl_client.py`, `fl_server.py`, and `run_fl_simulation.py`.
The 1.6 API is stable for the current demo scope. Upgrade when the project
moves to production deployment.
