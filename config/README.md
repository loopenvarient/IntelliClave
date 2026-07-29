# config/

Shared configuration constants for the whole IntelliClave stack.

## Purpose

This folder holds **one source of truth** for defaults used across federated learning, differential privacy, security attacks, and the dashboard API. Instead of hardcoding values in every module, code imports from `constants.py`.

## Key files

| File | What it defines |
|------|-----------------|
| `constants.py` | FL rounds, client count, learning rate, batch size, DP ε target, model-inversion defence (`MI_NOISE_SCALE`, `MI_TEMPERATURE`, `MI_DEFENCE_ENABLED`), output masking for `/predict` |

## How it connects

- **`fl/`** — training defaults, PrivacyWrapper settings on server/clients
- **`privacy/`** — default ε, clipping norms
- **`security/attacks/`** — defence ablation baselines
- **`dashboard/backend/`** — inference defence env overrides read against these defaults

## Usage

```python
from config.constants import DEFAULT_EPSILON, MI_NOISE_SCALE, MI_TEMPERATURE
```

There is no standalone script here — edit `constants.py` when you want to change project-wide defaults.
