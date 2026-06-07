# docker/

Containerized deployment of the full FL stack.

## Purpose

This folder packages the IntelliClave server and clients into Docker images and orchestrates multi-client training with optional **DP** and **crypto** via docker-compose. Use it for reproducible demos and local multi-container testing without manual terminal management.

## Key files

| Path | Role |
|------|------|
| `Dockerfile.server` | FL server image |
| `Dockerfile.client` | FL client image |
| `docker-compose.yml` | Base compose (server + client template) |
| `generate_compose.py` | Generate N-client compose file |
| `CRYPTO_SETUP.md` | Key mounting and secrets for production |

## How it connects

- Runs the same entrypoints as **`fl/run_server.py`** and **`fl/run_client.py`** inside containers
- Mounts **`data/processed/`**, **`crypto/certs/keys/`**, **`results/`** as volumes
- Environment variables control rounds, ε, `--dp`, `--crypto` flags

## Commands

```bash
# Generate 3-client stack
python docker/generate_compose.py --clients 3
docker compose -f docker/docker-compose.generated.yml up --build

# Base compose (single client template)
docker compose -f docker/docker-compose.yml up --build

# Scale to 5 clients
python docker/generate_compose.py --clients 5
```

## Typical env overrides

`ROUNDS`, `EPSILON`, `USE_DP=--dp`, `USE_CRYPTO=--crypto` — see compose files and `CRYPTO_SETUP.md`.
