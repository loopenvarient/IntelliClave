# crypto/

Cryptographic protection for federated learning weight updates and TLS transport.

## Purpose

This layer encrypts model weight payloads exchanged between FL clients and the server. It uses **AES-256-GCM** for symmetric encryption and **RSA-2048** for key exchange. A separate TLS cert generator supports secure transport setup.

## Key files

| Path | Role |
|------|------|
| `certs/crypto_layer.py` | Encrypt/decrypt weight arrays |
| `certs/crypto_context.py` | Server keypair load/create lifecycle |
| `certs/generate_tls_certs.py` | Generate CA, server, and client TLS certificates |
| `certs/test_crypto.py` | Four-test suite (round-trip, tamper rejection, integrity) |
| `certs/keys/` | Auto-generated `server_private.pem` / `server_public.pem` |
| `certs/tls/` | Generated TLS certificate material |

## How it connects

- **`fl/run_server.py --crypto`** — server decrypts client weights before aggregation
- **`fl/run_client.py --crypto`** — clients encrypt updates before sending
- **`docker/`** and **`kubernetes/`** — mount `crypto/certs/keys` into containers
- **`tee/benchmarks/`** — measures encrypt/decrypt overhead vs baseline

## Commands

```bash
python crypto/certs/test_crypto.py
python crypto/certs/generate_tls_certs.py
```

Keys are created automatically on the first FL server run with `--crypto`.
