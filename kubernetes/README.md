# kubernetes/

Production-style cluster deployment (minikube).

## Purpose

This folder deploys IntelliClave on Kubernetes: FL server, multiple clients, dashboard, network policies, persistent volumes, and crypto secrets. Supports both standard and **Gramine-SGX** deployment variants.

## Key files

| Path | Role |
|------|------|
| `cold_start.sh` | Full bootstrap: minikube check → namespace → secrets → deploy → wait |
| `validate.sh` | `kubectl apply --dry-run=client` on all manifests |
| `namespace.yaml` | `intelliclave` namespace |
| `deployments/fl-server.yaml` | FL server pod |
| `deployments/fl-client-{1,2,3}.yaml` | Client pods |
| `deployments/fl-server-sgx.yaml` | Gramine-SGX server variant |
| `deployments/fl-clients-sgx.yaml` | Gramine-SGX client variant |
| `deployments/dashboard.yaml` | Dashboard backend + frontend |
| `services/fl-server-service.yaml` | ClusterIP for FL server |
| `policies/network-policy.yaml` | Restrict FL (8080) and dashboard (8001) traffic |
| `secrets/fl-crypto-keys-secret.yaml` | RSA keys from `crypto/certs/keys/` |
| `volumes/` | PVCs for server (1Gi) and clients (100Mi each) |

## How it connects

- Uses images built from **`docker/Dockerfile.*`**
- Same data/crypto/results mounts as Docker compose
- **`SGX=true`** switches to Gramine-SGX manifests
- Attestation flow from **`tee/`** in SGX mode

## Commands

```bash
minikube start --driver=docker --cpus=4 --memory=6144
bash kubernetes/cold_start.sh

# Hardware SGX / Gramine-SGX mode
SGX=true bash kubernetes/cold_start.sh

# Validate YAML only
bash kubernetes/validate.sh
```

## Prerequisites

- `kubectl`, `minikube`
- Docker images built or pulled per `cold_start.sh`
- Crypto keys populated: `crypto/certs/keys/server_private.pem` and `server_public.pem`
