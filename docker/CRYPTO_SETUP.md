# IntelliClave Docker Crypto Key Setup Guide

## Overview

IntelliClave uses RSA-2048 keypairs for weight encryption in federated learning. Keys are stored in `crypto/certs/keys/` and mounted into containers.

### Key Files
- **`server_private.pem`** — server private key (kept secure, never leaves server)
- **`server_public.pem`** — server public key (distributed to all clients)

---

## Development Setup (Recommended)

### Option A: Auto-Generation (Default)

In development, the server automatically generates keys on first run if they don't exist.

#### Prerequisites
```bash
# Ensure crypto directory exists
mkdir -p crypto/certs/keys
```

#### Run Docker Compose
```bash
cd docker
docker compose up --build
```

On first run:
- Server container starts and calls `CryptoContext.load_or_create()`
- RSA-2048 keypair is automatically generated and saved to `crypto/certs/keys/`
- Keys are shared with all clients
- Training begins

#### Verify Keys Were Created
```bash
ls -la crypto/certs/keys/
# Should show:
# -rw------- server_private.pem
# -rw-r--r-- server_public.pem
```

**Note**: The mount point for server is **read-write** (no `:ro` suffix) to allow key generation:
```yaml
fl-server:
  volumes:
    - ../crypto/certs/keys:/app/crypto/certs/keys  # ← RW mount (no :ro)
```

---

### Option B: Pre-Generated Keys

If you want to generate keys offline before running Docker:

```bash
cd crypto/certs
python -c "
from crypto_context import CryptoContext
ctx = CryptoContext.load_or_create('keys')
print(f'Keys ready at: {ctx.key_dir}')
"
```

Then run Docker:
```bash
docker compose up --build
```

Server will load existing keys instead of generating new ones.

---

## Production Setup

### Using Docker Secrets (Recommended)

For production deployments with Docker Swarm or Kubernetes, use Docker secrets to protect the private key.

#### 1. Generate Keys Outside the Cluster

```bash
# On your local machine (secure environment)
mkdir -p crypto/certs/keys
python -c "from crypto_certs.crypto_context import CryptoContext; CryptoContext.load_or_create('crypto/certs/keys')"
```

#### 2. Create Docker Secrets

**For Docker Swarm:**
```bash
docker secret create server_private_key crypto/certs/keys/server_private.pem
docker secret create server_public_key crypto/certs/keys/server_public.pem
```

**For Kubernetes:**
```bash
kubectl create secret generic server-keys \
  --from-file=private=crypto/certs/keys/server_private.pem \
  --from-file=public=crypto/certs/keys/server_public.pem \
  -n intelliclave
```

#### 3. Update docker-compose.yml (or helm values.yaml)

```yaml
# docker-compose.prod.yml
services:
  fl-server:
    volumes:
      - server-results:/app/results
      # Remove mounted crypto directory
    secrets:
      - server_private_key
      - server_public_key
    environment:
      PRIVATE_KEY_PATH: /run/secrets/server_private_key
      PUBLIC_KEY_PATH: /run/secrets/server_public_key
    
secrets:
  server_private_key:
    external: true
  server_public_key:
    external: true
```

#### 4. Update CryptoContext to Read from Secrets

Modify `crypto/certs/crypto_context.py`:

```python
@classmethod
def load_or_create(cls, key_dir: str = _DEFAULT_KEY_DIR) -> "CryptoContext":
    # In production, read from /run/secrets (Docker) or env vars
    if os.path.exists("/run/secrets/server_private_key"):
        with open("/run/secrets/server_private_key", "rb") as f:
            private_key = load_private_key(f.read())
        with open("/run/secrets/server_public_key", "rb") as f:
            public_key = load_public_key(f.read())
        print("[CryptoContext] Loaded keypair from Docker secrets")
        return cls(private_key=private_key, public_key=public_key)
    
    # Development fallback: generate in key_dir
    os.makedirs(key_dir, exist_ok=True)
    priv_path = os.path.join(key_dir, "server_private.pem")
    pub_path = os.path.join(key_dir, "server_public.pem")
    ...
```

---

## Client Key Mounting

Clients need **read-only access** to the server's public key for encryption:

```yaml
fl-client-1:
  volumes:
    - ../crypto/certs/keys:/app/crypto/certs/keys:ro  # ← Read-only
```

This prevents clients from modifying or exposing the private key.

---

## Security Best Practices

### ✅ DO
- Generate keys in a secure environment before production deployment
- Use Docker secrets or Kubernetes secrets to protect private keys
- Mount private keys read-only on server if pre-generated
- Regularly rotate keys (implement key versioning)
- Audit key access logs

### ❌ DON'T
- Store keys in version control
- Mount keys as read-write in production if not needed
- Share private keys across multiple servers
- Run key generation in untrusted environments
- Log or export private keys for debugging

---

## Troubleshooting

### Server Fails to Start: "Permission Denied"

**Symptom:**
```
FileNotFoundError: [Errno 13] Permission denied: 'crypto/certs/keys/server_private.pem'
```

**Cause:** Crypto mount is read-only (`:ro`) but keys don't exist yet.

**Solution:**
```bash
# Option 1: Pre-generate keys
python crypto/certs/crypto_context.py

# Option 2: Remove :ro from docker-compose.yml
sed -i 's|crypto/certs/keys:ro|crypto/certs/keys|' docker/docker-compose.yml

# Then run again
docker compose up --build
```

### Keys Not Shared with Clients

**Symptom:** Client crashes with "Cannot find server public key"

**Cause:** Keys mounted as separate volumes, not shared.

**Solution:** Ensure clients mount the same `crypto/certs/keys` directory:
```yaml
fl-client-1:
  volumes:
    - ../crypto/certs/keys:/app/crypto/certs/keys:ro  # Must match server path
```

### Key Permissions Too Restrictive (Kubernetes)

**Symptom:** Container runs as non-root user, can't read `/run/secrets/`

**Solution:** Kubernetes secrets are readable by any pod in the namespace. Ensure RBAC policies restrict which pods can access the secret:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: server-secret-reader
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: secret-reader
subjects:
  - kind: ServiceAccount
    name: fl-server
    namespace: intelliclave
```

---

## See Also

- `crypto/certs/crypto_context.py` — CryptoContext implementation
- `crypto/certs/crypto_layer.py` — RSA encryption/decryption
- `fl/fl_server.py` — Server key usage in FL aggregation
- `fl/fl_client.py` — Client key usage for weight encryption
