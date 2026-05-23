#!/bin/bash
# ── IntelliClave — Full K8s SGX Cold Start ────────────────────────────────────
#
# Brings the entire IntelliClave cluster up from zero:
#   1. Start minikube with Docker driver
#   2. Create intelliclave namespace
#   3. Generate RSA keypair and create fl-crypto-keys Secret
#   4. Apply all K8s resources (PVCs, services, network policy, deployments)
#   5. Wait for fl-server pod to be ready (attestation init container runs)
#   6. Wait for all 3 client pods to be ready (attestation verified)
#   7. Deploy Redis + dashboard API
#   8. Print status summary
#
# Usage:
#   bash kubernetes/cold_start.sh
#
# SGX mode (production — requires SGX-capable node):
#   SGX=true bash kubernetes/cold_start.sh
#
# gramine-direct mode (WSL2 prototype — default):
#   bash kubernetes/cold_start.sh

set -e

SGX="${SGX:-false}"
NAMESPACE="intelliclave"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "======================================================="
echo "IntelliClave — K8s SGX Cold Start"
echo "  Mode      : $([ "$SGX" = "true" ] && echo "gramine-sgx (production)" || echo "gramine-direct (prototype)")"
echo "  Namespace : $NAMESPACE"
echo "======================================================="
echo ""

# ── Step 1: Start minikube ────────────────────────────────────────────────────
echo "[1/8] Starting minikube..."
if minikube status --profile=minikube 2>/dev/null | grep -q "Running"; then
    echo "  minikube already running — skipping"
else
    minikube start \
        --driver=docker \
        --cpus=4 \
        --memory=6144 \
        --profile=minikube
    echo "  ✓ minikube started"
fi

# ── Step 2: Create namespace ──────────────────────────────────────────────────
echo ""
echo "[2/8] Creating namespace..."
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"
echo "  ✓ namespace/$NAMESPACE ready"

# ── Step 3: Generate crypto keypair + create Secret ──────────────────────────
echo ""
echo "[3/8] Generating RSA-2048 keypair and creating fl-crypto-keys Secret..."

KEY_DIR="$ROOT_DIR/crypto/certs/keys"
mkdir -p "$KEY_DIR"

if [ ! -f "$KEY_DIR/server_private.pem" ] || [ ! -f "$KEY_DIR/server_public.pem" ]; then
    python3 -c "
import sys; sys.path.insert(0, '$ROOT_DIR/crypto/certs')
from crypto_context import CryptoContext
ctx = CryptoContext.load_or_create('$KEY_DIR')
print('  Keypair generated at $KEY_DIR')
"
else
    echo "  Keypair already exists at $KEY_DIR — reusing"
fi

# Create or update the K8s secret
kubectl create secret generic fl-crypto-keys \
    --from-file=server_private.pem="$KEY_DIR/server_private.pem" \
    --from-file=server_public.pem="$KEY_DIR/server_public.pem" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
echo "  ✓ Secret/fl-crypto-keys created"

# ── Step 4: Apply all K8s resources ──────────────────────────────────────────
echo ""
echo "[4/8] Applying K8s resources..."

# Volumes first
kubectl apply -f "$SCRIPT_DIR/volumes/server-pvc.yaml"
kubectl apply -f "$SCRIPT_DIR/volumes/client-pvcs.yaml"
kubectl apply -f "$SCRIPT_DIR/volumes/shared-storage.yaml"

# Services
kubectl apply -f "$SCRIPT_DIR/services/fl-server-service.yaml"

# Network policy
kubectl apply -f "$SCRIPT_DIR/policies/network-policy.yaml"

# Deployments — choose SGX or standard based on SGX flag
if [ "$SGX" = "true" ]; then
    echo "  Applying SGX deployments..."
    kubectl apply -f "$SCRIPT_DIR/deployments/fl-server-sgx.yaml"
    kubectl apply -f "$SCRIPT_DIR/deployments/fl-clients-sgx.yaml"
else
    echo "  Applying standard deployments (gramine-direct)..."
    kubectl apply -f "$SCRIPT_DIR/deployments/fl-server.yaml"
    kubectl apply -f "$SCRIPT_DIR/deployments/fl-client-1.yaml"
    kubectl apply -f "$SCRIPT_DIR/deployments/fl-client-2.yaml"
    kubectl apply -f "$SCRIPT_DIR/deployments/fl-client-3.yaml"
fi

kubectl apply -f "$SCRIPT_DIR/deployments/redis.yaml"
kubectl apply -f "$SCRIPT_DIR/deployments/dashboard.yaml"

echo "  ✓ All resources applied"

# ── Step 5: Wait for fl-server ────────────────────────────────────────────────
echo ""
echo "[5/8] Waiting for fl-server pod (attestation init container runs first)..."
kubectl rollout status deployment/fl-server \
    --namespace="$NAMESPACE" \
    --timeout=120s
echo "  ✓ fl-server ready"

# ── Step 6: Wait for all 3 clients ───────────────────────────────────────────
echo ""
echo "[6/8] Waiting for FL clients (attestation verified before connecting)..."

for CLIENT in fl-client-1 fl-client-2 fl-client-3; do
    echo "  Waiting for $CLIENT..."
    kubectl rollout status deployment/$CLIENT \
        --namespace="$NAMESPACE" \
        --timeout=120s
    echo "  ✓ $CLIENT ready"
done

# ── Step 7: Wait for dashboard + Redis ────────────────────────────────────────
echo ""
echo "[7/8] Waiting for Redis and dashboard-backend..."
kubectl rollout status deployment/redis \
    --namespace="$NAMESPACE" \
    --timeout=120s
kubectl rollout status deployment/dashboard-backend \
    --namespace="$NAMESPACE" \
    --timeout=180s
echo "  ✓ Redis and dashboard-backend ready"

# ── Step 8: Status summary ────────────────────────────────────────────────────
echo ""
echo "[8/8] Cluster status:"
kubectl get pods --namespace="$NAMESPACE" -o wide
echo ""
kubectl get services --namespace="$NAMESPACE"

echo ""
echo "======================================================="
echo "COLD START COMPLETE ✓"
echo ""
echo "  FL server   : running (attestation published)"
echo "  FL client 1 : running (attestation verified)"
echo "  FL client 2 : running (attestation verified)"
echo "  FL client 3 : running (attestation verified)"
echo ""
echo "  Monitor training:"
echo "    kubectl logs -f deployment/fl-server -n $NAMESPACE"
echo ""
echo "  Dashboard API : kubectl port-forward svc/dashboard-service 8001:8001 -n $NAMESPACE"
echo "  Check attestation:"
echo "    cat results/attestation.json"
echo "  Build images first (from repo root):"
echo "    bash scripts/build_docker_images.sh"
echo "======================================================="
