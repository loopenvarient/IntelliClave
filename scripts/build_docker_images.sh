#!/usr/bin/env bash
# Build all IntelliClave container images (run from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Building FL server..."
docker build -f docker/Dockerfile.server -t intelliclave-server:latest .
docker tag intelliclave-server:latest fl-server-image:latest

echo "Building FL client..."
docker build -f docker/Dockerfile.client -t intelliclave-client:latest .
docker tag intelliclave-client:latest fl-client-image:latest

echo "Building dashboard API..."
docker build -f dashboard/backend/Dockerfile -t intelliclave-dashboard:latest .

echo "Building dashboard UI..."
docker build -f dashboard/frontend/Dockerfile \
  --build-arg VITE_API_URL=http://localhost:8001 \
  -t intelliclave-dashboard-ui:latest .

echo ""
echo "Done. Tags:"
echo "  intelliclave-server:latest"
echo "  intelliclave-client:latest"
echo "  intelliclave-dashboard:latest   (K8s: dashboard-backend)"
echo "  intelliclave-dashboard-ui:latest"
echo ""
echo "Load into minikube (optional):"
echo "  minikube image load intelliclave-server:latest"
echo "  minikube image load intelliclave-client:latest"
echo "  minikube image load intelliclave-dashboard:latest"
