# IntelliClave Dashboard API Authentication

## Overview

The IntelliClave dashboard API (`dashboard/backend/main.py`) implements **Bearer token authentication** for sensitive endpoints. This guide covers setup, configuration, and usage.

## Protected Endpoints

The following endpoints require a valid API key:

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/predict` | POST | Run inference on features | ✅ YES |
| `/privacy_log` | GET | View privacy budget logs | ✅ YES |
| `/health` | GET | Health check | ❌ No |
| `/status` | GET | Training status summary | ❌ No |
| `/results` | GET | Round-by-round metrics | ❌ No |
| `/attestation` | GET | SGX attestation record | ❌ No |
| `/benchmarks` | GET | Performance benchmarks | ❌ No |
| `/attacks` | GET | Security attack results | ❌ No |
| `/query_stats` | GET | Rate limit status | ❌ No |
| `/model_info` | GET | Model metadata | ❌ No |

---

## Setup

### Option A: Auto-Generated Key (Development)

By default, if no `API_KEY` environment variable is set, the dashboard generates a random 43-character API key on startup.

**Usage:**
```bash
cd dashboard/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

**Output:**
```
[Dashboard] Generated random API key: Drmhg3X...4kL5E6
[Dashboard] Set API_KEY environment variable to use a custom key
```

### Option B: Custom Key (Recommended)

Set the `API_KEY` environment variable before starting the dashboard:

```bash
export API_KEY="your-secure-api-key-here"
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

**For Docker:**
```yaml
services:
  dashboard:
    environment:
      API_KEY: "${API_KEY}"
    # or use secrets:
    # secrets:
    #   - api_key
```

### Option C: Kubernetes Secrets

Store the API key in a Kubernetes secret:

```bash
kubectl create secret generic dashboard-api-key \
  --from-literal=api-key=your-secure-key \
  -n intelliclave
```

Update deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashboard
spec:
  template:
    spec:
      containers:
      - name: dashboard
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: dashboard-api-key
              key: api-key
```

---

## Making Authenticated Requests

### Using cURL

```bash
# Protected endpoint: /predict
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"features": [1.0, 2.0, 3.0, ...], "return_confidence": true}'

# Public endpoint: /health (no auth needed)
curl http://localhost:8001/health
```

### Using Python Requests

```python
import requests

API_KEY = "your-api-key"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Predict
response = requests.post(
    "http://localhost:8001/predict",
    json={"features": [1.0, 2.0, 3.0], "return_confidence": True},
    headers=headers
)
print(response.json())

# Privacy log
response = requests.get(
    "http://localhost:8001/privacy_log",
    headers=headers
)
print(response.json())
```

### Using JavaScript/Fetch

```javascript
const API_KEY = "your-api-key";

async function predict(features) {
  const response = await fetch("http://localhost:8001/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`
    },
    body: JSON.stringify({
      features: features,
      return_confidence: true
    })
  });
  return response.json();
}

// Usage
const result = await predict([1.0, 2.0, 3.0]);
console.log(result);
```

---

## Error Handling

### Invalid or Missing API Key

**Status Code:** 401 Unauthorized

**Response:**
```json
{
  "detail": "Invalid or missing API key. Use: Authorization: Bearer <API_KEY>"
}
```

**Fix:**
- Verify the API key is correct
- Check the `Authorization` header format: `Bearer YOUR_KEY` (with space)
- Ensure the header is spelled correctly (case-sensitive: `Authorization`)

### Correct Header Format

✅ **CORRECT:**
```
Authorization: Bearer Drmhg3Xk7pL...
```

❌ **INCORRECT:**
```
Authorization: Drmhg3Xk7pL...       (missing "Bearer ")
Authorization: bearer Drmhg3Xk7pL... (lowercase "bearer" — HTTP headers are case-insensitive, but standard is uppercase)
Api-Key: Drmhg3Xk7pL...             (wrong header name)
```

---

## Security Best Practices

### ✅ DO
- Use strong, randomly generated API keys (43+ characters minimum)
- Store keys in environment variables or secrets managers
- Use HTTPS/TLS in production (not HTTP)
- Rotate keys periodically
- Log authentication attempts for audit trails
- Use different keys for different services/environments

### ❌ DON'T
- Hardcode API keys in code or config files
- Commit keys to version control (even in .env files)
- Share keys across multiple services (use service-specific keys)
- Log full API keys in application output
- Use simple/guessable keys ("123456", "password", etc.)
- Expose keys in client-side JavaScript without a backend proxy

### Key Rotation (Production)

1. Generate a new API key
2. Deploy with both old and new keys accepted (2-key config)
3. Update clients to use the new key
4. After grace period, disable the old key
5. Monitor logs for old key usage during transition

---

## Integration with Dashboard Frontend

The React frontend (`dashboard/frontend/`) needs to pass the API key when calling protected endpoints:

**Example:** `dashboard/frontend/src/api.js`

```javascript
const API_KEY = process.env.REACT_APP_API_KEY;
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8001";

function getHeaders() {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${API_KEY}`
  };
}

export async function predict(features) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      features: features,
      return_confidence: true
    })
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}
```

**Environment variables** (in `.env`):
```
REACT_APP_API_KEY=your-api-key
REACT_APP_API_URL=http://localhost:8001
```

---

## Troubleshooting

### "Invalid or missing API key" on correct key

**Cause:** Header format issue

**Solution:**
```bash
# Check the exact header format
curl -v -X GET http://localhost:8001/health \
  -H "Authorization: Bearer YOUR_KEY"
# Look for "Authorization: Bearer" in request headers
```

### API key changes not taking effect

**Cause:** Dashboard still running with cached old key

**Solution:**
```bash
# Restart the dashboard
docker restart dashboard-container
# or kill and restart the uvicorn process
```

### Different API key for development vs production

**Solution:** Use environment-specific configs

```bash
# Development
export API_KEY="dev-key-12345"
python -m uvicorn main:app

# Production
export API_KEY="$(cat /run/secrets/api_key)"
python -m uvicorn main:app
```

---

## Advanced: Custom Auth Implementation

If you want to replace Bearer token auth with a different method:

1. **JWT tokens** — Set `Authorization: Bearer <JWT_TOKEN>` (verify signature)
2. **mTLS certificates** — Use client certificates instead of API keys
3. **OAuth 2.0** — Integrate with an identity provider (Keycloak, Auth0)
4. **API Gateway** — Let Kong or AWS API Gateway handle auth

To implement, modify `dashboard/backend/main.py`:

```python
from fastapi.security import HTTPBearer, HTTPAuthCredentials

# Example: Replace HTTPBearer with custom auth
def custom_auth(request: Request):
    # Your custom logic here
    token = request.headers.get("X-Custom-Token")
    if token != CUSTOM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/predict")
def predict(payload: PredictRequest, _: None = Depends(custom_auth)):
    ...
```

---

## See Also

- `dashboard/backend/main.py` — API implementation
- `dashboard/frontend/` — React dashboard (needs API_KEY for protected endpoints)
- `docker-compose.yml` — Docker environment variable setup
- `kubernetes/` — Kubernetes secret configuration
