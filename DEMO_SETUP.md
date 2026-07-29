# IntelliClave — Demo Machine Setup

_For presentation day — get everything running from scratch_

---

## Requirements

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Docker | 24+ | `docker --version` |
| minikube | 1.32+ | `minikube version` |
| kubectl | 1.28+ | `kubectl version --client` |
| Git | any | `git --version` |

---

## Step 1 — Clone and install

```bash
git clone <repo-url>
cd IntelliClave

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd dashboard/frontend/intelliclave-ui
npm install
cd ../../..
```

---

## Step 2 — Verify everything works (run before the demo)

```bash
# Crypto — should print: ALL CRYPTO TESTS PASSED ✓ (4/4)
python crypto/certs/test_crypto.py

# Attestation — should print: [ATTESTATION] ✓ ATTESTATION VERIFIED
python tee/attestation/attestation_simulator.py

# Sealed storage — should print: ALL SEALED STORAGE TESTS PASSED ✓
python tee/sealed_storage/sealed_storage.py

# Attestation integration — should print: ATTESTATION INTEGRATION DEMO PASSED ✓
python tee/attestation/attestation_integration.py
```

---

## Step 3 — Start the dashboard (keep running during demo)

Open **two terminals** and leave them running:

**Terminal A — Backend:**
```bash
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
Expected: `Uvicorn running on http://0.0.0.0:8001`

**Terminal B — Frontend:**
```bash
cd dashboard/frontend/intelliclave-ui
npm start
```
Expected: Vite starts, browser opens at `http://localhost:3000`

**Verify dashboard is live:**
```bash
# Should return {"status":"ok"}
curl http://localhost:8001/health

# Run full E2E test — should print: ALL E2E TESTS PASSED ✓ (11/11)
python dashboard/backend/test_e2e.py
```

---

## Step 4 — Demo: Show FL + DP running (optional live run)

Open **four terminals**:

**Terminal 1 — FL Server:**
```bash
python fl/run_server.py --rounds 5 --min-clients 3
```
Wait for: `Flower server running...`

**Terminal 2 — Client 1 (FitLife):**
```bash
python fl/fl_client.py --csv data/processed/client1.csv --id 1 --dp --epsilon 10.0 --rounds 5
```

**Terminal 3 — Client 2 (MediTrack):**
```bash
python fl/fl_client.py --csv data/processed/client2.csv --id 2 --dp --epsilon 10.0 --rounds 5
```

**Terminal 4 — Client 3 (CareWatch):**
```bash
python fl/fl_client.py --csv data/processed/client3.csv --id 3 --dp --epsilon 10.0 --rounds 5
```

Expected final output:
```
[Server][DP] Round 5 avg_ε=9.992 — saved to fl_privacy.json
[Server] Round 5 {'round': 5, 'loss': 0.27581, 'accuracy': 0.90102, 'macro_f1': 0.90019}
```

> If you don't want to run live, the pre-recorded results are already in
> `results/fl_rounds/` and the dashboard shows them automatically.

---

## Step 5 — Demo: Show security (quick)

```bash
# Show crypto tamper detection
python crypto/certs/test_crypto.py

# Show attestation — server verified, rogue server blocked
python tee/attestation/attestation_integration.py
```

---

## Step 6 — Demo: Show Kubernetes (optional)

```bash
# Start minikube
minikube start --driver=docker --cpus=4 --memory=6144

# Full cold start (creates namespace, secrets, deploys all pods)
bash kubernetes/cold_start.sh

# Check pods are running
kubectl get pods -n intelliclave
```

---

## Demo Script (5-minute version)

| Time | What to show | Where |
|------|-------------|-------|
| 0:00 | Dashboard overview — KPI cards, live data | `http://localhost:3000` |
| 0:45 | FL Training chart — accuracy improving over rounds | Dashboard panel |
| 1:15 | Privacy Budget panel — ε consumed, limit line | Dashboard panel |
| 1:45 | TEE Attestation panel — MRENCLAVE, VERIFIED badge | Dashboard panel |
| 2:15 | TEE Overhead chart — 35% overhead, acceptable | Dashboard panel |
| 2:45 | Crypto test — tamper detection | Terminal |
| 3:15 | Attestation integration — rogue server blocked | Terminal |
| 3:45 | Graph 6 — 4-panel results summary | `results/graphs/graph6_final_results.png` |
| 4:15 | K8s cold start (or show YAML) | Terminal / editor |
| 4:45 | Q&A | — |

---

## Key Numbers to Know

| Question | Answer |
|----------|--------|
| What dataset? | UCI HAR — 10,299 samples, 6 activities, 3 non-IID clients |
| FL baseline accuracy? | 96.99% (no DP, 10 rounds) |
| FL+DP accuracy? | 91.27% (ε=10, 5 rounds) |
| Privacy cost? | −5.72% accuracy |
| Why ε=10? | Balance between privacy and utility — ε=1 gives only 50.89% |
| Membership inference AUC? | 0.503 — near random, model is resistant |
| Gradient poisoning drop? | −3.78% at 100% label flip — FedAvg dilutes it |
| TEE overhead? | 35.2% avg — only 0.14% of total FL round time |
| Why gramine-direct not gramine-sgx? | WSL2 has no SGX hardware — zero code changes for production |
| How many K8s pods? | 5 (server + 3 clients + dashboard) |
| Crypto scheme? | AES-256-GCM + RSA-2048 key encapsulation |

---

## If Something Breaks

**Backend won't start:**
```bash
# Check port is free
netstat -ano | findstr :8001
# Kill if occupied, then restart
```

**Frontend won't start:**
```bash
cd dashboard/frontend/intelliclave-ui
npm install  # reinstall dependencies
npm start
```

**Dashboard shows no data:**
```bash
# Confirm backend is running
curl http://localhost:8001/health
# Should return {"status":"ok"}
```

**FL clients can't connect to server:**
```bash
# Make sure server started first and shows "Flower server running"
# Check server address — default is 127.0.0.1:8080
```

**Crypto keys missing:**
```bash
python crypto/certs/generate_tls_certs.py
python -c "from crypto.certs.crypto_context import CryptoContext; CryptoContext.load_or_create()"
```

---

## Files to Have Open During Demo

```
dashboard/frontend/intelliclave-ui  → browser at localhost:3000
results/graphs/graph6_final_results.png
tee/attestation/attestation_integration.py
crypto/certs/test_crypto.py
results/cold_start_test.log
HANDOFF.md
```
