# dashboard/

Live monitoring UI and inference API for IntelliClave.

## Purpose

This folder provides a **FastAPI backend** and **React frontend** to visualize FL training progress, privacy budget, TEE attestation, security attack results, and client distributions. The backend also serves **defended inference** via `/predict`.

## Structure

```
dashboard/
├── backend/
│   ├── main.py          # FastAPI app (9 REST endpoints)
│   └── test_e2e.py      # Endpoint tests (no live server required)
├── frontend/intelliclave-ui/
│   ├── src/App.jsx      # Main dashboard UI
│   └── package.json     # React + Vite + Recharts
└── API_AUTH.md          # Token auth documentation
```

## API endpoints

| Endpoint | Data source |
|----------|-------------|
| `GET /status` | `status.json` + `distribution_report.json` |
| `GET /results` | `results/results.json` + privacy merge + per-class F1 |
| `GET /attacks` | `results/attacks/*.json` (defended / robust variants preferred) |
| `GET /attestation` | `attestation.json` |
| `GET /benchmarks` | `results/benchmarks_baseline.json` |
| `GET /privacy_log` | `results/fl_rounds/run_*/fl_privacy.json` |
| `GET /comparison` | `results/local_vs_global.json` (local-only vs global FL) |
| `POST /predict` | Live `global_model_latest.pth` with PrivacyWrapper |

## How it connects

- Reads artifacts written by **`fl/`**, **`tee/`**, **`security/`**, **`privacy/`**
- Loads model from **`results/fl_rounds/`** using **`fl/model.py`**
- Defence settings from **`config/constants.py`** (overridable via env vars)

## Commands

```bash
# Backend
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Frontend
cd dashboard/frontend/intelliclave-ui
npm install
npm run dev    # http://localhost:5173

# Tests
python dashboard/backend/test_e2e.py

# Windows — opens both in separate terminals
powershell -ExecutionPolicy Bypass -File scripts/start_dashboard.ps1
```

## Auth

Default tokens: `admin / adminpass`, `viewer / viewerpass`. See `API_AUTH.md` for production guidance.
