# IntelliClave — Data Contracts

_Last updated: 2026-04-25_

---

## Dataset: UCI HAR (Human Activity Recognition)

| Split | Samples | Raw Features |
|-------|---------|--------------|
| Train | 7352    | 561          |
| Test  | 2947    | 561          |

- Feature range: `-1.0` to `1.0` (pre-normalised)
- Labels: 6 activity classes (integers 1–6)

---

## Class Distribution (train)

| Label | Activity           | Count |
|-------|--------------------|-------|
| 1     | WALKING            | 1226  |
| 2     | WALKING_UPSTAIRS   | 1073  |
| 3     | WALKING_DOWNSTAIRS | 986   |
| 4     | SITTING            | 1286  |
| 5     | STANDING           | 1374  |
| 6     | LAYING             | 1407  |

- Min: 986 / Max: 1407 — roughly balanced, acceptable for FL training

---

## Subject × Class Matrix (train subjects only)

| Subj | WALK | WALK_UP | WALK_DN | SIT | STAND | LAY | TOTAL | Dominant |
|------|------|---------|---------|-----|-------|-----|-------|----------|
| 1    | 95   | 53      | 49      | 47  | 53    | 50  | 347   | WALK     |
| 3    | 58   | 59      | 49      | 52  | 61    | 62  | 341   | LAY      |
| 5    | 56   | 47      | 47      | 44  | 56    | 52  | 302   | WALK     |
| 6    | 57   | 51      | 48      | 55  | 57    | 57  | 325   | WALK     |
| 7    | 57   | 51      | 47      | 48  | 53    | 52  | 308   | WALK     |
| 8    | 48   | 41      | 38      | 46  | 54    | 54  | 281   | STAND    |
| 11   | 59   | 54      | 46      | 53  | 47    | 57  | 316   | WALK     |
| 14   | 59   | 54      | 45      | 54  | 60    | 51  | 323   | STAND    |
| 15   | 54   | 48      | 42      | 59  | 53    | 72  | 328   | LAY      |
| 16   | 51   | 51      | 47      | 69  | 78    | 70  | 366   | STAND    |
| 17   | 61   | 48      | 46      | 64  | 78    | 71  | 368   | STAND    |
| 19   | 52   | 40      | 39      | 73  | 73    | 83  | 360   | LAY      |
| 21   | 52   | 47      | 45      | 85  | 89    | 90  | 408   | LAY      |
| 22   | 46   | 42      | 36      | 62  | 63    | 72  | 321   | LAY      |
| 23   | 59   | 51      | 54      | 68  | 68    | 72  | 372   | LAY      |
| 25   | 74   | 65      | 58      | 65  | 74    | 73  | 409   | WALK     |
| 26   | 59   | 55      | 50      | 78  | 74    | 76  | 392   | SIT      |
| 27   | 57   | 51      | 44      | 70  | 80    | 74  | 376   | STAND    |
| 28   | 54   | 51      | 46      | 72  | 79    | 80  | 382   | LAY      |
| 29   | 53   | 49      | 48      | 60  | 65    | 69  | 344   | LAY      |
| 30   | 65   | 65      | 62      | 62  | 59    | 70  | 383   | LAY      |

Test-only subjects (no train data): 2, 4, 9, 10, 12, 13, 18, 20, 24

---

## Non-IID Client Split Design

| Client   | Subjects          | Train Samples | WALK% | SIT%  | STAND% | LAY%  | Character        |
|----------|-------------------|---------------|-------|-------|--------|-------|------------------|
| Client 1 | 1,3,5,6,7,8       | 1904          | 19.5% | 15.3% | 17.5%  | 17.2% | More active      |
| Client 2 | 11,14,15,16,17,19 | 2061          | 16.3% | 18.0% | 18.9%  | 19.6% | More sedentary   |
| Client 3 | 21,22,23,25,26,27,28,29,30 | 3387 | 15.3% | 18.4% | 19.2% | 20.0% | Most sedentary   |

- Client 1 skews active (higher WALK%), Clients 2 & 3 skew sedentary (higher SIT/STAND/LAY%)
- This creates genuine non-IID heterogeneity across FL clients

---

## Feature Engineering Decision

| Method | Components | Explained Variance | Decision  |
|--------|------------|--------------------|-----------|
| PCA    | 50         | 0.9308 (93.08%)    | **ADOPT** |

- Threshold: `> 0.90` → met
- **Final feature count: 50 (PCA)**
- PCA fitted on `X_train` only — no data leakage
- `X_test` transformed using the same fitted PCA object
- Model saved: `data/samples/pca_model.pkl`

---

## Subject Split (Train / Test)

| Split | Count | Subject IDs                                                |
|-------|-------|------------------------------------------------------------|
| Train | 21    | 1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30  |
| Test  | 9     | 2,4,9,10,12,13,18,20,24                                    |

- Overlap: **none** — fully disjoint
- Ideal for non-IID federated learning simulation

---

## Processed Client CSVs (FROZEN — do not modify)

> PCA applied to full dataset (train+test combined), split by subject.
> PCA fitted on X_train only — no leakage.

| File                        | Rows | Features | Classes | WALK | WALK_UP | WALK_DN | SIT | STAND | LAY |
|-----------------------------|------|----------|---------|------|---------|---------|-----|-------|-----|
| data/processed/client1.csv  | 3105 | 50       | 6       | 595  | 498     | 450     | 492 | 533   | 537 |
| data/processed/client2.csv  | 3426 | 50       | 6       | 550  | 511     | 458     | 595 | 653   | 659 |
| data/processed/client3.csv  | 3768 | 50       | 6       | 577  | 535     | 498     | 690 | 720   | 748 |

- NaN: 0, Inf: 0 across all files — verified
- Client 1 (subj 1–10): more active profile (higher WALK%)
- Client 2 (subj 11–20): transitional
- Client 3 (subj 21–30): most sedentary (higher SIT/STAND/LAY%)
- Distribution chart: `data/client_distributions.png`

---

## Class Weights (for loss function)

Computed via `sklearn.utils.class_weight.compute_class_weight('balanced')` over all 10299 samples.
Saved to `data/class_weights.json`.

| Activity           | Weight   |
|--------------------|----------|
| WALKING            | 0.996806 |
| WALKING_UPSTAIRS   | 1.111723 |
| WALKING_DOWNSTAIRS | 1.220839 |
| SITTING            | 0.965954 |
| STANDING           | 0.900577 |
| LAYING             | 0.882973 |

HAR is roughly balanced — weights are close to 1.0. Used for correctness, not heavy correction.

---

## Deployment Scenario

See `data/deployment_scenario.md` for full narrative.

- Company A (FitLife): active users → Client 1
- Company B (MediTrack): mixed healthcare users → Client 2
- Company C (CareWatch): elderly care users → Client 3

Regulatory compliance: GDPR Art. 25, GDPR Art. 5(1), HIPAA, PMDC — all satisfied.
Raw data never leaves each company. Only encrypted gradients are shared.

---

## Dashboard API Contracts

_Last updated: 2026-05-23_

All endpoints are served by `dashboard/backend/main.py` (FastAPI, port 8001).
CORS origins are configurable via `CORS_ORIGINS` env var (default: `http://localhost:3000`).

### GET /health
Returns `{"status": "ok"}`. No auth required.

### GET /status
Returns `status.json` enriched with training freshness fields:

| Field | Type | Description |
|-------|------|-------------|
| `training_freshness` | string | `live` \| `stale` \| `training` \| `pre-recorded` |
| `last_training_completed_at` | string \| null | ISO-8601 timestamp of last completed run |
| `stale_hours` | float \| null | Hours since last training (null if never trained) |
| `expected_training_cadence_hours` | int | Stale threshold (default 24, env `EXPECTED_TRAINING_CADENCE_HOURS`) |

### GET /results
Returns `results/results.json` enriched with the same freshness fields.
Includes `client_metrics`, `convergence`, and `convergence_summary` sub-objects.

### GET /attestation
Returns `attestation.json`. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `tee_verified` | bool | Always `true` (record only written on success) |
| `simulation_mode` | bool | `true` on gramine-direct / WSL2; `false` on real SGX hardware |
| `simulation_warning` | string | Human-readable warning when `simulation_mode=true` |
| `platform` | string | `"gramine-direct (simulation — no hardware SGX)"` or `"Intel SGX (hardware)"` |
| `mode` | string | `"gramine-direct"` or `"gramine-sgx"` |
| `mrenclave` | string | SHA-256 of manifest + code (software-computed in simulation) |
| `mutual_attestation` | bool | Whether client quotes were also verified |

> **Important:** when `simulation_mode=true` the attestation provides no hardware
> trust boundary. Quotes are software-computed from file hashes. Use `gramine-sgx`
> on SGX-capable hardware for production deployments.

### GET /benchmarks
Returns `results/benchmarks_baseline.json` — TEE overhead measurements.

### GET /attacks
Returns a summary object with keys `model_inversion`, `membership_inference`,
`gradient_poisoning`. Each value is the `summary` sub-object from the
corresponding attack JSON, or `null` if the file does not exist.

### GET /privacy_log
Returns `results/privacy_log.json`. Format:

| Field | Type | Description |
|-------|------|-------------|
| `log` | list | Per-round per-client cumulative ε entries |
| `cumulative_summary` | object | Worst-case ε across all clients and rounds |

### GET /model_info
Returns metadata about the currently loaded model checkpoint:
`input_dim`, `num_classes`, `class_names`, `model_type`, `checkpoint_path`.

### GET /query_stats
Returns rate-limit and lifetime budget status for the calling IP:
`requests_remaining_window`, `lifetime_budget_remaining`, `store_backend`.

### POST /predict
Request body:

```json
{ "features": [float, ...], "return_confidence": false }
```

`features` length must equal `input_dim` from `/model_info`.

Response:

```json
{ "predicted_class": 0, "predicted_label": "WALKING", "confidence": null }
```

Inference protections applied on every call:
- Laplace output perturbation (`noise_scale = L1_sensitivity / ε_inf`, default `ε_inf=4.0`)
- Lifetime query budget: 10 000 predictions per IP (HTTP 429 on exhaustion)
- Per-window rate limit: 100 req / 60 s per IP (HTTP 429 on breach)
- Confidence only returned when `return_confidence=true`; full probability vector never exposed

---

## FL Hyperparameter Contracts

These defaults are used by `fl/run_fl_simulation.py`, `fl/run_server.py`, and
`fl/run_client.py`. Override via CLI flags.

| Parameter | Default | Constraint | Notes |
|-----------|---------|------------|-------|
| `--clients` / `--min-clients` | 3 | ≥ 1 | Krum/Multi-Krum require ≥ 5 |
| `--rounds` | 10 | ≥ 1 | Must match between server and clients |
| `--local-epochs` | 3 | ≥ 1 | |
| `--lr` | 1e-3 | > 0 | Adam optimizer |
| `--batch-size` | 32 | ≥ 1 | |
| `--epsilon` | 10.0 | > 0 | DP target ε; values < 5 trigger preflight |
| `--max-grad-norm` | 2.0 | > 0 | DP clipping norm C; tune with `clipping_norm_sweep.py` |
| `--robust-agg` | `fedavg` | one of: `fedavg`, `krum`, `multi-krum`, `trimmed-mean`, `median` | `krum`/`multi-krum` blocked at n < 5 |
| `--byzantine-fraction` | 0.33 | 0 < f < 0.5 | Assumed fraction of Byzantine clients |
| `--strategy` | `fedavg` | `fedavg` \| `fedprox` | FedProx uses `--proximal-mu` |

### Krum client-count requirement

Krum's formal Byzantine robustness guarantee requires `n ≥ 2f + 3`.
With the default `--byzantine-fraction 0.33` and `f=1`, this means **≥ 5 clients**.
At n=3, `auto_f(3)=0` so Krum degenerates to selecting a single update with zero
Byzantine tolerance. The CLI enforces this: `--robust-agg krum` with `--clients < 5`
exits with code 1. Use `trimmed-mean` or `median` for the 3-client demo.

### DP preflight at low epsilon

When `--dp --epsilon <5` is passed, `dp_preflight.run_dp_preflight` runs before
training starts. It estimates the Opacus noise multiplier and:
- If `noise_multiplier > 3.0` with no saved sweep result → exits with code 1
- If a prior `clipping_norm_sweep` result exists → applies the recommended norm
- If `--auto-clipping-sweep` is set → runs a quick sweep and applies the best norm
- Pass `--skip-dp-preflight` to bypass (not recommended at ε < 5)
