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
