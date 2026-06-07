# UCI HAR Dataset/

Raw reference dataset for Human Activity Recognition (HAR).

## Purpose

This folder contains the **original UCI HAR Dataset** — 561 sensor features, 6 activity classes, subject IDs. It is the default raw input when running the data pipeline in textfiles mode. The pipeline splits subjects into three non-IID federated clients (FitLife, MediTrack, CareWatch narrative in reports).

## Key files

| Path | Content |
|------|---------|
| `train/X_train.txt`, `y_train.txt`, `subject_train.txt` | Training features, labels, subject IDs |
| `test/X_test.txt`, `y_test.txt`, `subject_test.txt` | Test split |
| `train/Inertial Signals/`, `test/Inertial Signals/` | Raw accelerometer/gyroscope channels |
| `features.txt` | Feature name list (561 features) |
| `activity_labels.txt` | Six activity class names |
| `README.txt` | Original UCI dataset documentation |

## How it connects

- Consumed by **`data/datascripts/pipeline.py --mode textfiles`**
- Output written to **`data/processed/client1.csv`**, `client2.csv`, `client3.csv`
- Analysis scripts: **`data/datascripts/verify_har.py`**, `har_analysis.py`, `matrix_analysis.py`
- Documented in **`contracts.md`** and **`report/`** dataset sections

## Commands

```bash
# Convert HAR text files → federated client CSVs
python data/datascripts/pipeline.py --mode textfiles

# Validate and analyse
python data/datascripts/verify_har.py
python data/datascripts/har_analysis.py
```

## Note

After pipeline runs, FL training uses **`data/processed/`** CSVs, not this raw folder directly.
