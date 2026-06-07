# data/

Dataset ingestion, preprocessing, and federated client partitioning.

## Purpose

This folder turns raw tabular data (or the bundled UCI HAR dataset) into **per-client CSV files** in `processed/`. The pipeline is dataset-agnostic: provide CSVs with a label column and the rest of the system infers feature count, class names, and schema automatically.

## Key files

| Path | Role |
|------|------|
| `processed/client1.csv` … `client3.csv` | Frozen non-IID client splits used by FL training |
| `class_weights.json` | Per-class weights for imbalanced loss |
| `datascripts/pipeline.py` | Main prep script (per-client, split, or textfiles mode) |
| `datascripts/weights.py` | Recompute class weights after repartitioning |
| `datascripts/check_data.py` | Validate schema, NaNs, labels |
| `datascripts/har_analysis.py` | UCI HAR distribution analysis |
| `datascripts/verify_har.py` | HAR integrity checks |

## How it connects

- **`UCI HAR Dataset/`** — default raw input for `--mode textfiles`
- **`fl/data_utils.py`** — loads `processed/*.csv` for training and evaluation
- **`evaluation/cross_validation.py`** — uses the same client CSVs
- **`docker/`** / **`kubernetes/`** — mount `data/processed/` read-only into containers

## Commands

```bash
# One CSV per hospital/client
python data/datascripts/pipeline.py --mode per-client \
    --client-csvs a.csv b.csv c.csv --label-col diagnosis

# Split one combined CSV (Dirichlet non-IID)
python data/datascripts/pipeline.py --mode split \
    --csv combined.csv --label-col outcome --n-clients 3

# UCI HAR text files → client CSVs
python data/datascripts/pipeline.py --mode textfiles

python data/datascripts/weights.py
python data/datascripts/check_data.py
```
