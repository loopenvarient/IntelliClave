# evaluation/

Offline model evaluation and report figures.

## Purpose

This folder evaluates model quality **outside** the live FL loop: stratified cross-validation on combined client data and generation of summary figures for reports and demos.

## Key files

| Path | Role |
|------|------|
| `cross_validation.py` | K-fold stratified CV → `results/cross_validation.json` |
| `metrics.py` | Accuracy, macro-F1, per-class F1, AUC-ROC helpers |
| `generate_graph6.py` | 4-panel final results figure → `results/graphs/graph6_final_results.png` |

## How it connects

- Uses **`fl/model.py`** and **`fl/data_utils.py`**
- Reads aggregated metrics from **`results/`** (FL runs, attacks, ε sweep, benchmarks)
- Dashboard may use per-class F1 computed at runtime; CV JSON is a separate local-training baseline
- Figures referenced in **`report/figures/`** and presentation materials

## Commands

```bash
python evaluation/cross_validation.py --folds 5 --epochs 10
python evaluation/generate_graph6.py
```

## Note

Cross-validation here measures **centralized per-client training**, not the federated global model directly. FL global metrics come from **`fl/`** runs and **`results/results.json`**.
