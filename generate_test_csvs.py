"""
generate_test_csvs.py

Generates realistic test CSV files for IntelliClave prediction demo.

Files produced in  tests/csv/:
  test_balanced.csv        — 120 rows, equal 20 samples per class
  test_client1_bias.csv    — 60 rows, biased toward classes 0 & 1 (like Client 1)
  test_client2_bias.csv    — 60 rows, biased toward class 5   (like Client 2)
  test_client3_bias.csv    — 60 rows, biased toward class 4   (like Client 3)
  test_small.csv           — 18 rows, 3 per class (quick smoke test)
  test_hard.csv            — 60 rows, class 2 & 3 dominant (hardest for all models)
  test_single_row.csv      — 1 row  (single-sample sanity check)

Each CSV has columns: feature_0 … feature_N, label
Feature values are drawn from per-class Gaussian clusters so the model
can actually distinguish them (same distribution used during FL training).

Run:
    python generate_test_csvs.py

Requirements:
    pip install numpy pandas scikit-learn
"""

import os, json
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "csv")
INPUT_DIM   = 50       # must match model input_dim — adjust if yours differs
NUM_CLASSES = 6

# Try to read actual preprocessing from the project
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_prep_candidates = [
    os.path.join(_root, "results", "fl_rounds", "global_normalization.json"),
    os.path.join(_root, "results", "preprocessing.json"),
]
_global_mean = None
_global_std  = None
for _p in _prep_candidates:
    if os.path.exists(_p):
        with open(_p) as _f:
            _prep = json.load(_f)
        _global_mean = np.array(_prep.get("mean", [0.0] * INPUT_DIM))
        _global_std  = np.array(_prep.get("std",  [1.0] * INPUT_DIM))
        INPUT_DIM = len(_global_mean)
        print(f"[Info] Loaded preprocessing from {_p} (input_dim={INPUT_DIM})")
        break

if _global_mean is None:
    print(f"[Info] No preprocessing found — using synthetic mean/std (input_dim={INPUT_DIM})")
    np.random.seed(SEED)
    _global_mean = np.random.uniform(0.20, 0.35, INPUT_DIM)
    _global_std  = np.random.uniform(0.05, 0.12, INPUT_DIM)

np.random.seed(SEED)

# Per-class cluster centres (in normalised feature space)
# Each class lives in a distinct region so a trained MLP can separate them
CLASS_OFFSETS = np.array([
    [ 1.8,  0.0,  0.0,  0.0,  0.0],   # class 0
    [ 0.0,  1.8,  0.0,  0.0,  0.0],   # class 1
    [ 0.0,  0.0,  1.8,  0.0,  0.0],   # class 2
    [ 0.0,  0.0,  0.0,  1.8,  0.0],   # class 3
    [ 0.0,  0.0,  0.0,  0.0,  1.8],   # class 4
    [-1.8,  0.0,  0.0,  0.0,  0.0],   # class 5
])  # shape (6, 5) — first 5 features get a per-class bump; rest are noise

NOISE_STD = 0.6   # within-class scatter


def make_sample(cls: int) -> np.ndarray:
    """Return one raw (un-normalised) feature vector for a given class."""
    # Start from global mean
    x = _global_mean.copy().astype(np.float64)
    # Apply class-specific offset on first min(5, INPUT_DIM) features
    n_offset = min(5, INPUT_DIM)
    x[:n_offset] += CLASS_OFFSETS[cls][:n_offset] * _global_std[:n_offset]
    # Add within-class Gaussian noise on all features
    x += np.random.normal(0, NOISE_STD * _global_std, INPUT_DIM)
    return x


def make_df(class_counts: dict) -> pd.DataFrame:
    """
    class_counts: {class_id: n_samples}
    Returns a shuffled DataFrame with feature columns + label.
    """
    rows, labels = [], []
    for cls, n in class_counts.items():
        for _ in range(n):
            rows.append(make_sample(cls))
            labels.append(cls)
    df = pd.DataFrame(rows, columns=[f"feature_{i}" for i in range(INPUT_DIM)])
    df["label"] = labels
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ── Dataset definitions ───────────────────────────────────────────────────────
datasets = {
    "test_balanced": {
        "counts": {i: 20 for i in range(NUM_CLASSES)},
        "desc":   "120 rows — equal 20 samples per class (ideal balanced test)"
    },
    "test_client1_bias": {
        "counts": {0: 20, 1: 18, 2: 6, 3: 7, 4: 5, 5: 4},
        "desc":   "60 rows — class 0 & 1 dominant (mirrors Client 1 training distribution)"
    },
    "test_client2_bias": {
        "counts": {0: 2, 1: 5, 2: 10, 3: 4, 4: 1, 5: 38},
        "desc":   "60 rows — class 5 dominant (mirrors Client 2 training distribution)"
    },
    "test_client3_bias": {
        "counts": {0: 4, 1: 2, 2: 8, 3: 12, 4: 30, 5: 4},
        "desc":   "60 rows — class 4 dominant (mirrors Client 3 training distribution)"
    },
    "test_hard": {
        "counts": {0: 5, 1: 5, 2: 20, 3: 20, 4: 5, 5: 5},
        "desc":   "60 rows — class 2 & 3 dominant (hardest for local models)"
    },
    "test_small": {
        "counts": {i: 3 for i in range(NUM_CLASSES)},
        "desc":   "18 rows — 3 per class (quick smoke test)"
    },
    "test_single_row": {
        "counts": {3: 1},
        "desc":   "1 row — single sample sanity check (true label = 3)"
    },
}

# ── Write files ───────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\nGenerating {len(datasets)} test CSV files → {OUT_DIR}\n")
print(f"{'File':<30} {'Rows':>6}  Description")
print("-" * 75)

manifest = {}
for name, cfg in datasets.items():
    df = make_df(cfg["counts"])
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    manifest[name] = {
        "file":        f"{name}.csv",
        "rows":        len(df),
        "input_dim":   INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "class_dist":  {str(k): int(v) for k, v in cfg["counts"].items()},
        "description": cfg["desc"],
    }
    print(f"  {name+'.csv':<30} {len(df):>5}   {cfg['desc']}")

# Write manifest
manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n  manifest.json written → {manifest_path}")
print(f"\n✓ Done. Upload any of these CSVs on the Predictions page.")
print(f"  The model expects {INPUT_DIM} feature columns + an optional 'label' column.")