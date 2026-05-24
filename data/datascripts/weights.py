"""
data/datascripts/weights.py

Compute balanced class weights from all client CSVs and write
data/class_weights.json (integer-keyed, 0-based).

Works with any dataset — class names and count are inferred from the data.

Usage:
    python data/datascripts/weights.py
    python data/datascripts/weights.py --processed-dir data/processed --label-col label
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def compute_weights(processed_dir: str, label_col: str, out_path: str):
    # Discover all CSVs in the processed directory
    csv_files = sorted(
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir)
        if f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {processed_dir}")

    print("=== Computing class weights ===")
    y_all = []
    for path in csv_files:
        df = pd.read_csv(path)
        y_all.extend(df[label_col].tolist())
        print(f"  {os.path.basename(path)}: {len(df)} rows loaded")

    y_arr = np.array(y_all)
    unique_labels = sorted(np.unique(y_arr).tolist())
    n_classes = len(unique_labels)

    # Shift to 0-based for sklearn
    offset = int(min(unique_labels))
    y_zero = y_arr.astype(np.int64) - offset

    print(f"\nTotal samples : {len(y_arr)}")
    print(f"Classes       : {unique_labels}")
    print(f"Class counts  : {np.bincount(y_zero)}")

    weights = compute_class_weight(
        "balanced",
        classes=np.arange(n_classes),
        y=y_zero,
    )

    # Save as integer-keyed (0-based)
    weights_dict = {str(i): round(float(w), 6) for i, w in enumerate(weights)}

    print("\nClass weights (0-based index):")
    for idx, w in weights_dict.items():
        print(f"  class_{idx:<4} {w:.6f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Verify reload
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == n_classes
    print("Reload verified ✓")


if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(
        description="Compute balanced class weights from processed client CSVs."
    )
    parser.add_argument(
        "--processed-dir",
        default=os.path.join(_root, "data", "processed"),
        help="Directory containing client CSV files.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the label column (default: label).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_root, "data", "class_weights.json"),
        help="Output path for class_weights.json.",
    )
    args = parser.parse_args()
    compute_weights(args.processed_dir, args.label_col, args.out)
