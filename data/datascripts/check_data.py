"""
data/datascripts/check_data.py

Generic data integrity check for processed client CSVs.

Verifies:
  - All expected CSV files exist and are non-empty
  - No NaN or Inf values
  - All CSVs share the same feature schema
  - class_weights.json exists and has the right number of entries
  - PCA model (if present) loads correctly

Usage:
    python data/datascripts/check_data.py
    python data/datascripts/check_data.py --processed-dir data/processed
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd


def run_checks(processed_dir: str, data_dir: str):
    all_ok = True

    # ── 1. Client CSVs ────────────────────────────────────────────────────────
    print("\n[1] Client CSVs")
    csv_files = sorted(
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir)
        if f.endswith(".csv")
    )
    if not csv_files:
        print(f"  ✗ No CSV files found in {processed_dir}")
        all_ok = False
    else:
        schemas     = []
        label_sets  = []
        for path in csv_files:
            name = os.path.basename(path)
            df   = pd.read_csv(path)
            feat_cols = [c for c in df.columns if c != "label"]
            nan_count = df.isna().sum().sum()
            inf_count = np.isinf(df[feat_cols].values.astype(float)).sum()
            classes   = sorted(df["label"].unique().tolist())
            n_feat    = len(feat_cols)
            schemas.append((n_feat, list(feat_cols)))   # ordered list, not set
            label_sets.append((name, set(classes)))

            status = "✓" if nan_count == 0 and inf_count == 0 else "✗"
            print(
                f"  {status} {name}: rows={len(df)}, features={n_feat}, "
                f"NaN={nan_count}, Inf={inf_count}, classes={classes}"
            )
            if nan_count > 0 or inf_count > 0:
                all_ok = False

        # ── Feature schema consistency ────────────────────────────────────────
        ref_n, ref_cols = schemas[0]
        schema_ok = True
        for (n, cols), path in zip(schemas[1:], csv_files[1:]):
            name = os.path.basename(path)
            if n != ref_n:
                print(f"  ✗ {name}: {n} features vs {ref_n} in reference — count mismatch!")
                schema_ok = False
            elif cols != ref_cols:
                if set(cols) == set(ref_cols):
                    print(f"  ✗ {name}: same columns but different ORDER — "
                          f"this will silently corrupt model weights!")
                else:
                    missing = set(ref_cols) - set(cols)
                    extra   = set(cols) - set(ref_cols)
                    print(f"  ✗ {name}: column mismatch — "
                          f"missing={sorted(missing)}, extra={sorted(extra)}")
                schema_ok = False
        if schema_ok:
            print(f"  ✓ All CSVs share the same feature schema "
                  f"({ref_n} features, same order)")
        else:
            all_ok = False

        # ── Label set consistency ─────────────────────────────────────────────
        ref_name, ref_labels = label_sets[0]
        label_ok = True
        for name, labels in label_sets[1:]:
            if labels != ref_labels:
                missing = ref_labels - labels
                extra   = labels - ref_labels
                print(f"  ✗ Label mismatch: {name} vs {ref_name}")
                if missing:
                    print(f"    Classes in {ref_name} but not {name}: {sorted(missing)}")
                if extra:
                    print(f"    Classes in {name} but not {ref_name}: {sorted(extra)}")
                print(f"    This means the model will never see some classes "
                      f"from certain clients during training.")
                label_ok = False
        if label_ok:
            print(f"  ✓ All CSVs share the same label set: {sorted(ref_labels)}")
        else:
            all_ok = False

    # ── 2. PCA model (optional) ───────────────────────────────────────────────
    print("\n[2] PCA Model (optional)")
    pca_path = os.path.join(data_dir, "samples", "pca_model.pkl")
    if os.path.exists(pca_path):
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
            var = pca.explained_variance_ratio_.sum()
            print(
                f"  ✓ pca_model.pkl: n_components={pca.n_components_}, "
                f"explained_variance={var:.4f}"
            )
        except Exception as e:
            print(f"  ✗ pca_model.pkl failed to load: {e}")
            all_ok = False
    else:
        print("  — pca_model.pkl not found (only needed if PCA was used in pipeline)")

    # ── 3. Class weights ──────────────────────────────────────────────────────
    print("\n[3] Class Weights")
    cw_path = os.path.join(data_dir, "class_weights.json")
    if os.path.exists(cw_path):
        with open(cw_path) as f:
            cw = json.load(f)
        # Filter out comment keys
        weight_entries = {k: v for k, v in cw.items() if not k.startswith("_")}
        vals_ok = all(0.1 < v < 10.0 for v in weight_entries.values())
        print(
            f"  {'✓' if vals_ok else '✗'} class_weights.json: "
            f"{len(weight_entries)} entries, values_in_range={'✓' if vals_ok else '✗'}"
        )
        for k, v in weight_entries.items():
            print(f"    [{k}]  {v:.6f}")
        if not vals_ok:
            all_ok = False
    else:
        print("  — class_weights.json not found (weighted loss will be disabled)")

    # ── 4. Other artefacts ────────────────────────────────────────────────────
    print("\n[4] Other Artefacts")
    for fname in ["client_distributions.png"]:
        path = os.path.join(data_dir, fname)
        print(f"  {'✓' if os.path.exists(path) else '—'} {fname}")

    # ── 5. Script inventory ───────────────────────────────────────────────────
    print("\n[5] Scripts in data/datascripts/")
    scripts_dir = os.path.join(data_dir, "datascripts")
    for f in sorted(os.listdir(scripts_dir)):
        print(f"  {f}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED ✓' if all_ok else 'ISSUES FOUND — see above ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _data_dir = os.path.join(_root, "data")

    parser = argparse.ArgumentParser(
        description="Generic data integrity check for processed client CSVs."
    )
    parser.add_argument(
        "--processed-dir",
        default=os.path.join(_data_dir, "processed"),
        help="Directory containing client CSV files.",
    )
    args = parser.parse_args()
    run_checks(args.processed_dir, _data_dir)
