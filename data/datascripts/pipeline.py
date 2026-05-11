"""
data/datascripts/pipeline.py

Universal FL data preparation pipeline.

Supports three input modes — pick whichever matches your data:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE A — One CSV per client (simplest, recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each organisation already has their own CSV with a label column.
Just copy them to data/processed/ as client1.csv, client2.csv, etc.
No pipeline script needed — go straight to training.

Or use this script to validate and plot them:
    python data/datascripts/pipeline.py --mode per-client \\
        --client-csvs hospital_a.csv hospital_b.csv hospital_c.csv \\
        --label-col diagnosis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE B — One combined CSV, split into clients
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have one big CSV. Split it by a grouping column (e.g. site_id, region)
or randomly into N equal parts.

    # Split by a column:
    python data/datascripts/pipeline.py --mode split \\
        --csv data.csv --label-col outcome --split-col site_id

    # Random split into 4 clients:
    python data/datascripts/pipeline.py --mode split \\
        --csv data.csv --label-col outcome --n-clients 4

    # With optional PCA dimensionality reduction:
    python data/datascripts/pipeline.py --mode split \\
        --csv data.csv --label-col outcome --split-col site_id --pca-components 20

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE C — Raw numeric text files (UCI HAR format, backward-compatible)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For datasets distributed as separate X/y/subject text files.

    python data/datascripts/pipeline.py --mode textfiles \\
        --x-train X_train.txt --y-train y_train.txt \\
        --x-test  X_test.txt  --y-test  y_test.txt  \\
        --subjects-train subject_train.txt \\
        --subjects-test  subject_test.txt  \\
        --n-clients 3 --no-pca

    # Default (UCI HAR dataset in project root):
    python data/datascripts/pipeline.py --mode textfiles

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output: data/processed/client1.csv, client2.csv, ...
Each CSV has feature columns + a "label" column.
Run data/datascripts/weights.py afterwards to recompute class weights.
"""
import argparse
import os
import pickle
import shutil

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OUT = os.path.join(_ROOT, "data", "processed")

# UCI HAR defaults (Mode C only)
_HAR_BASE           = os.path.join(_ROOT, "UCI HAR Dataset")
_DEFAULT_X_TRAIN    = os.path.join(_HAR_BASE, "train", "X_train.txt")
_DEFAULT_Y_TRAIN    = os.path.join(_HAR_BASE, "train", "y_train.txt")
_DEFAULT_X_TEST     = os.path.join(_HAR_BASE, "test",  "X_test.txt")
_DEFAULT_Y_TEST     = os.path.join(_HAR_BASE, "test",  "y_test.txt")
_DEFAULT_SUBJ_TRAIN = os.path.join(_HAR_BASE, "train", "subject_train.txt")
_DEFAULT_SUBJ_TEST  = os.path.join(_HAR_BASE, "test",  "subject_test.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_pca(X: np.ndarray, X_fit: np.ndarray, n_components: int,
               model_path: str, fit: bool) -> tuple:
    """Apply or load PCA. Returns (X_transformed, feature_col_names)."""
    if fit:
        print(f"\n=== Fitting PCA (n_components={n_components}) ===")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_fit)
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(pca, f)
        print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"  PCA model saved -> {model_path}")
    else:
        print(f"\n=== Loading PCA model from {model_path} ===")
        with open(model_path, "rb") as f:
            pca = pickle.load(f)

    X_out = pca.transform(X)
    cols  = [f"pca_{i}" for i in range(X_out.shape[1])]
    print(f"  Shape after PCA: {X_out.shape}")
    return X_out, cols


def _encode_labels(y_raw: np.ndarray) -> tuple:
    """
    Encode labels to 0-based integers regardless of original type.
    Returns (y_encoded, original_unique_values).
    """
    unique = sorted(np.unique(y_raw).tolist(), key=str)
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique):
        # Numeric: shift to 0-based
        offset = int(min(unique))
        return y_raw.astype(np.int64) - offset, unique
    else:
        # String: alphabetical encoding
        le = LabelEncoder()
        return le.fit_transform(y_raw.astype(str)), list(le.classes_)


def _save_client_csvs(splits: list, client_names: list, out_dir: str) -> dict:
    """Write per-client DataFrames to out_dir. Returns {name: df}."""
    os.makedirs(out_dir, exist_ok=True)
    saved = {}
    for name, df in zip(client_names, splits):
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        saved[name] = df
        feat_cols = [c for c in df.columns if c != "label"]
        print(f"\n  {name}: {len(df)} rows, {len(feat_cols)} features -> {path}")
        print(df["label"].value_counts().sort_index().to_string())
    return saved


def _verify(saved: dict):
    """Basic integrity checks on all saved CSVs."""
    print("\n=== Verifying CSVs ===")
    ref_cols = None
    for name, df in saved.items():
        feat_cols = [c for c in df.columns if c != "label"]
        nan_count = df.isna().sum().sum()
        inf_count = np.isinf(df[feat_cols].values.astype(float)).sum()
        classes   = sorted(df["label"].unique().tolist())
        ok = nan_count == 0 and inf_count == 0
        print(f"  {'✓' if ok else '✗'} {name}: rows={len(df)}, "
              f"features={len(feat_cols)}, NaN={nan_count}, "
              f"Inf={inf_count}, classes={classes}")
        assert nan_count == 0, f"{name} has NaN values"
        assert inf_count == 0, f"{name} has Inf values"
        if ref_cols is None:
            ref_cols = set(feat_cols)
        else:
            assert set(feat_cols) == ref_cols, \
                f"{name} has different feature columns than the first client"
    print("  All checks passed ✓")


def _plot(saved: dict, out_dir: str):
    """Save a class distribution chart."""
    names   = list(saved.keys())
    n       = len(names)
    n_cols  = min(n, 4)
    n_rows  = (n + n_cols - 1) // n_cols
    colors  = ["#4C72B0","#DD8452","#55A868","#C44E52",
               "#8172B2","#937860","#DA8BC3","#8C8C8C"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, name in zip(flat, names):
        df     = saved[name]
        counts = df["label"].value_counts().sort_index()
        bars   = ax.bar([str(l) for l in counts.index], counts.values,
                        color=[colors[i % len(colors)] for i in range(len(counts))])
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Samples")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(val),
                    ha="center", va="bottom", fontsize=8)

    for ax in flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Class Distribution per FL Client", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart = os.path.join(_ROOT, "data", "client_distributions.png")
    plt.savefig(chart, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved -> {chart}")


def _finish(saved: dict, out_dir: str):
    print("\n=== Done ===")
    for name in saved:
        print(f"  {out_dir}/{name}.csv")
    print("\nNext steps:")
    print("  1. python data/datascripts/weights.py   # recompute class weights")
    print("  2. python fl/run_fl_simulation.py        # run FL training")


# ─────────────────────────────────────────────────────────────────────────────
# MODE A — one CSV per client
# ─────────────────────────────────────────────────────────────────────────────

def mode_per_client(
    client_csvs: list,
    label_col: str,
    out_dir: str,
    pca_components: int,
    pca_model_path: str,
    use_pca: bool,
    fit_pca: bool,
):
    """
    Validate and copy/transform one CSV per client into data/processed/.
    Each CSV must have a label column (named by --label-col).
    """
    print(f"=== Mode A: per-client CSVs ({len(client_csvs)} clients) ===")
    client_names = [f"client{i+1}" for i in range(len(client_csvs))]
    splits = []

    for path, name in zip(client_csvs, client_names):
        print(f"\n  Loading {path}...")
        df = pd.read_csv(path).dropna()

        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in {path}. "
                f"Columns: {list(df.columns)}"
            )

        feat_cols = [c for c in df.columns if c != label_col]
        X = df[feat_cols].values.astype(np.float32)
        y_raw = df[label_col].values
        y_enc, _ = _encode_labels(y_raw)

        if use_pca:
            fit = fit_pca or not os.path.exists(pca_model_path)
            X, feat_cols = _apply_pca(X, X, pca_components, pca_model_path, fit)
            fit_pca = False  # only fit on first client

        out_df = pd.DataFrame(X, columns=feat_cols)
        out_df["label"] = y_enc
        splits.append(out_df)

    saved = _save_client_csvs(splits, client_names, out_dir)
    _verify(saved)
    _plot(saved, out_dir)
    _finish(saved, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# MODE B — one combined CSV, split into clients
# ─────────────────────────────────────────────────────────────────────────────

def mode_split(
    csv_path: str,
    label_col: str,
    split_col: str,
    n_clients: int,
    out_dir: str,
    pca_components: int,
    pca_model_path: str,
    use_pca: bool,
    fit_pca: bool,
):
    """
    Load one combined CSV and split it into per-client CSVs.
    Split by a grouping column (e.g. site_id) or randomly.
    """
    print(f"=== Mode B: combined CSV split ===")
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path).dropna()
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. Columns: {list(df.columns)}"
        )

    # Encode labels
    y_enc, unique_labels = _encode_labels(df[label_col].values)
    df = df.copy()
    df["label"] = y_enc

    # Drop original label col if it was renamed
    if label_col != "label":
        df = df.drop(columns=[label_col])

    # Split
    if split_col and split_col in df.columns:
        groups  = sorted(df[split_col].unique().tolist(), key=str)
        chunks  = np.array_split(groups, n_clients)
        splits  = [df[df[split_col].isin(c)].drop(columns=[split_col]).reset_index(drop=True)
                   for c in chunks]
        print(f"  Split by '{split_col}': {len(groups)} groups → {n_clients} clients")
    else:
        if split_col:
            print(f"  WARNING: split column '{split_col}' not found — using random split")
        df_shuf = df.sample(frac=1, random_state=42).reset_index(drop=True)
        splits  = [s.reset_index(drop=True) for s in np.array_split(df_shuf, n_clients)]
        print(f"  Random split into {n_clients} clients")

    # Optional PCA
    if use_pca:
        feat_cols = [c for c in splits[0].columns if c != "label"]
        X_all = np.vstack([s[feat_cols].values.astype(np.float32) for s in splits])
        fit   = fit_pca or not os.path.exists(pca_model_path)
        X_all_pca, pca_cols = _apply_pca(X_all, X_all, pca_components, pca_model_path, fit)

        # Re-split transformed data
        sizes  = [len(s) for s in splits]
        starts = [0] + list(np.cumsum(sizes[:-1]))
        new_splits = []
        for s, start, size in zip(splits, starts, sizes):
            out_df = pd.DataFrame(X_all_pca[start:start+size], columns=pca_cols)
            out_df["label"] = s["label"].values
            new_splits.append(out_df)
        splits = new_splits

    client_names = [f"client{i+1}" for i in range(n_clients)]
    saved = _save_client_csvs(splits, client_names, out_dir)
    _verify(saved)
    _plot(saved, out_dir)
    _finish(saved, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# MODE C — raw numeric text files (UCI HAR format)
# ─────────────────────────────────────────────────────────────────────────────

def mode_textfiles(
    x_train_path: str,
    y_train_path: str,
    x_test_path: str,
    y_test_path: str,
    subjects_train_path: str,
    subjects_test_path: str,
    n_clients: int,
    out_dir: str,
    pca_components: int,
    pca_model_path: str,
    use_pca: bool,
    fit_pca: bool,
):
    """
    Load X/y/subject text files, optionally apply PCA, split by subject groups.
    Backward-compatible with the original UCI HAR pipeline.
    """
    print("=== Mode C: raw text files ===")
    X_train = np.loadtxt(x_train_path)
    y_train = np.loadtxt(y_train_path, dtype=int)
    X_test  = np.loadtxt(x_test_path)
    y_test  = np.loadtxt(y_test_path, dtype=int)
    s_train = np.loadtxt(subjects_train_path, dtype=int)
    s_test  = np.loadtxt(subjects_test_path, dtype=int)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    s_all = np.concatenate([s_train, s_test])
    print(f"  X: {X_all.shape}, y: {y_all.shape}, subjects: {s_all.shape}")

    y_enc, _ = _encode_labels(y_all)

    if use_pca:
        fit = fit_pca or not os.path.exists(pca_model_path)
        X_all, feat_cols = _apply_pca(X_all, X_train, pca_components, pca_model_path, fit)
    else:
        feat_cols = [f"feat_{i}" for i in range(X_all.shape[1])]
        print(f"  Using {len(feat_cols)} raw features (no PCA)")

    # Split by subject groups
    unique_subjects = np.unique(s_all)
    groups   = np.array_split(unique_subjects, n_clients)
    masks    = [np.isin(s_all, g) for g in groups]
    client_names = [f"client{i+1}" for i in range(n_clients)]

    splits = []
    for mask in masks:
        df = pd.DataFrame(X_all[mask], columns=feat_cols)
        df["label"] = y_enc[mask]
        splits.append(df)

    saved = _save_client_csvs(splits, client_names, out_dir)
    _verify(saved)
    _plot(saved, out_dir)
    _finish(saved, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IntelliClave universal FL data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES

  Mode A — one CSV per client (healthcare, finance, any industry):
    python data/datascripts/pipeline.py --mode per-client \\
        --client-csvs hospital_a.csv hospital_b.csv hospital_c.csv \\
        --label-col diagnosis

  Mode B — one combined CSV, split by site:
    python data/datascripts/pipeline.py --mode split \\
        --csv combined_data.csv --label-col outcome --split-col site_id

  Mode B — one combined CSV, random split into 4 clients:
    python data/datascripts/pipeline.py --mode split \\
        --csv combined_data.csv --label-col target --n-clients 4

  Mode B — with PCA dimensionality reduction:
    python data/datascripts/pipeline.py --mode split \\
        --csv combined_data.csv --label-col target \\
        --split-col region --pca-components 20

  Mode C — UCI HAR text files (default):
    python data/datascripts/pipeline.py --mode textfiles

  Mode C — custom text files, no PCA:
    python data/datascripts/pipeline.py --mode textfiles \\
        --x-train X.txt --y-train y.txt \\
        --subjects-train s_train.txt --subjects-test s_test.txt --no-pca
        """,
    )

    parser.add_argument(
        "--mode", choices=["per-client", "split", "textfiles"],
        default="textfiles",
        help="Input mode: per-client | split | textfiles (default: textfiles)",
    )

    # ── Mode A ────────────────────────────────────────────────────────────────
    parser.add_argument("--client-csvs", nargs="+", default=None,
                        help="[Mode A] Paths to one CSV per client.")

    # ── Mode B ────────────────────────────────────────────────────────────────
    parser.add_argument("--csv",       default=None,
                        help="[Mode B] Path to combined CSV.")
    parser.add_argument("--split-col", default=None,
                        help="[Mode B] Column to split clients by (e.g. site_id). "
                             "Omit for random split.")

    # ── Mode C ────────────────────────────────────────────────────────────────
    parser.add_argument("--x-train",        default=_DEFAULT_X_TRAIN)
    parser.add_argument("--y-train",        default=_DEFAULT_Y_TRAIN)
    parser.add_argument("--x-test",         default=_DEFAULT_X_TEST)
    parser.add_argument("--y-test",         default=_DEFAULT_Y_TEST)
    parser.add_argument("--subjects-train", default=_DEFAULT_SUBJ_TRAIN)
    parser.add_argument("--subjects-test",  default=_DEFAULT_SUBJ_TEST)

    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument("--label-col",      default="label",
                        help="Name of the label/target column (default: label).")
    parser.add_argument("--n-clients",      type=int, default=3,
                        help="Number of FL clients (default: 3).")
    parser.add_argument("--out-dir",        default=_DEFAULT_OUT,
                        help="Output directory for client CSVs.")
    parser.add_argument("--pca-components", type=int, default=50,
                        help="PCA components to keep (ignored if --no-pca).")
    parser.add_argument("--pca-model",      default=os.path.join(_ROOT, "data", "samples", "pca_model.pkl"),
                        help="Path to save/load PCA model.")
    parser.add_argument("--no-pca",         action="store_true",
                        help="Skip PCA — use raw features as-is.")
    parser.add_argument("--fit-pca",        action="store_true",
                        help="Fit a new PCA model even if one already exists.")

    args = parser.parse_args()
    use_pca  = not args.no_pca
    fit_pca  = args.fit_pca or not os.path.exists(args.pca_model)

    if args.mode == "per-client":
        if not args.client_csvs:
            parser.error("--mode per-client requires --client-csvs")
        mode_per_client(
            client_csvs=args.client_csvs,
            label_col=args.label_col,
            out_dir=args.out_dir,
            pca_components=args.pca_components,
            pca_model_path=args.pca_model,
            use_pca=use_pca,
            fit_pca=fit_pca,
        )

    elif args.mode == "split":
        if not args.csv:
            parser.error("--mode split requires --csv")
        mode_split(
            csv_path=args.csv,
            label_col=args.label_col,
            split_col=args.split_col,
            n_clients=args.n_clients,
            out_dir=args.out_dir,
            pca_components=args.pca_components,
            pca_model_path=args.pca_model,
            use_pca=use_pca,
            fit_pca=fit_pca,
        )

    else:  # textfiles
        mode_textfiles(
            x_train_path=args.x_train,
            y_train_path=args.y_train,
            x_test_path=args.x_test,
            y_test_path=args.y_test,
            subjects_train_path=args.subjects_train,
            subjects_test_path=args.subjects_test,
            n_clients=args.n_clients,
            out_dir=args.out_dir,
            pca_components=args.pca_components,
            pca_model_path=args.pca_model,
            use_pca=use_pca,
            fit_pca=fit_pca,
        )
