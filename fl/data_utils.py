import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Imputation strategies supported by load_csv_data
ImputeStrategy = Literal["median", "mean", "mode", "drop"]


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_processed_data_dir() -> str:
    return os.path.join(get_project_root(), "data", "processed")


def get_default_client_csvs(n_clients: int = 3) -> List[str]:
    return [
        os.path.join(get_processed_data_dir(), f"client{i}.csv")
        for i in range(1, n_clients + 1)
    ]


def get_class_weights_path() -> str:
    return os.path.join(get_project_root(), "data", "class_weights.json")


def get_preprocessing_metadata_path(base_path: str) -> str:
    """
    Return the preprocessing metadata path for a run directory or checkpoint path.
    """
    if base_path.endswith(".pth"):
        base_path = os.path.dirname(base_path)
    return os.path.join(base_path, "preprocessing.json")


def save_preprocessing_metadata(
    base_path: str,
    feature_names: List[str],
    mean: np.ndarray,
    std: np.ndarray,
    normalization: str = "global",
) -> str:
    """
    Persist the preprocessing stats used during training alongside a checkpoint.
    """
    path = get_preprocessing_metadata_path(base_path)
    payload = {
        "normalization": normalization,
        "feature_names": feature_names,
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_preprocessing_metadata(base_path: str) -> Optional[Dict]:
    """
    Load preprocessing metadata written by save_preprocessing_metadata().
    """
    path = get_preprocessing_metadata_path(base_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    payload["mean"] = np.array(payload["mean"], dtype=np.float32)
    payload["std"] = np.array(payload["std"], dtype=np.float32)
    return payload


@dataclass
class DatasetMetadata:
    input_dim: int
    num_classes: int
    class_names: List[str]
    feature_names: List[str]
    train_size: int
    test_size: int
    # Task type — "classification" or "regression"
    task: str = "classification"


@dataclass
class ScalerStats:
    """
    Per-feature mean and std computed from a client's training split.
    Shared with the server during the normalization coordination round
    so that a global scaler can be broadcast back to all clients.
    No raw data leaves the client.
    """
    mean: np.ndarray
    std: np.ndarray
    n_samples: int

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_samples": self.n_samples,
        }

    @staticmethod
    def from_dict(d: Dict) -> "ScalerStats":
        return ScalerStats(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
            n_samples=int(d["n_samples"]),
        )


def aggregate_scaler_stats(stats_list: List[ScalerStats]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and std from a list of per-client ScalerStats.

    Uses a weighted combination so that clients with more samples contribute
    proportionally more to the global statistics.

    Returns (global_mean, global_std) as numpy arrays.
    """
    total_n = sum(s.n_samples for s in stats_list)
    global_mean = sum(s.mean * s.n_samples for s in stats_list) / total_n

    # Pooled variance: Var_global = weighted avg of (var_i + (mean_i - mean_global)^2)
    global_var = sum(
        s.n_samples * (s.std ** 2 + (s.mean - global_mean) ** 2)
        for s in stats_list
    ) / total_n
    global_std = np.sqrt(global_var)
    # Avoid division by zero for constant features
    global_std = np.where(global_std < 1e-8, 1.0, global_std)
    return global_mean, global_std


def infer_default_preprocessing(n_clients: int = 3) -> Optional[Dict]:
    """
    Backward-compatible fallback for legacy runs that did not save preprocessing.
    """
    csv_paths = [path for path in get_default_client_csvs(n_clients) if os.path.exists(path)]
    if not csv_paths:
        return None
    stats_list = [compute_client_scaler_stats(path) for path in csv_paths]
    mean, std = aggregate_scaler_stats(stats_list)
    _, feature_names = infer_csv_schema(csv_paths[0])
    return {
        "normalization": "global",
        "feature_names": feature_names,
        "mean": mean,
        "std": std,
    }


class CSVDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class RegressionCSVDataset(Dataset):
    """Dataset for regression tasks — labels are float tensors."""
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def validate_client_schemas(
    csv_paths: List[str],
    target_col: str = "label",
) -> None:
    """
    Assert that all client CSVs share the same feature columns in the same order.

    Raises ValueError with a clear message if any mismatch is found.
    Call this before starting FL training to catch misaligned datasets early.
    """
    if len(csv_paths) < 2:
        return

    reference_path = csv_paths[0]
    reference_df = pd.read_csv(reference_path, nrows=1)
    reference_cols = [c for c in reference_df.columns if c != target_col]

    errors = []
    for path in csv_paths[1:]:
        df = pd.read_csv(path, nrows=1)
        cols = [c for c in df.columns if c != target_col]

        if cols != reference_cols:
            missing    = set(reference_cols) - set(cols)
            extra      = set(cols) - set(reference_cols)
            wrong_order = (set(cols) == set(reference_cols)) and (cols != reference_cols)

            msg = (f"\n  Schema mismatch: {os.path.basename(path)} "
                   f"vs {os.path.basename(reference_path)}")
            if missing:
                msg += f"\n    Missing columns : {sorted(missing)}"
            if extra:
                msg += f"\n    Extra columns   : {sorted(extra)}"
            if wrong_order:
                msg += (f"\n    Column order differs (order matters for model weights)"
                        f"\n    Expected: {reference_cols}"
                        f"\n    Got     : {cols}")
            errors.append(msg)

    if errors:
        raise ValueError(
            "Client CSV schema validation failed — all clients must have identical "
            "feature columns in the same order:" + "".join(errors)
        )

    print(f"[data_utils] Schema validation passed — {len(csv_paths)} clients, "
          f"{len(reference_cols)} features each ✓")


def infer_csv_schema(csv_path: str, target_col: str = "label") -> Tuple[int, List[str]]:
    """Return (n_features, feature_names) by reading the CSV header."""
    df = pd.read_csv(csv_path, nrows=1)
    feature_names = [col for col in df.columns if col != target_col]
    if not feature_names:
        raise ValueError(f"No feature columns found in {csv_path}")
    return len(feature_names), feature_names


def infer_class_names(csv_path: str, target_col: str = "label") -> List[str]:
    """
    Infer ordered class names from the CSV.
    Numeric labels → ["class_0", "class_1", ...].
    String labels  → sorted unique values.
    """
    df = pd.read_csv(csv_path, usecols=[target_col])
    unique = sorted(df[target_col].dropna().unique().tolist())
    if all(isinstance(v, (int, float)) for v in unique):
        return [f"class_{int(v)}" for v in unique]
    return [str(v) for v in unique]


def compute_client_scaler_stats(
    csv_path: str,
    target_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    impute: ImputeStrategy = "median",
) -> ScalerStats:
    """
    Compute mean and std from a client's training split only (no leakage).
    Called on each client before FL training starts; the result is sent to
    the server for global normalization coordination.
    No raw feature values are exposed — only summary statistics.
    """
    df = pd.read_csv(csv_path)
    feature_names = [c for c in df.columns if c != target_col]

    # Apply same imputation as load_csv_data
    if df.isna().sum().sum() > 0:
        if impute == "drop":
            df = df.dropna()
        elif impute in ("median", "mean"):
            fill = df[feature_names].median() if impute == "median" else df[feature_names].mean()
            df[feature_names] = df[feature_names].fillna(fill)
        elif impute == "mode":
            for col in feature_names:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

    X = df[feature_names].values.astype(np.float32)
    y = df[target_col].values

    # Use only the training split to compute stats (same split as load_csv_data)
    unique = sorted(np.unique(y).tolist())
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique):
        offset = int(min(unique))
        y_enc = y.astype(np.int64) - offset
    else:
        label_map = {v: i for i, v in enumerate(unique)}
        y_enc = np.array([label_map[v] for v in y], dtype=np.int64)

    X_tr, _, _, _ = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
    return ScalerStats(mean=mean, std=std, n_samples=len(X_tr))


def load_class_weights(
    num_classes: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    Load class weights from data/class_weights.json.
    Supports integer-keyed {"0": w} and legacy name-keyed {"WALKING": w} formats.
    Returns None if the file does not exist or the count doesn't match.
    """
    weights_path = get_class_weights_path()
    if not os.path.exists(weights_path):
        return None

    with open(weights_path) as f:
        weights_dict: Dict = json.load(f)

    keys = [k for k in weights_dict.keys() if not k.startswith("_")]
    try:
        ordered = [weights_dict[k] for k in sorted(keys, key=int)]
    except ValueError:
        ordered = [weights_dict[k] for k in sorted(keys)]

    if len(ordered) != num_classes:
        print(
            f"[data_utils] WARNING: class_weights.json has {len(ordered)} entries "
            f"but dataset has {num_classes} classes — ignoring weights."
        )
        return None

    tensor = torch.tensor(ordered, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_csv_data(
    csv_path: str,
    target_col: str = "label",
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
    label_offset: Optional[int] = None,
    impute: ImputeStrategy = "median",
    task: str = "classification",
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    drop_last_for_dp: bool = False,
):
    """
    Load any classification or regression CSV, impute missing values,
    scale features, stratify-split, and return DataLoaders + DatasetMetadata.

    Missing value handling
    ----------------------
    impute="median"  : fill numeric NaNs with column median (default)
    impute="mean"    : fill numeric NaNs with column mean
    impute="mode"    : fill all NaNs with column mode
    impute="drop"    : drop rows with any NaN (loses data — warns you)

    Task
    ----
    task="classification" : labels encoded as integers, CrossEntropyLoss compatible
    task="regression"     : labels kept as floats, MSELoss compatible

    Normalization
    -------------
    drop_last_for_dp : bool  — set True when using Opacus DP training.
                               Opacus requires the last incomplete batch to be
                               dropped. Default False (no data loss for non-DP).
                               these are used instead of per-client statistics.
                               This ensures all clients scale their features
                               consistently, improving FL convergence.

    Label handling (classification only)
    -------------------------------------
    Auto-detects label range and shifts to 0-based. Pass label_offset to override.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    total_rows    = len(df)
    total_missing = df.isna().sum().sum()

    # ── Missing value handling ────────────────────────────────────────────────
    if total_missing > 0:
        feature_names_all = [col for col in df.columns if col != target_col]
        if impute == "drop":
            df = df.dropna()
            dropped = total_rows - len(df)
            warnings.warn(
                f"[load_csv_data] {os.path.basename(csv_path)}: dropped "
                f"{dropped}/{total_rows} rows ({dropped/total_rows:.1%}) due to "
                f"missing values. Consider impute='median' to retain data.",
                UserWarning, stacklevel=2,
            )
        else:
            per_col  = df[feature_names_all].isna().sum()
            affected = per_col[per_col > 0]
            warnings.warn(
                f"[load_csv_data] {os.path.basename(csv_path)}: {total_missing} "
                f"missing values in {len(affected)} column(s) — filling with "
                f"{impute}. Columns: {affected.to_dict()}",
                UserWarning, stacklevel=2,
            )
            if impute == "median":
                df[feature_names_all] = df[feature_names_all].fillna(
                    df[feature_names_all].median()
                )
            elif impute == "mean":
                df[feature_names_all] = df[feature_names_all].fillna(
                    df[feature_names_all].mean()
                )
            elif impute == "mode":
                for col in feature_names_all:
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].mode().iloc[0])
            label_missing = df[target_col].isna().sum()
            if label_missing > 0:
                warnings.warn(
                    f"[load_csv_data] {os.path.basename(csv_path)}: {label_missing} "
                    f"rows have missing labels — dropping those rows.",
                    UserWarning, stacklevel=2,
                )
                df = df.dropna(subset=[target_col])
    # ─────────────────────────────────────────────────────────────────────────

    feature_names = [col for col in df.columns if col != target_col]
    X     = df[feature_names].values.astype(np.float32)
    raw_y = df[target_col].values

    # ── Label encoding ────────────────────────────────────────────────────────
    if task == "regression":
        y          = raw_y.astype(np.float32)
        num_classes = 1
        class_names = []
    else:
        unique_raw = sorted(np.unique(raw_y).tolist())
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique_raw):
            offset      = label_offset if label_offset is not None else int(min(unique_raw))
            y           = raw_y.astype(np.int64) - offset
            num_classes = len(unique_raw)
            class_names = [f"class_{int(v)}" for v in unique_raw]
        else:
            label_map   = {name: idx for idx, name in enumerate(unique_raw)}
            y           = np.array([label_map[v] for v in raw_y], dtype=np.int64)
            num_classes = len(unique_raw)
            class_names = [str(v) for v in unique_raw]

        if y.min() < 0 or y.max() >= num_classes:
            raise ValueError(
                f"After offset correction, labels in {csv_path} are out of range "
                f"[0, {num_classes}). Got min={y.min()}, max={y.max()}."
            )
    # ─────────────────────────────────────────────────────────────────────────

    # ── Train/test split ──────────────────────────────────────────────────────
    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # ── Feature scaling ───────────────────────────────────────────────────────
    if global_mean is not None and global_std is not None:
        # Use server-coordinated global statistics — consistent across all clients
        X_tr = (X_tr - global_mean) / global_std
        X_te = (X_te - global_mean) / global_std
        print(f"[load_csv_data] Using global normalization stats for {os.path.basename(csv_path)}")
    else:
        # Per-client scaling (default) — fit on train split only
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)
    # ─────────────────────────────────────────────────────────────────────────

    if task == "regression":
        train_ds = RegressionCSVDataset(X_tr, y_tr)
        test_ds  = RegressionCSVDataset(X_te, y_te)
    else:
        train_ds = CSVDataset(X_tr, y_tr)
        test_ds  = CSVDataset(X_te, y_te)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=drop_last_for_dp,
    )
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    metadata = DatasetMetadata(
        input_dim=X_tr.shape[1],
        num_classes=num_classes,
        class_names=class_names,
        feature_names=feature_names,
        train_size=len(y_tr),
        test_size=len(y_te),
        task=task,
    )
    return train_loader, test_loader, metadata
