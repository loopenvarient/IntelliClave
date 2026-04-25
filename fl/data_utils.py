import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

NUM_CLASSES = len(ACTIVITY_NAMES)


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


@dataclass
class DatasetMetadata:
    input_dim: int
    num_classes: int
    class_names: List[str]
    feature_names: List[str]
    train_size: int
    test_size: int


class CSVDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def infer_csv_schema(csv_path: str, target_col: str = "label") -> Tuple[int, List[str]]:
    df = pd.read_csv(csv_path, nrows=1)
    feature_names = [col for col in df.columns if col != target_col]
    if not feature_names:
        raise ValueError(f"No feature columns found in {csv_path}")
    return len(feature_names), feature_names


def load_class_weights(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    weights_path = get_class_weights_path()
    if not os.path.exists(weights_path):
        return None

    with open(weights_path) as f:
        weights_dict = json.load(f)

    weights = [weights_dict[name] for name in ACTIVITY_NAMES]
    tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_csv_data(
    csv_path: str,
    target_col: str = "label",
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
):
    """
    Load a HAR client CSV, scale features, stratify split, and return dataloaders.

    Labels in the source CSVs are 1..6. They are converted to 0..5 for PyTorch.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).dropna()
    feature_names = [col for col in df.columns if col != target_col]

    X = df[feature_names].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    unique_labels = sorted(np.unique(y).tolist())
    expected_labels = list(range(1, NUM_CLASSES + 1))
    if unique_labels != expected_labels:
        raise ValueError(
            f"Unexpected labels in {csv_path}: {unique_labels}. "
            f"Expected {expected_labels}."
        )

    y = y - 1

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    train_loader = DataLoader(
        CSVDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        CSVDataset(X_te, y_te),
        batch_size=batch_size,
        shuffle=False,
    )

    metadata = DatasetMetadata(
        input_dim=X_tr.shape[1],
        num_classes=NUM_CLASSES,
        class_names=ACTIVITY_NAMES,
        feature_names=feature_names,
        train_size=len(y_tr),
        test_size=len(y_te),
    )
    return train_loader, test_loader, metadata
