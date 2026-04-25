"""
Standalone local training for one HAR client CSV before FL.

Usage:
    python fl/train_local.py --csv data/processed/client1.csv --epochs 20
    python fl/train_local.py --csv data/processed/client1.csv --epochs 20 --save results/local/client1.pth
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (  # noqa: E402
    get_default_client_csvs,
    load_class_weights,
    load_csv_data,
)
from model import get_model  # noqa: E402


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    preds_all: List[int] = []
    labels_all: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(y_batch.numpy().tolist())

    preds_arr = np.array(preds_all)
    labels_arr = np.array(labels_all)
    acc = accuracy_score(labels_arr, preds_arr)
    macro_f1 = f1_score(labels_arr, preds_arr, average="macro")
    return acc, macro_f1


def build_criterion(device: torch.device, use_class_weights: bool = True):
    class_weights = load_class_weights(device=device) if use_class_weights else None
    return nn.CrossEntropyLoss(weight=class_weights)


def train_local(
    csv_path: str,
    epochs: int = 20,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    batch_size: int = 32,
    use_class_weights: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_local] device={device} csv={csv_path}")

    train_loader, test_loader, metadata = load_csv_data(
        csv_path,
        batch_size=batch_size,
    )
    model = get_model(metadata.input_dim, metadata.num_classes).to(device)
    criterion = build_criterion(device, use_class_weights=use_class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, macro_f1 = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "loss": round(loss, 5),
                "accuracy": round(acc, 5),
                "macro_f1": round(macro_f1, 5),
            }
        )
        print(
            f"  Epoch {epoch:3d} loss={loss:.4f} "
            f"accuracy={acc:.4f} macro_f1={macro_f1:.4f}"
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        history_path = save_path.replace(".pth", "_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[train_local] Saved model -> {save_path}")
        print(f"[train_local] Saved history -> {history_path}")

    return model, history, metadata


if __name__ == "__main__":
    default_csv = get_default_client_csvs()[0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save", default=None)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights even if data/class_weights.json exists.",
    )
    args = parser.parse_args()

    _, history, metadata = train_local(
        csv_path=args.csv,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
        batch_size=args.batch_size,
        use_class_weights=not args.no_class_weights,
    )
    print(
        f"\nFinal accuracy={history[-1]['accuracy']} "
        f"macro_f1={history[-1]['macro_f1']} "
        f"input_dim={metadata.input_dim} classes={metadata.num_classes}"
    )
