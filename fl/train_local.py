"""
Standalone local training for one client CSV before FL.

Usage:
    python fl/train_local.py --csv data/processed/client1.csv --epochs 20
    python fl/train_local.py --csv data/processed/client1.csv --epochs 20 --save results/local/client1.pth

    # DP mode:
    python fl/train_local.py --csv data/processed/client1.csv --epochs 5 --dp
    python fl/train_local.py --csv data/processed/client1.csv --epochs 5 --dp --epsilon 5.0
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import LABEL_SMOOTHING  # noqa: E402
from config.constants import CONFIDENCE_PENALTY  # noqa: E402
from config.constants import DEFAULT_EPSILON  # noqa: E402
from data_utils import (  # noqa: E402
    compute_client_scaler_stats,
    get_default_client_csvs,
    get_preprocessing_metadata_path,
    load_class_weights,
    load_csv_data,
    save_preprocessing_metadata,
)
from model import get_model  # noqa: E402


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    confidence_penalty: float = CONFIDENCE_PENALTY,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        if confidence_penalty > 0:
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            loss = loss - confidence_penalty * entropy
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


def build_criterion(device: torch.device, num_classes: int, use_class_weights: bool = True):
    class_weights = (
        load_class_weights(num_classes=num_classes, device=device)
        if use_class_weights
        else None
    )
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )


def train_local(
    csv_path: str,
    epochs: int = 20,
    lr: float = 1e-3,
    save_path=None,
    batch_size: int = 32,
    use_class_weights: bool = True,
    use_dp: bool = False,
    target_epsilon: float = DEFAULT_EPSILON,
    max_grad_norm: float = 0.3,
    model_type: str = "mlp",
    weight_decay: float = 1e-4,
    confidence_penalty: float = CONFIDENCE_PENALTY,
    early_stopping_patience: int = 3,
    early_stopping_metric: str = "macro_f1",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_local] device={device} csv={csv_path}")

    train_loader, test_loader, train_metadata = load_csv_data(
        csv_path,
        batch_size=batch_size,
        drop_last_for_dp=use_dp,
    )
    model = get_model(train_metadata.input_dim, train_metadata.num_classes, model_type=model_type).to(device)
    criterion = build_criterion(device, num_classes=train_metadata.num_classes, use_class_weights=use_class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Attach Opacus PrivacyEngine if DP mode is enabled
    privacy_engine = None
    if use_dp:
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator

            # Auto-fix any BatchNorm → GroupNorm (HARClassifier uses ReLU only,
            # so this is a safety net — it will report no changes needed)
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                print(f"[DP] Auto-fixing model: {errors}")
                model = ModuleValidator.fix(model)

            # delta = 1 / number of training samples (standard DP rule)
            target_delta = 1.0 / train_metadata.train_size
            print(
                f"[DP] Attaching PrivacyEngine: "
                f"ε={target_epsilon}, δ={target_delta:.2e}, "
                f"max_grad_norm={max_grad_norm}, "
                f"train_size={train_metadata.train_size}"
            )

            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
            )
            print("[DP] PrivacyEngine attached successfully")

        except ImportError:
            print("[DP] WARNING: Opacus not installed. Running without DP.")
            use_dp = False

    training_history: List[Dict] = []
    best_metric = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            confidence_penalty=confidence_penalty,
        )
        acc, macro_f1 = evaluate(model, test_loader, device)

        epoch_entry = {
            "epoch": epoch,
            "loss": round(loss, 5),
            "accuracy": round(acc, 5),
            "macro_f1": round(macro_f1, 5),
        }
        if use_dp and privacy_engine is not None:
            eps = privacy_engine.get_epsilon(delta=1.0 / train_metadata.train_size)
            epoch_entry["epsilon"] = round(eps, 5)
            epoch_entry["delta"] = round(1.0 / train_metadata.train_size, 8)

        training_history.append(epoch_entry)

        if use_dp and "epsilon" in epoch_entry:
            print(
                f"  Epoch {epoch:3d} loss={loss:.4f} "
                f"accuracy={acc:.4f} macro_f1={macro_f1:.4f} "
                f"ε={epoch_entry['epsilon']:.4f}"
            )
        else:
            print(
                f"  Epoch {epoch:3d} loss={loss:.4f} "
                f"accuracy={acc:.4f} macro_f1={macro_f1:.4f}"
            )

        monitored = epoch_entry.get(early_stopping_metric)
        if early_stopping_patience > 0 and monitored is not None:
            improved = False
            if early_stopping_metric == "loss":
                improved = best_metric is None or monitored < best_metric - 1e-4
            else:
                improved = best_metric is None or monitored > best_metric + 1e-4

            if improved:
                best_metric = monitored
                stale_epochs = 0
            else:
                stale_epochs += 1
                print(
                    f"[train_local][early_stopping] no improvement for "
                    f"{stale_epochs}/{early_stopping_patience} epochs "
                    f"on {early_stopping_metric}={monitored:.5f}"
                )
                if stale_epochs >= early_stopping_patience:
                    print(
                        f"[train_local][early_stopping] stopping at epoch {epoch} "
                        f"(best {early_stopping_metric}={best_metric:.5f})"
                    )
                    break

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        history_path = save_path.replace(".pth", "_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2)
        
        # Save preprocessing stats from the same train split used by load_csv_data()
        scaler_stats = compute_client_scaler_stats(csv_path)
        save_preprocessing_metadata(
            save_path,
            feature_names=train_metadata.feature_names,
            mean=scaler_stats.mean,
            std=scaler_stats.std,
            normalization="per_client",
        )

        print(f"[train_local] Saved model -> {save_path}")
        print(f"[train_local] Saved history -> {history_path}")
        print(
            f"[train_local] Saved preprocessing -> "
            f"{get_preprocessing_metadata_path(save_path)}"
        )

    if use_dp and privacy_engine is not None:
        final_eps = privacy_engine.get_epsilon(delta=1.0 / train_metadata.train_size)
        print(f"\n[DP] Final privacy budget consumed: ε={final_eps:.4f}")
        print(f"[DP] Accuracy with DP: {training_history[-1]['accuracy']:.4f}")
        print(f"[DP] Macro-F1 with DP: {training_history[-1]['macro_f1']:.4f}")

    return model, training_history, train_metadata


if __name__ == "__main__":
    default_csv = get_default_client_csvs()[0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save", default=None)
    parser.add_argument("--model-type", default="mlp",
                        choices=["mlp", "resnet-tabular", "transformer-tabular"],
                        help="Model architecture (default: mlp).")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weights even if data/class_weights.json exists.")
    parser.add_argument("--dp", action="store_true",
                        help="Enable Differential Privacy training via Opacus.")
    parser.add_argument("--epsilon", type=float, default=1.5,
                        help="Target epsilon (privacy budget). Default=1.0.")
    parser.add_argument("--max-grad-norm", type=float, default=0.3,
                        help="Gradient clipping threshold for DP-SGD. Default=0.3.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Adam weight decay for regularization. Default=1e-4.")
    parser.add_argument("--confidence-penalty", type=float, default=CONFIDENCE_PENALTY,
                        help="Entropy regularization strength. Default=0.01.")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop local retraining after N stagnant epochs. Default=3.")
    parser.add_argument("--early-stopping-metric", default="macro_f1",
                        choices=["loss", "accuracy", "macro_f1"],
                        help="Metric to monitor for early stopping. Default=macro_f1.")

    args = parser.parse_args()

    _, history, metadata = train_local(
        csv_path=args.csv,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
        batch_size=args.batch_size,
        use_class_weights=not args.no_class_weights,
        use_dp=args.dp,
        target_epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        model_type=args.model_type,
        weight_decay=args.weight_decay,
        confidence_penalty=args.confidence_penalty,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
    )
    print(
        f"\nFinal accuracy={history[-1]['accuracy']} "
        f"macro_f1={history[-1]['macro_f1']} "
        f"input_dim={metadata.input_dim} classes={metadata.num_classes}"
    )
