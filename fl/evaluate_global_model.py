import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import get_default_client_csvs, load_class_weights, load_csv_data  # noqa: E402
from model import get_model  # noqa: E402
from train_local import evaluate  # noqa: E402


def evaluate_checkpoint(
    checkpoint_path: str,
    csv_paths: List[str],
    batch_size: int = 64,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, metadata = load_csv_data(csv_paths[0], batch_size=batch_size)

    model = get_model(metadata.input_dim, metadata.num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    criterion = nn.CrossEntropyLoss(weight=load_class_weights(device=device))

    results = []
    total_examples = 0
    weighted_loss = 0.0
    weighted_accuracy = 0.0
    weighted_macro_f1 = 0.0

    for csv_path in csv_paths:
        _, test_loader, client_metadata = load_csv_data(csv_path, batch_size=batch_size)

        total_loss = 0.0
        n_examples = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch.to(device))
                loss = criterion(logits, y_batch.to(device))
                total_loss += loss.item() * len(y_batch)
                n_examples += len(y_batch)

        accuracy, macro_f1 = evaluate(model, test_loader, device)
        avg_loss = total_loss / n_examples

        results.append(
            {
                "client_csv": csv_path,
                "test_size": client_metadata.test_size,
                "loss": round(avg_loss, 5),
                "accuracy": round(accuracy, 5),
                "macro_f1": round(macro_f1, 5),
            }
        )

        total_examples += n_examples
        weighted_loss += avg_loss * n_examples
        weighted_accuracy += accuracy * n_examples
        weighted_macro_f1 += macro_f1 * n_examples

    summary = {
        "checkpoint": checkpoint_path,
        "per_client": results,
        "overall_weighted": {
            "loss": round(weighted_loss / total_examples, 5),
            "accuracy": round(weighted_accuracy / total_examples, 5),
            "macro_f1": round(weighted_macro_f1 / total_examples, 5),
        },
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the saved global FL model on the HAR client CSVs."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("results", "fl_rounds", "global_model_latest.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output",
        default=os.path.join("results", "fl_rounds", "global_model_eval.json"),
    )
    args = parser.parse_args()

    csv_paths = get_default_client_csvs()
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        csv_paths=csv_paths,
        batch_size=args.batch_size,
    )

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Checkpoint: {summary['checkpoint']}")
    for item in summary["per_client"]:
        print(
            f"{os.path.basename(item['client_csv'])}: "
            f"loss={item['loss']:.5f} "
            f"accuracy={item['accuracy']:.5f} "
            f"macro_f1={item['macro_f1']:.5f}"
        )
    overall = summary["overall_weighted"]
    print(
        f"Overall weighted: loss={overall['loss']:.5f} "
        f"accuracy={overall['accuracy']:.5f} "
        f"macro_f1={overall['macro_f1']:.5f}"
    )
    print(f"Saved -> {args.output}")
