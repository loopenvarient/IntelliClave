"""
security/attacks/summarize_attack_results.py

Print a compact comparison table for model inversion attack outputs.

Usage:
    python security/attacks/summarize_attack_results.py \
        --files results/attacks/model_inversion_baseline.json \
                results/attacks/model_inversion_dp.json \
                results/attacks/model_inversion_surrogate_dp.json
"""
import argparse
import json
import os
from typing import Iterable


def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", {})
    return {
        "file": os.path.basename(path),
        "mode": payload.get("mode", "unknown"),
        "avg_cosine_similarity": summary.get("avg_cosine_similarity"),
        "avg_model_confidence": summary.get("avg_model_confidence"),
        "avg_prediction_entropy": summary.get("avg_prediction_entropy"),
        "avg_prediction_sharpness": summary.get("avg_prediction_sharpness"),
        "avg_reconstruction_variance": summary.get("avg_reconstruction_variance"),
        "high_risk_classes": summary.get("high_risk_classes"),
        "verdict": summary.get("verdict"),
    }


def format_row(values: Iterable[object], widths: list[int]) -> str:
    cells = []
    for value, width in zip(values, widths):
        cells.append(str(value).ljust(width))
    return " | ".join(cells)


def main(files: list[str]) -> None:
    rows = [load_summary(path) for path in files]
    headers = [
        "file",
        "mode",
        "avg_cosine_similarity",
        "avg_model_confidence",
        "avg_prediction_entropy",
        "avg_prediction_sharpness",
        "avg_reconstruction_variance",
        "high_risk_classes",
        "verdict",
    ]
    widths = [max(len(str(row[h])) for row in rows + [{h: h}]) for h in headers]

    print(format_row(headers, widths))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row([row[h] for h in headers], widths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize model inversion results.")
    parser.add_argument("--files", nargs="+", required=True, help="Attack JSON files to compare.")
    args = parser.parse_args()
    main(args.files)
