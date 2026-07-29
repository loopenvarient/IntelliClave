"""
data/datascripts/har_analysis.py

Exploratory analysis script for the UCI HAR dataset.

NOTE: This script is dataset-specific to UCI HAR and is used only during
the data preparation phase. It is NOT part of the FL training pipeline.
The FL pipeline (fl/, privacy/, evaluation/) is fully dataset-agnostic.

Usage:
    python data/datascripts/har_analysis.py
    python data/datascripts/har_analysis.py --dataset-dir /path/to/UCI_HAR_Dataset
    python data/datascripts/har_analysis.py --pca-components 100 --pca-threshold 0.95
"""
import argparse
import os

import numpy as np
from sklearn.decomposition import PCA

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT = os.path.join(_ROOT, "UCI HAR Dataset")

# HAR-specific class labels (6 activities)
HAR_LABELS = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def main(dataset_dir: str, pca_components: int, pca_threshold: float):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"UCI HAR Dataset not found at: {dataset_dir}\n"
            "Download from https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones\n"
            "and extract to the project root, or pass --dataset-dir."
        )

    # ── 1. Load full dataset ──────────────────────────────────────────────────
    print("=== Loading Dataset ===")
    X_train = np.loadtxt(os.path.join(dataset_dir, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(dataset_dir, "train", "y_train.txt"), dtype=int)
    X_test  = np.loadtxt(os.path.join(dataset_dir, "test",  "X_test.txt"))
    y_test  = np.loadtxt(os.path.join(dataset_dir, "test",  "y_test.txt"), dtype=int)
    s_train = np.loadtxt(os.path.join(dataset_dir, "train", "subject_train.txt"), dtype=int)
    s_test  = np.loadtxt(os.path.join(dataset_dir, "test",  "subject_test.txt"), dtype=int)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # ── 2. Class distribution ─────────────────────────────────────────────────
    print("\n=== Class Distribution (train) ===")
    counts = np.bincount(y_train - 1)
    for i, (label, count) in enumerate(zip(HAR_LABELS, counts)):
        bar = "#" * (count // 50)
        print(f"  {i+1} {label:<22} {count:>5}  {bar}")
    print(f"  Min: {counts.min()}, Max: {counts.max()}, "
          f"Balanced: {counts.max() - counts.min() < 300}")

    # ── 3. PCA check ──────────────────────────────────────────────────────────
    print(f"\n=== PCA Check ({pca_components} components) ===")
    pca = PCA(n_components=pca_components)
    pca.fit_transform(X_train)
    variance = pca.explained_variance_ratio_.sum()
    use_pca  = variance > pca_threshold
    print(f"{pca_components} PCA components explain: {variance:.4f}")
    print(f"Use PCA({pca_components}): {use_pca}  "
          f"({'variance > ' + str(pca_threshold) + ' threshold met' if use_pca else 'below threshold'})")

    # ── 4. Subject distribution ───────────────────────────────────────────────
    print("\n=== Subject Distribution ===")
    train_subjects = set(np.unique(s_train).tolist())
    test_subjects  = set(np.unique(s_test).tolist())
    test_only      = test_subjects - train_subjects
    train_only     = train_subjects - test_subjects
    overlap        = train_subjects & test_subjects

    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Test  subjects ({len(test_subjects)}):  {sorted(test_subjects)}")
    print(f"Test-only  (never in train): {sorted(test_only)}")
    print(f"Train-only (never in test):  {sorted(train_only)}")
    print(f"Overlap (in both):           {sorted(overlap)}")
    print(f"Non-IID note: "
          f"{'Fully disjoint — ideal for non-IID FL split' if not overlap else f'{len(overlap)} subjects appear in both splits'}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"Raw features:    {X_train.shape[1]}")
    print(f"PCA-{pca_components} variance: {variance:.4f}")
    print(f"Final features:  {f'{pca_components} (PCA)' if use_pca else f'{X_train.shape[1]} (raw)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exploratory analysis of the UCI HAR dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir",    default=_DEFAULT,
                        help="Path to the extracted UCI HAR Dataset directory.")
    parser.add_argument("--pca-components", type=int,   default=50,
                        help="Number of PCA components to evaluate.")
    parser.add_argument("--pca-threshold",  type=float, default=0.90,
                        help="Variance threshold for deciding whether to use PCA.")
    args = parser.parse_args()
    main(
        dataset_dir=args.dataset_dir,
        pca_components=args.pca_components,
        pca_threshold=args.pca_threshold,
    )
