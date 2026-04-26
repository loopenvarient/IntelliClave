import numpy as np
from sklearn.decomposition import PCA
import os

base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'UCI HAR Dataset')

# ── 1. Load full dataset ──────────────────────────────────────────────────────
print("=== Loading Dataset ===")
X_train = np.loadtxt(os.path.join(base, 'train/X_train.txt'))
y_train = np.loadtxt(os.path.join(base, 'train/y_train.txt'), dtype=int)
X_test  = np.loadtxt(os.path.join(base, 'test/X_test.txt'))
y_test  = np.loadtxt(os.path.join(base, 'test/y_test.txt'), dtype=int)
s_train = np.loadtxt(os.path.join(base, 'train/subject_train.txt'), dtype=int)
s_test  = np.loadtxt(os.path.join(base, 'test/subject_test.txt'), dtype=int)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# ── 2. Class distribution ─────────────────────────────────────────────────────
print("\n=== Class Distribution (train) ===")
counts = np.bincount(y_train - 1)
labels = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
for i, (label, count) in enumerate(zip(labels, counts)):
    bar = '#' * (count // 50)
    print(f"  {i+1} {label:<22} {count:>5}  {bar}")
print(f"  Min: {counts.min()}, Max: {counts.max()}, Balanced: {counts.max()-counts.min() < 300}")

# ── 3. PCA check ──────────────────────────────────────────────────────────────
print("\n=== PCA Check ===")
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_train)
variance = pca.explained_variance_ratio_.sum()
print(f"50 PCA components explain: {variance:.4f}")
use_pca = variance > 0.90
print(f"Use PCA(50): {use_pca}  ({'variance > 0.90 threshold met' if use_pca else 'below threshold'})")

# ── 4. Subject distribution ───────────────────────────────────────────────────
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
print(f"Non-IID note: {'Fully disjoint — ideal for non-IID FL split' if not overlap else f'{len(overlap)} subjects appear in both splits'}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Raw features:    561")
print(f"PCA-50 variance: {variance:.4f}")
print(f"Final features:  {'50 (PCA)' if use_pca else '561 (raw)'}")
