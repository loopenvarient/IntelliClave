import numpy as np
from sklearn.decomposition import PCA
import pickle
import os

base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'UCI HAR Dataset')
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load data ─────────────────────────────────────────────────────────────────
X_train = np.loadtxt(os.path.join(base, 'train/X_train.txt'))
y_train = np.loadtxt(os.path.join(base, 'train/y_train.txt'), dtype=int)
X_test  = np.loadtxt(os.path.join(base, 'test/X_test.txt'))
y_test  = np.loadtxt(os.path.join(base, 'test/y_test.txt'), dtype=int)
s_train = np.loadtxt(os.path.join(base, 'train/subject_train.txt'), dtype=int)
s_test  = np.loadtxt(os.path.join(base, 'test/subject_test.txt'), dtype=int)

labels = ['WALK','WALK_UP','WALK_DN','SIT','STAND','LAY']

# ── MORNING: Subject × Class matrix ──────────────────────────────────────────
print("=== Subject × Class Matrix (train subjects only) ===")
print(f"{'Subj':>5} | " + " | ".join(f"{l:>7}" for l in labels) + " | TOTAL")
print("-" * 75)

subject_class_matrix = {}
train_subjects = sorted(np.unique(s_train).tolist())

for subject in range(1, 31):
    mask = (s_train == subject)
    if mask.sum() == 0:
        subject_class_matrix[subject] = np.zeros(6, dtype=int)
        print(f"  {subject:>3} | {'(test-only subject)':>50}")
        continue
    counts = np.bincount(y_train[mask] - 1, minlength=6)
    subject_class_matrix[subject] = counts
    dominant = labels[np.argmax(counts)]
    print(f"  {subject:>3} | " + " | ".join(f"{c:>7}" for c in counts) + f" | {mask.sum():>5}  [{dominant}]")

# ── Non-IID split summary ─────────────────────────────────────────────────────
print("\n=== Non-IID Client Split (train subjects) ===")
clients = {
    'Client 1 (subj 1-10)':  [s for s in range(1,  11) if s in train_subjects],
    'Client 2 (subj 11-20)': [s for s in range(11, 21) if s in train_subjects],
    'Client 3 (subj 21-30)': [s for s in range(21, 31) if s in train_subjects],
}
for client, subjects in clients.items():
    mask = np.isin(s_train, subjects)
    counts = np.bincount(y_train[mask] - 1, minlength=6)
    print(f"\n  {client}  subjects={subjects}  total={mask.sum()}")
    for i, (l, c) in enumerate(zip(labels, counts)):
        pct = c / mask.sum() * 100
        print(f"    {l:<10} {c:>5}  ({pct:.1f}%)")

# ── AFTERNOON: PCA fitting ────────────────────────────────────────────────────
print("\n=== PCA Fitting (X_train only — no leakage) ===")
pca = PCA(n_components=50)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)

print(f"X_train_pca shape: {X_train_pca.shape}  (expected (7352, 50))")
print(f"X_test_pca  shape: {X_test_pca.shape}   (expected (2947, 50))")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Save PCA model
out_dir = os.path.join(root, 'data', 'samples')
os.makedirs(out_dir, exist_ok=True)
pca_path = os.path.join(out_dir, 'pca_model.pkl')
with open(pca_path, 'wb') as f:
    pickle.dump(pca, f)
print(f"PCA model saved → {pca_path}")

# Verify reload
with open(pca_path, 'rb') as f:
    pca_loaded = pickle.load(f)
assert pca_loaded.n_components_ == 50
print("PCA reload verified ✓")
