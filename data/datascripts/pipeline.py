import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'UCI HAR Dataset')
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load data ─────────────────────────────────────────────────────────────────
print("=== Loading data ===")
X_train = np.loadtxt(os.path.join(base, 'train/X_train.txt'))
y_train = np.loadtxt(os.path.join(base, 'train/y_train.txt'), dtype=int)
X_test  = np.loadtxt(os.path.join(base, 'test/X_test.txt'))
y_test  = np.loadtxt(os.path.join(base, 'test/y_test.txt'), dtype=int)
s_train = np.loadtxt(os.path.join(base, 'train/subject_train.txt'), dtype=int)
s_test  = np.loadtxt(os.path.join(base, 'test/subject_test.txt'), dtype=int)

# ── Load fitted PCA ───────────────────────────────────────────────────────────
pca_path = os.path.join(root, 'data', 'samples', 'pca_model.pkl')
with open(pca_path, 'rb') as f:
    pca = pickle.load(f)
print(f"PCA loaded from {pca_path}")

# ── MORNING: Combine + PCA transform ─────────────────────────────────────────
print("\n=== Combining train + test ===")
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
s_all = np.concatenate([s_train, s_test])
print(f"X_all: {X_all.shape}, y_all: {y_all.shape}, s_all: {s_all.shape}")

X_all_pca = pca.transform(X_all)
print(f"X_all_pca: {X_all_pca.shape}")

# ── Split by subject & save CSVs ──────────────────────────────────────────────
print("\n=== Creating client CSVs ===")
out_dir = os.path.join(root, 'data', 'processed')
os.makedirs(out_dir, exist_ok=True)

feature_cols = [f'pca_{i}' for i in range(50)]
splits = [
    (np.isin(s_all, range(1,  11)), 'client1'),
    (np.isin(s_all, range(11, 21)), 'client2'),
    (np.isin(s_all, range(21, 31)), 'client3'),
]

for mask, name in splits:
    df = pd.DataFrame(X_all_pca[mask], columns=feature_cols)
    df['label'] = y_all[mask]
    path = os.path.join(out_dir, f'{name}.csv')
    df.to_csv(path, index=False)
    print(f"\n{name}: {len(df)} rows → {path}")
    print(df['label'].value_counts().sort_index().to_string())

# ── AFTERNOON: Verify CSVs ────────────────────────────────────────────────────
print("\n=== Verifying CSVs ===")
activity_labels = {1:'WALKING',2:'WALK_UP',3:'WALK_DN',4:'SITTING',5:'STANDING',6:'LAYING'}
all_dfs = {}

for _, name in splits:
    path = os.path.join(out_dir, f'{name}.csv')
    df = pd.read_csv(path)
    all_dfs[name] = df

    nan_count  = df.isna().sum().sum()
    inf_count  = np.isinf(df[feature_cols].values).sum()
    classes    = sorted(df['label'].unique().tolist())
    feat_count = len(feature_cols)

    print(f"\n{name}:")
    print(f"  rows={len(df)}, features={feat_count}, NaN={nan_count}, Inf={inf_count}")
    print(f"  classes present: {classes}  ({'OK' if classes == [1,2,3,4,5,6] else 'MISSING CLASSES'})")
    assert nan_count == 0,  f"{name} has NaN values!"
    assert inf_count == 0,  f"{name} has Inf values!"
    assert classes == [1,2,3,4,5,6], f"{name} missing classes!"
    assert feat_count == 50, f"{name} wrong feature count!"
    print(f"  All checks passed ✓")

# ── Plot class distributions ──────────────────────────────────────────────────
print("\n=== Saving class distribution chart ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']

for ax, (_, name) in zip(axes, splits):
    df = all_dfs[name]
    counts = df['label'].value_counts().sort_index()
    bars = ax.bar([activity_labels[i] for i in counts.index], counts.values, color=colors)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Activity')
    ax.set_ylabel('Sample count')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=8)

plt.suptitle('Class Distribution per FL Client', fontsize=15, fontweight='bold')
plt.tight_layout()
chart_path = os.path.join(root, 'data', 'client_distributions.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"Chart saved → {chart_path}")

print("\n=== FREEZE: CSVs are final — do not modify ===")
print("data/processed/client1.csv")
print("data/processed/client2.csv")
print("data/processed/client3.csv")
