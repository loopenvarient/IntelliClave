import numpy as np
import pandas as pd
import pickle
import json
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(root, 'data')

print("=" * 60)
print("DATA INTEGRITY CHECK")
print("=" * 60)

# ── 1. Client CSVs ────────────────────────────────────────────────────────────
print("\n[1] Client CSVs")
feature_cols = [f'pca_{i}' for i in range(50)]
expected_rows = {'client1': 3105, 'client2': 3426, 'client3': 3768}
all_ok = True

for name, exp_rows in expected_rows.items():
    path = os.path.join(data_dir, 'processed', f'{name}.csv')
    exists = os.path.exists(path)
    if not exists:
        print(f"  MISSING: {path}")
        all_ok = False
        continue
    df = pd.read_csv(path)
    nan   = df.isna().sum().sum()
    inf   = np.isinf(df[feature_cols].values).sum()
    cols  = len([c for c in df.columns if c.startswith('pca_')])
    classes = sorted(df['label'].unique().tolist())
    rows_ok = len(df) == exp_rows
    print(f"  {name}: rows={len(df)} (expected {exp_rows}) {'✓' if rows_ok else '✗'} | "
          f"features={cols} {'✓' if cols==50 else '✗'} | "
          f"NaN={nan} {'✓' if nan==0 else '✗'} | "
          f"Inf={inf} {'✓' if inf==0 else '✗'} | "
          f"classes={classes} {'✓' if classes==[1,2,3,4,5,6] else '✗'}")
    if not (rows_ok and cols==50 and nan==0 and inf==0 and classes==[1,2,3,4,5,6]):
        all_ok = False

# ── 2. PCA model ──────────────────────────────────────────────────────────────
print("\n[2] PCA Model")
pca_path = os.path.join(data_dir, 'samples', 'pca_model.pkl')
if os.path.exists(pca_path):
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    print(f"  pca_model.pkl: n_components={pca.n_components_} {'✓' if pca.n_components_==50 else '✗'} | "
          f"variance={pca.explained_variance_ratio_.sum():.4f} {'✓' if pca.explained_variance_ratio_.sum()>0.90 else '✗'}")
else:
    print("  MISSING: pca_model.pkl")
    all_ok = False

# ── 3. Class weights ──────────────────────────────────────────────────────────
print("\n[3] Class Weights")
cw_path = os.path.join(data_dir, 'class_weights.json')
expected_keys = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
if os.path.exists(cw_path):
    with open(cw_path) as f:
        cw = json.load(f)
    keys_ok = list(cw.keys()) == expected_keys
    vals_ok  = all(0.5 < v < 2.0 for v in cw.values())
    print(f"  class_weights.json: keys={'✓' if keys_ok else '✗'} | values_in_range={'✓' if vals_ok else '✗'}")
    for k, v in cw.items():
        print(f"    {k:<25} {v:.6f}")
else:
    print("  MISSING: class_weights.json")
    all_ok = False

# ── 4. Artefacts ──────────────────────────────────────────────────────────────
print("\n[4] Other Artefacts")
for f in ['client_distributions.png', 'deployment_scenario.md']:
    path = os.path.join(data_dir, f)
    print(f"  {f}: {'✓ exists' if os.path.exists(path) else '✗ MISSING'}")

# ── 5. Script inventory ───────────────────────────────────────────────────────
print("\n[5] Scripts in data/datascripts/")
scripts_dir = os.path.join(data_dir, 'datascripts')
for f in sorted(os.listdir(scripts_dir)):
    print(f"  {f}")

# ── 6. Path correctness check in scripts ─────────────────────────────────────
print("\n[6] Path correctness in scripts")
issues = []
for script in ['har_analysis.py', 'matrix_analysis.py', 'pipeline.py', 'weights.py']:
    path = os.path.join(scripts_dir, script)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        src = f.read()
    # These scripts still reference UCI HAR Dataset paths via base = dirname(__file__)
    # but they now live in data/datascripts — flag if they use train/ or test/ directly
    # broken if base = dirname(__file__) without pointing to UCI HAR Dataset
    if ("base = os.path.dirname(os.path.abspath(__file__))" in src and
            ("os.path.join(base, 'train/" in src or "os.path.join(base, 'test/" in src)):
        issues.append(f"  ✗ {script}: base points to script dir, not UCI HAR Dataset — paths broken!")
    else:
        print(f"  ✓ {script}: paths look correct")

for issue in issues:
    print(issue)
    all_ok = False

print("\n" + "=" * 60)
print(f"OVERALL: {'ALL CHECKS PASSED ✓' if all_ok else 'ISSUES FOUND — see above ✗'}")
print("=" * 60)
