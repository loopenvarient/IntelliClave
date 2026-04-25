import numpy as np
import pandas as pd
import json
import os
from sklearn.utils.class_weight import compute_class_weight

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load labels from all 3 client CSVs ───────────────────────────────────────
print("=== Computing class weights ===")
y_all_train = []
for client in [1, 2, 3]:
    df = pd.read_csv(os.path.join(root, f'data/processed/client{client}.csv'))
    y_all_train.extend(df['label'].tolist())
    print(f"  client{client}: {len(df)} rows loaded")

y_arr = np.array(y_all_train)
print(f"\nTotal samples: {len(y_arr)}")
print(f"Class counts:  {np.bincount(y_arr - 1)}")

# compute_class_weight expects 0-indexed classes
weights = compute_class_weight('balanced', classes=np.arange(6), y=y_arr - 1)

activity_names = [
    'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
    'SITTING', 'STANDING', 'LAYING'
]

weights_dict = {name: round(float(w), 6) for name, w in zip(activity_names, weights)}

print("\nClass weights:")
for name, w in weights_dict.items():
    print(f"  {name:<25} {w:.6f}")

out_path = os.path.join(root, 'data', 'class_weights.json')
with open(out_path, 'w') as f:
    json.dump(weights_dict, f, indent=2)
print(f"\nSaved → {out_path}")

# Verify reload
with open(out_path) as f:
    loaded = json.load(f)
assert list(loaded.keys()) == activity_names
print("Reload verified ✓")
